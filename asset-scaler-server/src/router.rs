//! Loopback HTTP transport.

use std::{net::IpAddr, sync::Arc};

use axum::{
    Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Query, Request, State},
    http::{HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::processing::{
    AssetMaskMode, AssetProcessingError, AssetProcessingOptions, AssetProcessor, MAX_UPLOAD_BYTES,
};

#[derive(Clone)]
struct AppState {
    processor: AssetProcessor,
    admission: Arc<Semaphore>,
}

#[derive(Debug, Deserialize)]
struct ProcessQuery {
    width: Option<u32>,
    height: Option<u32>,
    aa: Option<u8>,
    feather: Option<u8>,
    mask: Option<AssetMaskMode>,
}

/// Builds the single-image service router. Only one image job is admitted;
/// concurrent uploads get HTTP 429 and are not queued.
pub fn router(processor: AssetProcessor) -> Router {
    let state = AppState {
        processor,
        admission: Arc::new(Semaphore::new(1)),
    };
    Router::new()
        .route("/health", get(health))
        .route("/process", post(process).options(options))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        .layer(middleware::from_fn(cors))
        .with_state(state)
}

async fn health() -> Response {
    (StatusCode::OK, "ok").into_response()
}
async fn options() -> Response {
    StatusCode::NO_CONTENT.into_response()
}

async fn process(
    State(state): State<AppState>,
    Query(query): Query<ProcessQuery>,
    body: Bytes,
) -> Response {
    let options = match target_options(query) {
        Ok(options) => options,
        Err(message) => return (StatusCode::BAD_REQUEST, message).into_response(),
    };
    let Ok(permit) = state.admission.clone().try_acquire_owned() else {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            "another asset is processing; retry after it completes",
        )
            .into_response();
    };
    // The task owns admission and temporary storage until all blocking work
    // drains, even when an HTTP client disconnects.
    let job = tokio::spawn(async move {
        let _permit = permit;
        let cancellation = CancellationToken::new();
        state
            .processor
            .process_with_options(&body, options, &cancellation)
            .await
    });
    match job.await {
        Ok(Ok(png)) => ([(header::CONTENT_TYPE, "image/png")], png).into_response(),
        Ok(Err(error)) => asset_error_response(error),
        Err(error) => (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("asset processing worker stopped: {error}"),
        )
            .into_response(),
    }
}

fn target_options(query: ProcessQuery) -> Result<AssetProcessingOptions, &'static str> {
    let (width, height) = match (query.width, query.height) {
        (None, None) => (200, 200),
        (Some(width), Some(height)) => (width, height),
        _ => return Err("width and height must be provided together"),
    };
    let options = AssetProcessingOptions {
        width,
        height,
        aa_percent: query
            .aa
            .unwrap_or(AssetProcessingOptions::default().aa_percent),
        mask_feather_percent: query
            .feather
            .unwrap_or(AssetProcessingOptions::default().mask_feather_percent),
        mask_mode: query.mask.unwrap_or_default(),
    };
    options.validate()?;
    Ok(options)
}

fn asset_error_response(error: AssetProcessingError) -> Response {
    let status = match error {
        AssetProcessingError::EmptyUpload
        | AssetProcessingError::UploadTooLarge
        | AssetProcessingError::InvalidImage(_)
        | AssetProcessingError::SourceTooLarge => StatusCode::BAD_REQUEST,
        AssetProcessingError::Processing(_) | AssetProcessingError::TemporaryStorage(_) => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
        AssetProcessingError::BackgroundRemoval(_) => StatusCode::SERVICE_UNAVAILABLE,
    };
    (status, error.to_string()).into_response()
}

async fn cors(request: Request, next: Next) -> Response {
    let origin = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|value| value.to_str().ok())
        .filter(|origin| is_loopback_origin(origin))
        .map(str::to_owned);
    let mut response = next.run(request).await;
    let Some(origin) = origin else {
        return response;
    };
    let headers = response.headers_mut();
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        HeaderValue::from_str(&origin).expect("origin came from a valid request header"),
    );
    headers.insert(header::VARY, HeaderValue::from_static("Origin"));
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("POST, OPTIONS"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("content-type"),
    );
    response
}

fn is_loopback_origin(origin: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(origin) else {
        return false;
    };
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return false;
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

#[cfg(test)]
mod tests {
    use super::*;
    use asset_background_removal::InspyReNet;
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use std::path::PathBuf;
    use tower::ServiceExt;

    fn inactive_processor() -> AssetProcessor {
        AssetProcessor::new(InspyReNet {
            python: PathBuf::from("unused-in-test"),
            checkpoint: PathBuf::from("unused-in-test"),
            device: "cpu".into(),
        })
    }

    #[test]
    fn cors_accepts_only_loopback_origins() {
        for origin in [
            "http://localhost:5173",
            "https://127.0.0.1:3000",
            "http://[::1]:8080",
        ] {
            assert!(is_loopback_origin(origin), "{origin}");
        }
        for origin in [
            "https://example.com",
            "file://",
            "http://127.0.0.1.example.com",
            "http://localhost:80@evil.example",
            "http://localhost/path",
            "http://localhost/?query",
        ] {
            assert!(!is_loopback_origin(origin), "{origin}");
        }
    }

    #[tokio::test]
    async fn health_and_oversize_keep_loopback_cors_headers() {
        let origin = "http://localhost:5173";
        let health = router(inactive_processor())
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header(header::ORIGIN, origin)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        assert_eq!(
            health
                .headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .unwrap(),
            origin
        );
        assert_eq!(to_bytes(health.into_body(), 1024).await.unwrap(), "ok");
        let oversize = router(inactive_processor())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/process")
                    .header(header::ORIGIN, origin)
                    .body(Body::from(vec![0; MAX_UPLOAD_BYTES + 1]))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(oversize.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            oversize
                .headers()
                .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                .unwrap(),
            origin
        );
    }

    #[tokio::test]
    async fn rejects_a_busy_request() {
        let state = AppState {
            processor: inactive_processor(),
            admission: Arc::new(Semaphore::new(1)),
        };
        let permit = state.admission.clone().try_acquire_owned().unwrap();
        let response = process(
            State(state),
            Query(ProcessQuery {
                width: None,
                height: None,
                aa: None,
                feather: None,
                mask: None,
            }),
            Bytes::new(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        drop(permit);
    }

    #[test]
    fn target_options_default_and_validate_all_controls() {
        assert_eq!(
            target_options(ProcessQuery {
                width: None,
                height: None,
                aa: None,
                feather: None,
                mask: None
            })
            .unwrap(),
            AssetProcessingOptions::default()
        );
        assert_eq!(
            target_options(ProcessQuery {
                width: Some(128),
                height: Some(256),
                aa: Some(0),
                feather: Some(100),
                mask: Some(AssetMaskMode::EdgeMatte)
            })
            .unwrap(),
            AssetProcessingOptions {
                width: 128,
                height: 256,
                aa_percent: 0,
                mask_feather_percent: 100,
                mask_mode: AssetMaskMode::EdgeMatte
            }
        );
        for query in [
            ProcessQuery {
                width: Some(128),
                height: None,
                aa: None,
                feather: None,
                mask: None,
            },
            ProcessQuery {
                width: None,
                height: Some(256),
                aa: None,
                feather: None,
                mask: None,
            },
            ProcessQuery {
                width: Some(7),
                height: Some(256),
                aa: None,
                feather: None,
                mask: None,
            },
            ProcessQuery {
                width: Some(128),
                height: Some(513),
                aa: None,
                feather: None,
                mask: None,
            },
            ProcessQuery {
                width: None,
                height: None,
                aa: Some(101),
                feather: None,
                mask: None,
            },
            ProcessQuery {
                width: None,
                height: None,
                aa: None,
                feather: Some(101),
                mask: None,
            },
        ] {
            assert!(target_options(query).is_err());
        }
    }

    #[tokio::test]
    async fn query_rejection_precedes_runtime_and_edge_matte_bypasses_it() {
        let mut png = Vec::new();
        image::RgbaImage::new(1, 1)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        for uri in [
            "/process?width=128",
            "/process?height=256",
            "/process?width=7&height=256",
            "/process?aa=101",
            "/process?feather=101",
            "/process?mask=not-a-mask",
        ] {
            let response = router(inactive_processor())
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(uri)
                        .body(Body::from(png.clone()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{uri}");
        }
        let response = router(inactive_processor())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/process?mask=edge-matte")
                    .body(Body::from(png))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
