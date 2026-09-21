//! Single-image background removal and fixed-canvas scaling.

use std::{collections::VecDeque, io::Cursor, path::Path, sync::Arc};

use asset_background_removal::{InspyReNet, MaskFrame, apply_mask_to_rgba_with_feather};
use asset_scaler::GameAssetAa;
use image::{ImageBuffer, ImageFormat, ImageReader, Limits, Rgba, RgbaImage, imageops};
use thiserror::Error;
use tokio::sync::{OnceCell, mpsc};
use tokio_util::sync::CancellationToken;

pub const MAX_UPLOAD_BYTES: usize = 8 * 1024 * 1024;
const MAX_SOURCE_WIDTH: u32 = 3_840;
const MAX_SOURCE_HEIGHT: u32 = 2_160;
const MAX_DECODE_BYTES: u64 = 64 * 1024 * 1024;
const CORE_SCALER_MEMORY_LIMIT: u64 = 1024 * 1024 * 1024;
const DEFAULT_MODEL_MEMORY_LIMIT: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AssetMaskMode {
    #[default]
    Model,
    EdgeMatte,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssetProcessingOptions {
    pub width: u32,
    pub height: u32,
    pub aa_percent: u8,
    pub mask_feather_percent: u8,
    pub mask_mode: AssetMaskMode,
}

impl Default for AssetProcessingOptions {
    fn default() -> Self {
        Self {
            width: 200,
            height: 200,
            aa_percent: 20,
            mask_feather_percent: 0,
            mask_mode: AssetMaskMode::Model,
        }
    }
}

impl AssetProcessingOptions {
    pub fn validate(self) -> Result<(), &'static str> {
        if !(8..=512).contains(&self.width) || !(8..=512).contains(&self.height) {
            return Err("width and height must each be between 8 and 512 pixels");
        }
        if self.aa_percent > 100 {
            return Err("aa must be an integer from 0 through 100");
        }
        if self.mask_feather_percent > 100 {
            return Err("feather must be an integer from 0 through 100");
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct AssetProcessor {
    runtime: InspyReNet,
    runtime_identity: Arc<OnceCell<String>>,
}

#[derive(Debug, Error)]
pub enum AssetProcessingError {
    #[error("The upload is empty.")]
    EmptyUpload,
    #[error("The upload exceeds the {MAX_UPLOAD_BYTES} byte limit.")]
    UploadTooLarge,
    #[error("The upload is not a supported PNG, JPEG, or WebP image: {0}")]
    InvalidImage(String),
    #[error("The source image dimensions exceed {MAX_SOURCE_WIDTH} × {MAX_SOURCE_HEIGHT} pixels.")]
    SourceTooLarge,
    #[error("InSPyReNet: {0}")]
    BackgroundRemoval(String),
    #[error("Image processing: {0}")]
    Processing(String),
    #[error("Temporary asset processing storage: {0}")]
    TemporaryStorage(#[from] std::io::Error),
}

impl AssetProcessor {
    pub fn new(runtime: InspyReNet) -> Self {
        Self {
            runtime,
            runtime_identity: Arc::new(OnceCell::new()),
        }
    }

    pub fn from_environment() -> Result<Self, AssetProcessingError> {
        InspyReNet::from_environment()
            .map(Self::new)
            .map_err(AssetProcessingError::BackgroundRemoval)
    }

    pub async fn warm(&self, cancellation: &CancellationToken) -> Result<(), AssetProcessingError> {
        let identity = self.runtime_identity(cancellation).await?;
        self.runtime
            .validate_runtime_with_identity(cancellation, identity)
            .await
            .map_err(AssetProcessingError::BackgroundRemoval)
    }

    pub async fn process(
        &self,
        upload: &[u8],
        cancellation: &CancellationToken,
    ) -> Result<Vec<u8>, AssetProcessingError> {
        self.process_with_options(upload, AssetProcessingOptions::default(), cancellation)
            .await
    }

    /// The accepted request owns its temporary directory until the model
    /// producer and scaler have both drained, including on cancellation.
    /// To cancel safely, cancel the supplied token and await this future;
    /// dropping it alone cannot drain already-started blocking work.
    pub async fn process_with_options(
        &self,
        upload: &[u8],
        options: AssetProcessingOptions,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u8>, AssetProcessingError> {
        options
            .validate()
            .map_err(|message| AssetProcessingError::Processing(message.into()))?;
        validate_upload(upload)?;
        check_cancelled(cancellation)?;
        let upload = upload.to_vec();
        let source = tokio::task::spawn_blocking(move || normalize_upload(&upload))
            .await
            .map_err(|error| AssetProcessingError::Processing(error.to_string()))??;
        check_cancelled(cancellation)?;
        if options.mask_mode == AssetMaskMode::EdgeMatte {
            let source = tokio::task::spawn_blocking(move || edge_connected_matte(source))
                .await
                .map_err(|error| AssetProcessingError::Processing(error.to_string()))?;
            let token = cancellation.clone();
            let scaled = tokio::task::spawn_blocking(move || scale(&source, options, &token))
                .await
                .map_err(|error| AssetProcessingError::Processing(error.to_string()))??;
            check_cancelled(cancellation)?;
            return encode_rgba_png(&scaled);
        }
        validate_model_memory(source.dimensions(), options)?;

        let directory = tempfile::tempdir()?;
        let source_path = directory.path().join("source.png");
        tokio::task::spawn_blocking(move || write_rgba_png(&source_path, &source))
            .await
            .map_err(|error| AssetProcessingError::Processing(error.to_string()))??;
        let sources = vec![directory.path().join("source.png")];
        let (sender, mut receiver) = mpsc::channel(1);
        let work_cancel = cancellation.child_token();
        let identity = self.runtime_identity(&work_cancel).await?;
        let producer = self.runtime.stream_masks_with_identity(
            &sources,
            directory.path(),
            &work_cancel,
            sender,
            identity,
        );
        let consumer_cancel = work_cancel.clone();
        let consumer = async {
            let frame = tokio::select! {
                _ = consumer_cancel.cancelled() => return Err(AssetProcessingError::Processing("Downscaling was cancelled.".into())),
                frame = receiver.recv() => frame.ok_or_else(|| AssetProcessingError::Processing("InSPyReNet did not emit a mask.".into()))?,
            };
            process_mask(frame, options, consumer_cancel).await
        };
        tokio::pin!(producer);
        tokio::pin!(consumer);
        let png = tokio::select! {
            produced = &mut producer => match produced {
                Ok(()) => consumer.await,
                Err(error) => {
                    work_cancel.cancel();
                    let _ = consumer.await;
                    Err(AssetProcessingError::BackgroundRemoval(error))
                }
            },
            scaled = &mut consumer => match scaled {
                Ok(png) => match producer.await {
                    Ok(()) => Ok(png),
                    Err(error) => Err(AssetProcessingError::BackgroundRemoval(error)),
                },
                Err(error) => {
                    work_cancel.cancel();
                    let _ = producer.await;
                    Err(error)
                }
            },
        }?;
        check_cancelled(cancellation)?;
        Ok(png)
    }

    async fn runtime_identity(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<&str, AssetProcessingError> {
        self.runtime_identity
            .get_or_try_init(|| self.runtime.cache_identity(cancellation))
            .await
            .map(String::as_str)
            .map_err(AssetProcessingError::BackgroundRemoval)
    }
}

async fn process_mask(
    frame: MaskFrame,
    options: AssetProcessingOptions,
    cancellation: CancellationToken,
) -> Result<Vec<u8>, AssetProcessingError> {
    let worker_cancellation = cancellation.clone();
    let scaled = tokio::task::spawn_blocking(move || {
        check_cancelled(&worker_cancellation)?;
        let masked = apply_mask_to_rgba_with_feather(
            &frame.source,
            &frame.mask,
            options.mask_feather_percent,
        )
        .map_err(AssetProcessingError::Processing)?;
        scale(&masked, options, &worker_cancellation)
    })
    .await
    .map_err(|error| AssetProcessingError::Processing(error.to_string()))??;
    check_cancelled(&cancellation)?;
    encode_rgba_png(&scaled)
}

fn check_cancelled(token: &CancellationToken) -> Result<(), AssetProcessingError> {
    if token.is_cancelled() {
        Err(AssetProcessingError::Processing(
            "Downscaling was cancelled.".into(),
        ))
    } else {
        Ok(())
    }
}

fn scale(
    input: &RgbaImage,
    options: AssetProcessingOptions,
    token: &CancellationToken,
) -> Result<RgbaImage, AssetProcessingError> {
    if input.width() == 0 || input.height() == 0 {
        return Err(AssetProcessingError::Processing(
            "At least one source frame is required.".into(),
        ));
    }
    check_cancelled(token)?;
    let factor = (options.width as f64 / input.width() as f64)
        .min(options.height as f64 / input.height() as f64);
    let width = (input.width() as f64 * factor).round().max(1.0) as u32;
    let height = (input.height() as f64 * factor).round().max(1.0) as u32;
    let result = if (width, height) == input.dimensions() {
        input.clone()
    } else if factor >= 1.0 {
        enlarge(input, width, height)
    } else {
        asset_scaler::resize_with_memory_limit(
            input,
            width,
            height,
            GameAssetAa::new(options.aa_percent),
            &|| token.is_cancelled(),
            CORE_SCALER_MEMORY_LIMIT,
        )
        .map_err(|error| AssetProcessingError::Processing(error.to_string()))?
    };
    check_cancelled(token)?;
    let mut canvas = RgbaImage::new(options.width, options.height);
    imageops::replace(
        &mut canvas,
        &result,
        i64::from((options.width - width) / 2),
        i64::from((options.height - height) / 2),
    );
    Ok(canvas)
}

type FloatImage = ImageBuffer<Rgba<f32>, Vec<f32>>;
fn enlarge(input: &RgbaImage, width: u32, height: u32) -> RgbaImage {
    let source: FloatImage = ImageBuffer::from_fn(input.width(), input.height(), |x, y| {
        let pixel = input.get_pixel(x, y);
        let alpha = f32::from(pixel[3]) / 255.0;
        Rgba([
            linear(f32::from(pixel[0]) / 255.0) * alpha,
            linear(f32::from(pixel[1]) / 255.0) * alpha,
            linear(f32::from(pixel[2]) / 255.0) * alpha,
            alpha,
        ])
    });
    let enlarged = imageops::resize(&source, width, height, imageops::FilterType::CatmullRom);
    ImageBuffer::from_fn(width, height, |x, y| encode(enlarged.get_pixel(x, y).0))
}
fn linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}
fn srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}
fn encode(color: [f32; 4]) -> Rgba<u8> {
    let alpha = color[3].clamp(0.0, 1.0);
    let a = (alpha * 255.0).round() as u8;
    if a == 0 {
        return Rgba([0; 4]);
    }
    Rgba([
        (srgb((color[0] / alpha).clamp(0.0, 1.0)) * 255.0).round() as u8,
        (srgb((color[1] / alpha).clamp(0.0, 1.0)) * 255.0).round() as u8,
        (srgb((color[2] / alpha).clamp(0.0, 1.0)) * 255.0).round() as u8,
        a,
    ])
}

fn normalize_upload(upload: &[u8]) -> Result<RgbaImage, AssetProcessingError> {
    validate_upload(upload)?;
    let mut reader = ImageReader::new(Cursor::new(upload));
    reader.set_format(guess_supported_format(upload)?);
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_SOURCE_WIDTH);
    limits.max_image_height = Some(MAX_SOURCE_HEIGHT);
    limits.max_alloc = Some(MAX_DECODE_BYTES);
    reader.limits(limits);
    reader
        .decode()
        .map_err(|error| match error {
            image::ImageError::Limits(_) => AssetProcessingError::SourceTooLarge,
            error => AssetProcessingError::InvalidImage(error.to_string()),
        })
        .map(|image| image.into_rgba8())
}
fn validate_upload(upload: &[u8]) -> Result<(), AssetProcessingError> {
    if upload.is_empty() {
        return Err(AssetProcessingError::EmptyUpload);
    }
    if upload.len() > MAX_UPLOAD_BYTES {
        return Err(AssetProcessingError::UploadTooLarge);
    }
    Ok(())
}

fn model_memory_limit() -> u64 {
    std::env::var("ASSET_SCALER_PROCESSING_MEMORY_BUDGET_MIB")
        .or_else(|_| std::env::var("SPRITE_STUDIO_PROCESSING_MEMORY_BUDGET_MIB"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&mib| mib > 0)
        .map(|mib| mib.saturating_mul(1024 * 1024))
        .unwrap_or(DEFAULT_MODEL_MEMORY_LIMIT)
}

fn validate_model_memory(
    source: (u32, u32),
    options: AssetProcessingOptions,
) -> Result<(), AssetProcessingError> {
    let estimate = u64::from(source.0)
        .saturating_mul(u64::from(source.1))
        .saturating_mul(512)
        .saturating_add(
            u64::from(options.width)
                .saturating_mul(u64::from(options.height))
                .saturating_mul(160),
        )
        .max(1);
    let limit = model_memory_limit();
    if estimate > limit {
        return Err(AssetProcessingError::Processing(format!(
            "Game Asset scaling would exceed the {limit} byte memory limit."
        )));
    }
    Ok(())
}
fn guess_supported_format(upload: &[u8]) -> Result<ImageFormat, AssetProcessingError> {
    let format = image::guess_format(upload)
        .map_err(|error| AssetProcessingError::InvalidImage(error.to_string()))?;
    match format {
        ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP => Ok(format),
        _ => Err(AssetProcessingError::InvalidImage(
            "only PNG, JPEG, and WebP uploads are accepted".into(),
        )),
    }
}
fn write_rgba_png(path: &Path, image: &RgbaImage) -> Result<(), std::io::Error> {
    image
        .save_with_format(path, ImageFormat::Png)
        .map_err(std::io::Error::other)
}
fn encode_rgba_png(image: &RgbaImage) -> Result<Vec<u8>, AssetProcessingError> {
    let mut png = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
        .map_err(|error| AssetProcessingError::Processing(error.to_string()))?;
    Ok(png)
}

fn edge_connected_matte(mut image: RgbaImage) -> RgbaImage {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return image;
    }
    let matte = edge_median(&image);
    let mut seen = vec![false; width as usize * height as usize];
    let mut queue = VecDeque::new();
    for x in 0..width {
        queue.push_back((x, 0));
        if height > 1 {
            queue.push_back((x, height - 1));
        }
    }
    for y in 1..height.saturating_sub(1) {
        queue.push_back((0, y));
        if width > 1 {
            queue.push_back((width - 1, y));
        }
    }
    while let Some((x, y)) = queue.pop_front() {
        let index = y as usize * width as usize + x as usize;
        if seen[index] {
            continue;
        }
        seen[index] = true;
        if !is_edge_matte(image.get_pixel(x, y).0, matte) {
            continue;
        }
        image.put_pixel(x, y, Rgba([0; 4]));
        for (nx, ny) in [
            (x.checked_sub(1), Some(y)),
            (x.checked_add(1).filter(|&nx| nx < width), Some(y)),
            (Some(x), y.checked_sub(1)),
            (Some(x), y.checked_add(1).filter(|&ny| ny < height)),
        ] {
            if let (Some(nx), Some(ny)) = (nx, ny) {
                queue.push_back((nx, ny));
            }
        }
    }
    image
}
fn edge_median(image: &RgbaImage) -> [u8; 3] {
    let (width, height) = image.dimensions();
    let mut channels = [Vec::new(), Vec::new(), Vec::new()];
    for x in 0..width {
        for y in [0, height - 1] {
            for (values, value) in channels.iter_mut().zip(&image.get_pixel(x, y).0[..3]) {
                values.push(*value);
            }
        }
    }
    for y in 0..height {
        for x in [0, width - 1] {
            for (values, value) in channels.iter_mut().zip(&image.get_pixel(x, y).0[..3]) {
                values.push(*value);
            }
        }
    }
    channels.map(|mut values| {
        values.sort_unstable();
        let upper = values.len() / 2;
        ((u16::from(values[upper - 1]) + u16::from(values[upper])) / 2) as u8
    })
}
fn is_edge_matte(pixel: [u8; 4], matte: [u8; 3]) -> bool {
    pixel[3] == 0
        || pixel[..3]
            .iter()
            .zip(matte)
            .all(|(&channel, value)| channel == value)
}

#[cfg(all(test, unix))]
mod model_processing_tests {
    use super::*;
    use image::{GrayImage, Luma, Rgba};
    use std::os::unix::fs::PermissionsExt;

    use asset_background_removal::{
        InspyReNet, apply_mask_to_rgba_with_feather, test_service_lock,
    };

    fn fake_runtime(root: &Path) -> InspyReNet {
        let python = root.join("fake-inspyrenet.py");
        std::fs::write(
            &python,
            r#"#!/usr/bin/env python3
import json, os, struct, sys, zlib
if "--runtime-identity" in sys.argv:
    print('{"backend":"fake","torch":"fake","hip":null}', flush=True)
    raise SystemExit(0)
with open(sys.argv[0] + ".starts", "a", encoding="utf-8") as starts:
    starts.write("start\n")
print('{"event":"ready","runtime":{"backend":"fake"}}', flush=True)
def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
for line in sys.stdin:
    request = json.loads(line)
    for index, source in enumerate(request["frames"]):
        with open(source, "rb") as image:
            header = image.read(24)
        width, height = struct.unpack(">II", header[16:24])
        row = bytes((0, 128, 255)[x % 3] for x in range(width))
        raw = b"".join(b"\0" + row for _ in range(height))
        target_dir = os.path.join(request["output"], "masks")
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, "mask-%06d.png" % index)
        with open(target, "wb") as mask:
            mask.write(b"\x89PNG\r\n\x1a\n")
            mask.write(png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)))
            mask.write(png_chunk(b"IDAT", zlib.compress(raw)))
            mask.write(png_chunk(b"IEND", b""))
        print(json.dumps({"event":"mask","request":request["request"],"index":index}), flush=True)
        assert json.loads(sys.stdin.readline()) == {"event":"ack","request":request["request"],"index":index}
    print(json.dumps({"event":"complete","request":request["request"],"frame_count":len(request["frames"]),"width":width,"height":height}), flush=True)
"#,
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&python).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&python, permissions).unwrap();
        let checkpoint = root.join("checkpoint.pth");
        std::fs::write(&checkpoint, b"fake checkpoint").unwrap();
        InspyReNet {
            python,
            checkpoint,
            device: "cpu".into(),
        }
    }

    fn fault_runtime(root: &Path) -> InspyReNet {
        let python = root.join("fault-inspyrenet.py");
        std::fs::write(
            &python,
            r#"#!/usr/bin/env python3
import json, os, struct, sys, time, zlib
if "--runtime-identity" in sys.argv:
    print('{"backend":"fake","torch":"fake","hip":null}', flush=True)
    raise SystemExit(0)
checkpoint = sys.argv[sys.argv.index("--checkpoint") + 1]
mode_path = checkpoint + ".mode"
print('{"event":"ready","runtime":{"backend":"fake"}}', flush=True)
def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
for line in sys.stdin:
    request = json.loads(line)
    mode = open(mode_path, encoding="utf-8").read().strip()
    open(checkpoint + ".started", "w", encoding="utf-8").write(mode)
    if mode == "error":
        print(json.dumps({"event":"error","request":request["request"],"message":"forced failure"}), flush=True)
        continue
    if mode == "hang":
        time.sleep(30)
        continue
    source = request["frames"][0]
    with open(source, "rb") as image:
        header = image.read(24)
    width, height = struct.unpack(">II", header[16:24])
    target_dir = os.path.join(request["output"], "masks")
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, "mask-000000.png")
    raw = b"".join(b"\0" + bytes([255]) * width for _ in range(height))
    with open(target, "wb") as mask:
        mask.write(b"\x89PNG\r\n\x1a\n")
        mask.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)))
        mask.write(chunk(b"IDAT", zlib.compress(raw)))
        mask.write(chunk(b"IEND", b""))
    print(json.dumps({"event":"mask","request":request["request"],"index":0}), flush=True)
    assert json.loads(sys.stdin.readline()) == {"event":"ack","request":request["request"],"index":0}
    if mode == "mask-then-error":
        print(json.dumps({"event":"error","request":request["request"],"message":"forced after mask"}), flush=True)
    else:
        print(json.dumps({"event":"complete","request":request["request"],"frame_count":1,"width":width,"height":height}), flush=True)
"#,
        ).unwrap();
        let mut permissions = std::fs::metadata(&python).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&python, permissions).unwrap();
        let checkpoint = root.join("fault-checkpoint.pth");
        std::fs::write(&checkpoint, b"fake checkpoint").unwrap();
        std::fs::write(checkpoint.with_extension("pth.mode"), "normal").unwrap();
        InspyReNet {
            python,
            checkpoint,
            device: "cpu".into(),
        }
    }

    fn png_bytes(image: &RgbaImage) -> Vec<u8> {
        let mut bytes = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
            .unwrap();
        bytes
    }

    #[tokio::test]
    async fn edge_matte_mode_preserves_near_matte_top_stone_without_starting_the_model() {
        let root = tempfile::tempdir().unwrap();
        let processor = AssetProcessor::new(fake_runtime(root.path()));
        let matte = [112, 112, 108, 255];
        let mut source = RgbaImage::from_pixel(16, 16, Rgba(matte));
        for i in 3..13 {
            source.put_pixel(i, 3, Rgba([20, 18, 16, 255]));
            source.put_pixel(i, 12, Rgba([20, 18, 16, 255]));
            source.put_pixel(3, i, Rgba([20, 18, 16, 255]));
            source.put_pixel(12, i, Rgba([20, 18, 16, 255]));
        }
        // This is the light edge colour from the actual floor source. It is
        // reachable from the exterior and would be deleted by the former
        // ±26 matte tolerance, eating the tile's top rows after reduction.
        let top_stone = [121, 108, 89, 255];
        source.put_pixel(8, 2, Rgba(top_stone));
        // This is deliberately the same colour as the exterior matte, but it
        // is enclosed by opaque stone and must not become a transparency bite.
        source.put_pixel(8, 8, Rgba(matte));
        let output = processor
            .process_with_options(
                &png_bytes(&source),
                AssetProcessingOptions {
                    width: 16,
                    height: 16,
                    aa_percent: 0,
                    mask_feather_percent: 0,
                    mask_mode: AssetMaskMode::EdgeMatte,
                },
                &CancellationToken::new(),
            )
            .await
            .unwrap();
        let output = image::load_from_memory(&output).unwrap().into_rgba8();
        assert_eq!(output.get_pixel(0, 0).0, [0; 4]);
        assert_eq!(output.get_pixel(8, 2).0, top_stone);
        assert_eq!(output.get_pixel(3, 3).0, [20, 18, 16, 255]);
        assert_eq!(output.get_pixel(8, 8).0, matte);
        assert!(
            !root.path().join("fake-inspyrenet.py.starts").exists(),
            "edge-matte assets must not be sent to the semantic model"
        );
    }

    #[tokio::test]
    async fn model_protocol_errors_drain_before_a_later_request_restarts() {
        let _service = test_service_lock().await;
        let root = tempfile::tempdir().unwrap();
        let processor = AssetProcessor::new(fault_runtime(root.path()));
        let source = png_bytes(&RgbaImage::new(16, 16));
        let mode = root.path().join("fault-checkpoint.pth.mode");
        std::fs::write(&mode, "error").unwrap();
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                processor.process(&source, &CancellationToken::new())
            )
            .await
            .unwrap()
            .is_err()
        );
        std::fs::write(&mode, "mask-then-error").unwrap();
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                processor.process(&source, &CancellationToken::new())
            )
            .await
            .unwrap()
            .is_err()
        );
        std::fs::write(&mode, "normal").unwrap();
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                processor.process(&source, &CancellationToken::new())
            )
            .await
            .unwrap()
            .is_ok()
        );
    }

    #[tokio::test]
    async fn cancellation_while_waiting_for_a_model_mask_drains_and_restarts() {
        let _service = test_service_lock().await;
        let root = tempfile::tempdir().unwrap();
        let processor = AssetProcessor::new(fault_runtime(root.path()));
        let source = png_bytes(&RgbaImage::new(16, 16));
        let mode = root.path().join("fault-checkpoint.pth.mode");
        std::fs::write(&mode, "hang").unwrap();
        let cancellation = CancellationToken::new();
        let work = processor.process(&source, &cancellation);
        tokio::pin!(work);
        let started = root.path().join("fault-checkpoint.pth.started");
        tokio::select! {
            result = &mut work => panic!("hanging request ended before cancellation: {result:?}"),
            result = tokio::time::timeout(std::time::Duration::from_secs(2), async {
                while !started.is_file() {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
            }) => assert!(result.is_ok(), "model request did not reach the hanging runtime"),
        }
        cancellation.cancel();
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(2), &mut work)
                .await
                .unwrap()
                .is_err()
        );
        std::fs::write(&mode, "normal").unwrap();
        assert!(
            tokio::time::timeout(
                std::time::Duration::from_secs(2),
                processor.process(&source, &CancellationToken::new())
            )
            .await
            .unwrap()
            .is_ok()
        );
    }

    #[tokio::test]
    async fn processes_a_single_image_with_video_frame_alpha_and_scaling_semantics() {
        let _service = test_service_lock().await;
        let root = tempfile::tempdir().unwrap();
        let processor = AssetProcessor::new(fake_runtime(root.path()));
        let source = RgbaImage::from_fn(400, 250, |x, y| {
            Rgba([
                (x % 251) as u8,
                (y % 251) as u8,
                120,
                [64, 128, 255][((x + y) % 3) as usize],
            ])
        });
        let upload = png_bytes(&source);
        let token = CancellationToken::new();

        assert!(processor.process(b"not an image", &token).await.is_err());
        let first = processor.process(&upload, &token).await.unwrap();
        let second = processor.process(&upload, &token).await.unwrap();
        let custom = processor
            .process_with_options(
                &upload,
                AssetProcessingOptions {
                    width: 128,
                    height: 256,
                    ..AssetProcessingOptions::default()
                },
                &token,
            )
            .await
            .unwrap();
        let explicit_defaults = processor
            .process_with_options(&upload, AssetProcessingOptions::default(), &token)
            .await
            .unwrap();
        let stronger_aa = processor
            .process_with_options(
                &upload,
                AssetProcessingOptions {
                    aa_percent: 50,
                    ..AssetProcessingOptions::default()
                },
                &token,
            )
            .await
            .unwrap();

        let baseline = tempfile::tempdir().unwrap();
        let source_path = baseline.path().join("source.png");
        source.save(&source_path).unwrap();
        let mask_path = baseline.path().join("mask.png");
        GrayImage::from_fn(source.width(), source.height(), |x, _| {
            Luma([[0, 128, 255][(x % 3) as usize]])
        })
        .save(&mask_path)
        .unwrap();
        let masked = apply_mask_to_rgba_with_feather(&source_path, &mask_path, 0).unwrap();
        let expected = scale(&masked, AssetProcessingOptions::default(), &token).unwrap();
        for output in [&first, &second] {
            let output = image::load_from_memory(output).unwrap().into_rgba8();
            assert_eq!(output.dimensions(), (200, 200));
            assert_eq!(output, expected);
            assert!(output.pixels().any(|pixel| pixel[3] == 0));
            assert!(output.pixels().any(|pixel| pixel[3] > 0));
        }
        assert_eq!(first, explicit_defaults);
        let stronger_aa_expected = scale(
            &masked,
            AssetProcessingOptions {
                aa_percent: 50,
                ..AssetProcessingOptions::default()
            },
            &token,
        )
        .unwrap();
        assert_eq!(
            image::load_from_memory(&stronger_aa).unwrap().into_rgba8(),
            stronger_aa_expected
        );
        assert_ne!(
            first, stronger_aa,
            "the fixture distinguishes AA 20 from 50"
        );
        let custom_expected = scale(
            &masked,
            AssetProcessingOptions {
                width: 128,
                height: 256,
                ..AssetProcessingOptions::default()
            },
            &token,
        )
        .unwrap();
        let custom = image::load_from_memory(&custom).unwrap().into_rgba8();
        assert_eq!(custom.dimensions(), (128, 256));
        assert_eq!(custom, custom_expected);
        let starts =
            std::fs::read_to_string(root.path().join("fake-inspyrenet.py.starts")).unwrap();
        assert_eq!(starts.lines().count(), 1, "model process should stay warm");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fit_and_premultiplied_enlargement_stay_exact() {
        let mut source = RgbaImage::new(2, 1);
        source.put_pixel(0, 0, Rgba([255, 0, 0, 0]));
        source.put_pixel(1, 0, Rgba([0, 0, 255, 255]));
        let image = scale(
            &source,
            AssetProcessingOptions {
                width: 8,
                height: 8,
                ..Default::default()
            },
            &CancellationToken::new(),
        )
        .unwrap();
        assert_eq!(image.dimensions(), (8, 8));
        assert!(image.rows().take(2).flatten().all(|pixel| pixel[3] == 0));
        assert!(image.rows().skip(6).flatten().all(|pixel| pixel[3] == 0));
    }
    #[test]
    fn edge_matte_keeps_enclosed_matte() {
        let matte = Rgba([112, 112, 108, 255]);
        let mut image = RgbaImage::from_pixel(8, 8, matte);
        for i in 2..6 {
            image.put_pixel(i, 2, Rgba([20, 18, 16, 255]));
            image.put_pixel(i, 5, Rgba([20, 18, 16, 255]));
            image.put_pixel(2, i, Rgba([20, 18, 16, 255]));
            image.put_pixel(5, i, Rgba([20, 18, 16, 255]));
        }
        assert_eq!(edge_connected_matte(image).get_pixel(3, 3), &matte);
    }
}
