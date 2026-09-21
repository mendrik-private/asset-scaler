//! Standalone loopback server binary.

use std::net::SocketAddr;

use asset_scaler_server::AssetProcessor;
use tokio_util::sync::CancellationToken;

const DEFAULT_ADDRESS: &str = "127.0.0.1:47831";

#[tokio::main]
async fn main() {
    let address = std::env::var("ASSET_SCALER_SERVER_ADDRESS")
        .or_else(|_| std::env::var("SPRITE_STUDIO_ASSET_SERVER_ADDRESS"))
        .map_or_else(
            |_| DEFAULT_ADDRESS.parse().expect("default address is valid"),
            |value| match value.parse::<SocketAddr>() {
                Ok(address) if address.ip().is_loopback() => address,
                Ok(_) => exit("ASSET_SCALER_SERVER_ADDRESS must be a loopback address."),
                Err(error) => exit(&format!("Invalid ASSET_SCALER_SERVER_ADDRESS: {error}")),
            },
        );
    // Bind before touching the model so invalid or occupied endpoints never
    // start the expensive Python service.
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .unwrap_or_else(|error| exit(&format!("Could not bind {address}: {error}")));
    let processor =
        AssetProcessor::from_environment().unwrap_or_else(|error| exit(&error.to_string()));
    let cancellation = CancellationToken::new();
    processor
        .warm(&cancellation)
        .await
        .unwrap_or_else(|error| exit(&format!("Could not start InSPyReNet: {error}")));
    println!("Asset Scaler server listening on http://{address}");
    axum::serve(listener, asset_scaler_server::router(processor))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap_or_else(|error| exit(&format!("Asset server stopped unexpectedly: {error}")));
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

fn exit(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}
