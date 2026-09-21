//! Loopback transport and single-image processing for asset scaling.

mod processing;
mod router;

pub use processing::{
    AssetMaskMode, AssetProcessingError, AssetProcessingOptions, AssetProcessor, MAX_UPLOAD_BYTES,
};
pub use router::router;
