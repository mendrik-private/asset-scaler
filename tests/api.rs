use asset_scaler::{
    CancellationToken, Error, GameAssetAa, Session, resize, resize_with_memory_limit,
};
use image::{Rgba, RgbaImage};
use std::sync::Arc;

#[test]
fn cached_and_single_shot_callers_share_pixels_and_limits() {
    let image = Arc::new(RgbaImage::from_fn(32, 24, |x, y| {
        Rgba([
            (x * 7) as u8,
            (y * 9) as u8,
            90,
            if x < 8 { 0 } else { 255 },
        ])
    }));
    let token = CancellationToken::default();
    let session = Session::new(image.clone());
    for aa in [0, 50, 100, 0] {
        let aa = GameAssetAa::new(aa);
        assert_eq!(
            resize(&image, 12, 9, aa, &token).unwrap(),
            session.resize(12, 9, aa, &token).unwrap()
        );
    }
    assert!(matches!(
        resize_with_memory_limit(&image, 12, 9, GameAssetAa::default(), &token, 1),
        Err(Error::GameAssetMemoryLimit { limit_bytes: 1 })
    ));
    assert!(matches!(
        Session::with_memory_limit(image.clone(), 1).resize(12, 9, GameAssetAa::default(), &token),
        Err(Error::GameAssetMemoryLimit { limit_bytes: 1 })
    ));
    token.cancel();
    assert!(matches!(
        session.resize(12, 9, GameAssetAa::default(), &token),
        Err(Error::Cancelled)
    ));
    assert!(matches!(
        resize(&image, 32, 24, GameAssetAa::default(), &|| true),
        Err(Error::Cancelled)
    ));
}

#[test]
fn cancellation_during_analysis_never_publishes_a_partial_output() {
    let image = RgbaImage::from_pixel(32, 32, Rgba([0, 0, 0, 255]));
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let cancelled = || calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed) >= 3;
    assert!(matches!(
        resize(&image, 16, 16, GameAssetAa::default(), &cancelled),
        Err(Error::Cancelled)
    ));
}
