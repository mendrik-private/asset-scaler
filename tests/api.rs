use asset_scaler::{
    CancellationToken, Error, GameAssetAa, ResizeOptions, Session, resize,
    resize_with_memory_limit, resize_with_options,
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

#[test]
fn opaque_background_removal_can_be_disabled_for_cached_and_single_shot_resize() {
    let source = Arc::new(RgbaImage::from_fn(64, 64, |x, y| {
        if (20..44).contains(&x) && (20..44).contains(&y) {
            Rgba([120, 180, 90, 255])
        } else {
            Rgba([240, 230, 220, 255])
        }
    }));
    let transparent = Arc::new(RgbaImage::from_fn(64, 64, |x, y| {
        if (20..44).contains(&x) && (20..44).contains(&y) {
            Rgba([120, 180, 90, 255])
        } else {
            Rgba([240, 0, 220, 0])
        }
    }));
    let cancel = CancellationToken::default();
    let aa = GameAssetAa::new(20);
    let preserve = ResizeOptions::preserve_opaque_background();

    let one_shot = resize_with_options(&source, 16, 16, aa, &cancel, preserve).unwrap();
    let session = Session::with_options(source.clone(), preserve);
    let cached = session.resize(16, 16, aa, &cancel).unwrap();
    assert_eq!(cached, one_shot);
    assert_eq!(session.resize(16, 16, aa, &cancel).unwrap(), cached);
    assert_eq!(one_shot.get_pixel(0, 0), &Rgba([240, 230, 220, 255]));
    assert_eq!(
        one_shot.get_pixel(8, 8)[3],
        255,
        "opaque foreground survived"
    );

    assert_eq!(
        resize(&source, 16, 16, aa, &cancel)
            .unwrap()
            .get_pixel(0, 0),
        &Rgba([0; 4]),
        "the default retains automatic opaque-background removal"
    );
    assert_eq!(
        Session::new(source.clone())
            .resize(16, 16, aa, &cancel)
            .unwrap()
            .get_pixel(0, 0),
        &Rgba([0; 4]),
    );

    let transparent_one_shot =
        resize_with_options(&transparent, 16, 16, aa, &cancel, preserve).unwrap();
    let transparent_cached = Session::with_options(transparent.clone(), preserve)
        .resize(16, 16, aa, &cancel)
        .unwrap();
    assert_eq!(transparent_cached, transparent_one_shot);
    assert_eq!(transparent_one_shot.get_pixel(0, 0)[3], 0);
    assert_eq!(transparent_one_shot.get_pixel(8, 8)[3], 255);

    assert_eq!(
        resize_with_options(&source, 64, 64, aa, &cancel, preserve).unwrap(),
        *source
    );
    assert_eq!(
        session.resize(64, 64, aa, &cancel).unwrap(),
        *source,
        "identity requests bypass both analysis and cached target state"
    );
}

#[test]
fn prepared_original_contours_paint_each_fresh_removed_foreground() {
    let original = Arc::new(RgbaImage::from_fn(64, 64, |x, y| {
        if (16..48).contains(&x) && (16..48).contains(&y) {
            if (30..34).contains(&y) {
                Rgba([8, 12, 16, 255])
            } else {
                Rgba([120, 90, 70, 255])
            }
        } else {
            Rgba([240, 240, 240, 255])
        }
    }));
    let foreground = |colour: [u8; 3]| {
        RgbaImage::from_fn(64, 64, |x, y| {
            if (16..48).contains(&x) && (16..48).contains(&y) {
                Rgba([colour[0], colour[1], colour[2], 255])
            } else {
                Rgba([0; 4])
            }
        })
    };
    let session = Session::new(original);
    let cancel = CancellationToken::default();
    session.prepare(&cancel).unwrap();
    let red = foreground([220, 60, 30]);
    let green = foreground([40, 180, 80]);
    for aa in [0, 20, 50, 100] {
        let normal = session
            .resize(32, 32, GameAssetAa::new(aa), &cancel)
            .unwrap();
        let first = session
            .resize_with_foreground(&red, 32, 32, GameAssetAa::new(aa), &cancel)
            .unwrap();
        let second = session
            .resize_with_foreground(&green, 32, 32, GameAssetAa::new(aa), &cancel)
            .unwrap();
        assert!(
            first
                .pixels()
                .zip(second.pixels())
                .any(|(a, b)| a[3] > 0 && b[3] > 0 && a.0 != b.0),
            "AA {aa}: stale foreground result was reused"
        );
        assert_ne!(
            first, normal,
            "AA {aa}: normal cached result was reused for a foreground"
        );
        let foreground_only = resize(&red, 32, 32, GameAssetAa::new(aa), &cancel).unwrap();
        assert_ne!(
            first, foreground_only,
            "AA {aa}: original contour analysis was not used"
        );
        assert_eq!(*first.get_pixel(0, 0), Rgba([0; 4]));
        assert_eq!(*second.get_pixel(31, 31), Rgba([0; 4]));
    }
    assert!(matches!(
        session.resize_with_foreground(
            &RgbaImage::new(1, 1),
            32,
            32,
            GameAssetAa::new(20),
            &cancel,
        ),
        Err(Error::InvalidDimensions)
    ));
}

#[test]
fn foreground_prepare_checks_cancellation_and_its_extra_memory() {
    let source = Arc::new(RgbaImage::from_pixel(32, 32, Rgba([90, 70, 50, 255])));
    let foreground = RgbaImage::from_pixel(32, 32, Rgba([90, 70, 50, 255]));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        Session::new(source.clone()).prepare(&cancelled),
        Err(Error::Cancelled)
    ));

    let session = Session::with_memory_limit(source, 600_000);
    let cancel = CancellationToken::default();
    session.prepare(&cancel).unwrap();
    cancel.cancel();
    assert!(matches!(
        session.resize_with_foreground(&foreground, 16, 16, GameAssetAa::new(20), &cancel),
        Err(Error::Cancelled)
    ));
    let cancel = CancellationToken::default();
    assert!(matches!(
        session.resize_with_foreground(&foreground, 16, 16, GameAssetAa::new(20), &cancel),
        Err(Error::GameAssetMemoryLimit {
            limit_bytes: 600_000
        })
    ));
    assert!(matches!(
        Session::new(Arc::new(RgbaImage::new(0, 0))).prepare(&CancellationToken::default()),
        Err(Error::InvalidDimensions)
    ));
}
