use super::*;

fn outlined_fixture(transparent: bool) -> RgbaImage {
    RgbaImage::from_fn(64, 60, |x, y| {
        let inside = (17..47).contains(&x) && (14..48).contains(&y);
        let edge = inside && (x == 17 || x == 46 || y == 14 || y == 47);
        if edge {
            image::Rgba([8, 12, 20, 255])
        } else if inside {
            image::Rgba([
                100 + ((x - 18) * 3) as u8,
                50 + ((y - 15) * 4) as u8,
                40 + ((x + y) % 35) as u8,
                255,
            ])
        } else if transparent {
            image::Rgba([235, 1, 99, 0])
        } else {
            image::Rgba([37, 83, 149, 255])
        }
    })
}

fn expected_foreground_contour_mask(
    prepared: &Prepared,
    original: &RgbaImage,
    foreground: &RgbaImage,
    w: u32,
    h: u32,
    aa: GameAssetAa,
    cancel: &dyn Cancellation,
) -> GrayImage {
    let fill_linear = color::LinearImage::from_rgba(foreground);
    let fill_silhouette = silhouette::Silhouette::detect(foreground, cancel).unwrap();
    let contours = prepared
        .target_contours_with_ink_source(
            original,
            w,
            h,
            aa,
            InkSource {
                linear: &fill_linear,
                suppress_unsupported: true,
                brightness: None,
            },
            cancel,
        )
        .unwrap();
    let fill = prepared
        .fill_for_target(
            &fill_linear,
            fill_silhouette.as_ref(),
            &contours.strokes,
            FillTarget {
                target: (w, h),
                aa,
                foreground_support: true,
            },
            cancel,
        )
        .unwrap();
    let raw = raster::Mask {
        w: w as usize,
        h: h as usize,
        data: contours
            .strokes
            .core
            .as_raw()
            .iter()
            .map(|&value| value != 0)
            .collect(),
    };
    let canonical =
        target_cleanup::thin_outer(&raw, fill.foreground_support.as_deref().unwrap(), cancel)
            .unwrap();
    binary_contour_image(&canonical).unwrap()
}

#[test]
fn session_exposes_binary_source_and_baseline_foreground_contour_masks() {
    let original = Arc::new(outlined_fixture(true));
    let foreground = original.as_ref().clone();
    let cancel = CancellationToken::default();
    let session = Session::new(original.clone());
    let prepared = Prepared::new(&original, &cancel).unwrap();

    let source = session.source_contour_mask(&cancel).unwrap();
    assert_eq!(source.dimensions(), original.dimensions());
    let expected_source = binary_contour_image(&prepared.mask).unwrap();
    assert_eq!(source, expected_source);
    assert!(
        source
            .pixels()
            .all(|pixel| pixel[0] == 0 || pixel[0] == 255)
    );

    let mut across_aa = None;
    for aa in [
        GameAssetAa::new(0),
        GameAssetAa::new(20),
        GameAssetAa::new(100),
    ] {
        let target = session
            .foreground_contour_mask(&foreground, 31, 29, aa, &cancel)
            .unwrap();
        assert_eq!(target.dimensions(), (31, 29));
        assert_eq!(
            target,
            expected_foreground_contour_mask(
                &prepared,
                &original,
                &foreground,
                31,
                29,
                aa,
                &cancel,
            )
        );
        assert!(
            target
                .pixels()
                .all(|pixel| pixel[0] == 0 || pixel[0] == 255)
        );
        if let Some(previous) = &across_aa {
            assert_eq!(&target, previous, "binary core must not encode AA coverage");
        }
        across_aa = Some(target);
    }
}

#[test]
fn contour_mask_api_validates_dimensions_and_cancellation() {
    let source = Arc::new(outlined_fixture(true));
    let session = Session::new(source.clone());
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        session.source_contour_mask(&cancelled),
        Err(Error::Cancelled)
    ));
    assert!(matches!(
        session.foreground_contour_mask(
            &source,
            0,
            10,
            GameAssetAa::new(0),
            &CancellationToken::default(),
        ),
        Err(Error::InvalidDimensions)
    ));
    assert!(matches!(
        session.foreground_contour_mask(
            &RgbaImage::new(1, 1),
            10,
            10,
            GameAssetAa::new(0),
            &CancellationToken::default(),
        ),
        Err(Error::InvalidDimensions)
    ));
    assert!(matches!(
        Session::new(Arc::new(RgbaImage::new(0, 0)))
            .source_contour_mask(&CancellationToken::default()),
        Err(Error::InvalidDimensions)
    ));
}

fn source_alpha_footprint(source: &RgbaImage, x: u32, y: u32, w: u32, h: u32) -> (bool, f64, f64) {
    let sx = source.width() as f64 / w as f64;
    let sy = source.height() as f64 / h as f64;
    let (left, right) = (x as f64 * sx, (x + 1) as f64 * sx);
    let (top, bottom) = (y as f64 * sy, (y + 1) as f64 * sy);
    let mut all_zero = true;
    let mut alpha = 0.;
    let mut support = 0.;
    for yy in top.floor() as u32..(bottom.ceil() as u32).min(source.height()) {
        let yw = (bottom.min((yy + 1) as f64) - top.max(yy as f64)) / sy;
        for xx in left.floor() as u32..(right.ceil() as u32).min(source.width()) {
            let xw = (right.min((xx + 1) as f64) - left.max(xx as f64)) / sx;
            let sample = source.get_pixel(xx, yy)[3];
            all_zero &= sample == 0;
            alpha += f64::from(sample) / 255. * xw * yw;
            support += f64::from(sample != 0) * xw * yw;
        }
    }
    (all_zero, alpha, support)
}

/// Independent test oracle for direct source sampling. This intentionally
/// performs its own f32 premultiplied conversion and `image` Lanczos call,
/// rather than invoking the production Game Asset helper.
fn direct_source_lanczos_oracle(source: &color::LinearImage, w: u32, h: u32) -> RgbaImage {
    let samples: Vec<f32> = source
        .pixels
        .iter()
        .flat_map(|sample| {
            let alpha = sample[3] as f32;
            [
                sample[0] as f32 * alpha,
                sample[1] as f32 * alpha,
                sample[2] as f32 * alpha,
                alpha,
            ]
        })
        .collect();
    let premultiplied = image::ImageBuffer::<image::Rgba<f32>, Vec<f32>>::from_vec(
        source.w as u32,
        source.h as u32,
        samples,
    )
    .expect("linear source dimensions match its samples");
    let resized =
        image::imageops::resize(&premultiplied, w, h, image::imageops::FilterType::Lanczos3);
    RgbaImage::from_fn(w, h, |x, y| {
        let sample = resized.get_pixel(x, y);
        let alpha = f64::from(sample[3]).clamp(0., 1.);
        color::rgba(if alpha <= 1e-8 {
            [0.; 4]
        } else {
            [
                (f64::from(sample[0]) / alpha).clamp(0., 1.),
                (f64::from(sample[1]) / alpha).clamp(0., 1.),
                (f64::from(sample[2]) / alpha).clamp(0., 1.),
                alpha,
            ]
        })
    })
}

#[test]
fn outline_free_fill_uses_the_clean_fill_and_keeps_colour_modes_separate() {
    let original = outlined_fixture(true);
    let fill = RgbaImage::from_fn(original.width(), original.height(), |x, y| {
        let alpha = original.get_pixel(x, y)[3];
        image::Rgba([230, 170, 90, alpha])
    });
    let cancel = CancellationToken::default();
    let prepared = Prepared::new(&original, &cancel).unwrap();
    let target = prepared
        .target_contours(&original, 31, 29, GameAssetAa::new(30), &cancel)
        .unwrap();
    let expected = direct_source_lanczos_oracle(&color::LinearImage::from_rgba(&fill), 31, 29);
    let original_ink = resize_with_outline_free_fill(
        &original,
        &fill,
        31,
        29,
        GameAssetAa::new(30),
        OutlineColor::OriginalInk,
        &cancel,
    )
    .unwrap();
    let darkened_fill = resize_with_outline_free_fill(
        &original,
        &fill,
        31,
        29,
        GameAssetAa::new(30),
        OutlineColor::DarkenedFill { luminance: 0.25 },
        &cancel,
    )
    .unwrap();
    let clean_pixel = target
        .strokes
        .coverage
        .as_raw()
        .iter()
        .enumerate()
        .find_map(|(i, &coverage)| {
            (coverage == 0 && expected.as_raw()[i * 4 + 3] == 255).then_some(i)
        })
        .expect("fixture must contain an opaque unpainted fill pixel");
    assert_eq!(
        &original_ink.as_raw()[clean_pixel * 4..clean_pixel * 4 + 4],
        &expected.as_raw()[clean_pixel * 4..clean_pixel * 4 + 4],
        "clean regions must come solely from the outline-free fill"
    );
    let core_pixel = target
        .strokes
        .core
        .as_raw()
        .iter()
        .position(|&value| value == 255)
        .expect("fixture must retain a contour core");
    assert!(
        darkened_fill.as_raw()[core_pixel * 4] > original_ink.as_raw()[core_pixel * 4],
        "the fill-derived contour must not accidentally reuse original dark ink"
    );
    assert!(matches!(
        resize_with_outline_free_fill(
            &original,
            &RgbaImage::new(1, 1),
            31,
            29,
            GameAssetAa::new(30),
            OutlineColor::OriginalInk,
            &cancel,
        ),
        Err(Error::InvalidDimensions)
    ));
}

#[test]
fn silhouette_hides_flat_canvas_and_transparent_hidden_rgb() {
    let source = Arc::new(outlined_fixture(false));
    let cancel = CancellationToken::default();
    let session = Session::new(source.clone());
    let current = session
        .resize(31, 29, GameAssetAa::new(0), &cancel)
        .unwrap();
    let prepared = session.cache.lock().unwrap().prepared.clone().unwrap();
    let silhouette = prepared.silhouette.as_ref().unwrap();
    let coverage = silhouette.coverage(31, 29, &cancel).unwrap();
    let target = prepared
        .target_contours(&source, 31, 29, GameAssetAa::new(0), &cancel)
        .unwrap();
    for (i, &support) in coverage.iter().enumerate() {
        if support < 0.5 && target.strokes.coverage.as_raw()[i] == 0 {
            assert_eq!(&current.as_raw()[i * 4..i * 4 + 4], &[0; 4], "pixel {i}");
        }
    }
    let transparent = Arc::new(outlined_fixture(true));
    let transparent_session = Session::new(transparent);
    let transparent_result = transparent_session
        .resize(31, 29, GameAssetAa::new(0), &cancel)
        .unwrap();
    let transparent_prepared = transparent_session
        .cache
        .lock()
        .unwrap()
        .prepared
        .clone()
        .unwrap();
    let transparent_silhouette = transparent_prepared.silhouette.as_ref().unwrap();
    let transparent_coverage = transparent_silhouette.coverage(31, 29, &cancel).unwrap();
    let transparent_target = transparent_prepared
        .target_contours(
            &outlined_fixture(true),
            31,
            29,
            GameAssetAa::new(0),
            &cancel,
        )
        .unwrap();
    for (i, &support) in transparent_coverage.iter().enumerate() {
        if support < 0.5 && transparent_target.strokes.coverage.as_raw()[i] == 0 {
            assert_eq!(transparent_result.as_raw()[i * 4 + 3], 0, "pixel {i}");
            assert_eq!(&transparent_result.as_raw()[i * 4..i * 4 + 4], &[0; 4]);
        }
    }
}

#[test]
fn silhouette_support_handles_every_small_downscale_shape_and_aa() {
    let source = Arc::new(RgbaImage::from_fn(11, 9, |x, y| {
        let inside = (2..9).contains(&x) && (2..7).contains(&y);
        image::Rgba(if inside {
            [80 + x as u8 * 9, 30 + y as u8 * 11, 50, 255]
        } else {
            [37, 83, 149, 255]
        })
    }));
    let session = Session::new(source.clone());
    let cancel = CancellationToken::default();
    for aa in [
        GameAssetAa::new(0),
        GameAssetAa::new(50),
        GameAssetAa::new(100),
    ] {
        for w in 1..=source.width() {
            for h in 1..=source.height() {
                let output = session.resize(w, h, aa, &cancel).unwrap();
                assert_eq!(output.dimensions(), (w, h));
                let prepared = session.cache.lock().unwrap().prepared.clone().unwrap();
                let silhouette = prepared.silhouette.as_ref().unwrap();
                let coverage = silhouette
                    .coverage(w as usize, h as usize, &cancel)
                    .unwrap();
                let contours = prepared
                    .target_contours(&source, w, h, aa, &cancel)
                    .unwrap();
                for (i, &support) in coverage.iter().enumerate() {
                    if support == 0.
                        && contours.strokes.coverage.as_raw()[i] == 0
                        && (w, h) != source.dimensions()
                    {
                        assert_eq!(
                            output.get_pixel((i % w as usize) as u32, (i / w as usize) as u32),
                            &image::Rgba([0; 4])
                        );
                    }
                }
            }
        }
    }
    assert_eq!(
        session
            .resize(
                source.width(),
                source.height(),
                GameAssetAa::new(0),
                &cancel
            )
            .unwrap(),
        *source
    );
}

#[test]
fn transparent_intrinsic_alpha_survives_hard_silhouette_coverage() {
    let source = Arc::new(RgbaImage::from_fn(16, 16, |x, y| {
        if (4..12).contains(&x) && (4..12).contains(&y) {
            image::Rgba([120, 40, 200, 128])
        } else {
            image::Rgba([255, 2, 90, 0])
        }
    }));
    let output = Session::new(source)
        .resize(8, 8, GameAssetAa::new(0), &CancellationToken::default())
        .unwrap();
    assert_eq!(*output.get_pixel(3, 3), image::Rgba([120, 40, 200, 128]));
    assert_eq!(*output.get_pixel(0, 0), image::Rgba([0; 4]));
}

#[test]
fn elf_bow_connection_survives_silhouette_clipping_at_all_aa_levels() {
    let source = Arc::new(
        image::load_from_memory(include_bytes!("fixtures/elf.png"))
            .unwrap()
            .into_rgba8(),
    );
    let session = Session::new(source);
    let cancel = CancellationToken::default();
    for aa in [
        GameAssetAa::new(0),
        GameAssetAa::new(50),
        GameAssetAa::new(100),
    ] {
        let output = session.resize(128, 128, aa, &cancel).unwrap();
        for (x, y) in [(80, 66), (81, 66), (80, 67)] {
            assert!(
                output.get_pixel(x, y)[3] >= 240,
                "AA{} erased bow at ({x},{y}): {:?}",
                aa.percent(),
                output.get_pixel(x, y)
            );
        }
    }
}

#[test]
fn direct_lanczos_source_oracle_and_silhouette_support_hold_at_all_sizes() {
    let source = Arc::new(
        image::load_from_memory(include_bytes!("fixtures/elf.png"))
            .unwrap()
            .into_rgba8(),
    );
    let session = Session::new(source.clone());
    for size in [128, 160, 200] {
        let cancel = CancellationToken::default();
        let result = session
            .resize(size, size, GameAssetAa::default(), &cancel)
            .unwrap();
        let prepared = session.cache.lock().unwrap().prepared.clone().unwrap();
        let fill_source = prepared
            .silhouette
            .as_ref()
            .map(|silhouette| silhouette.isolated(&prepared.linear, &cancel).unwrap())
            .unwrap_or_else(|| prepared.linear.clone());
        let base = lanczos::resize(&fill_source, size as usize, size as usize, &cancel).unwrap();
        let actual = RgbaImage::from_fn(size, size, |x, y| {
            color::rgba(base.pixels[y as usize * base.w + x as usize])
        });
        assert_eq!(
            actual,
            direct_source_lanczos_oracle(&fill_source, size, size),
            "independent direct source Lanczos oracle at {size}px"
        );

        let target = prepared
            .target_contours(&source, size, size, GameAssetAa::default(), &cancel)
            .unwrap();
        let mut unpainted = 0;
        for y in 0..size {
            for x in 0..size {
                let i = (y * size + x) as usize;
                if target.strokes.coverage.as_raw()[i] == 0 {
                    let (all_zero, alpha, support) =
                        source_alpha_footprint(&source, x, y, size, size);
                    if all_zero {
                        assert_eq!(result.get_pixel(x, y), &image::Rgba([0; 4]));
                    } else {
                        assert!(
                            f64::from(result.get_pixel(x, y)[3]) / 255.
                                <= alpha / support + 1. / 255.,
                            "partial-support alpha grew at {size}px ({x},{y})"
                        );
                    }
                    unpainted += 1;
                }
                if target.strokes.core.get_pixel(x, y)[0] != 0 {
                    assert!(target.strokes.coverage.get_pixel(x, y)[0] >= 243);
                }
            }
        }
        assert!(unpainted > size * size / 2);
    }
}
#[test]
fn rectangle_previews_reuse_source_analysis() {
    let image = Arc::new(RgbaImage::from_fn(41, 25, |x, y| {
        image::Rgba(if x == 20 {
            [0, 0, 0, 255]
        } else {
            [(x * 6) as u8, (y * 9) as u8, 120, 255]
        })
    }));
    let session = Session::new(image.clone());
    let cancel = CancellationToken::default();
    let preview = session
        .resize(20, 12, GameAssetAa::default(), &cancel)
        .unwrap();
    let prepared = session.cache.lock().unwrap().prepared.clone().unwrap();
    assert_eq!(
        session
            .resize(20, 12, GameAssetAa::default(), &cancel)
            .unwrap(),
        preview
    );
    assert_eq!(
        session
            .resize(1, 10, GameAssetAa::default(), &cancel)
            .unwrap()
            .dimensions(),
        (1, 10)
    );
    assert!(Arc::ptr_eq(
        &prepared,
        session.cache.lock().unwrap().prepared.as_ref().unwrap()
    ));
}
#[test]
fn aa_changes_cache_identity_without_reanalyzing_source() {
    let source = Arc::new(RgbaImage::from_fn(96, 80, |x, y| {
        let diagonal = (y as f64 - (0.57 * x as f64 + 10.)).abs() < 3.;
        image::Rgba(if diagonal {
            [8, 10, 4, 255]
        } else {
            [130, 170, 90, 255]
        })
    }));
    let session = Session::new(source.clone());
    let cancel = CancellationToken::default();
    let mut outputs = Vec::new();
    let mut prepared = None;
    for percent in [0, 100, 50, 0] {
        let aa = GameAssetAa::new(percent);
        let preview = session.resize(32, 27, aa, &cancel).unwrap();
        let cache = session.cache.lock().unwrap();
        assert_eq!(cache.target.as_ref().unwrap().0, (32, 27, aa));
        if let Some(old) = &prepared {
            assert!(Arc::ptr_eq(old, cache.prepared.as_ref().unwrap()));
        } else {
            prepared = cache.prepared.clone();
        }
        let cached_result = cache.target.as_ref().unwrap().1.clone();
        drop(cache);
        assert_eq!(session.resize(32, 27, aa, &cancel).unwrap(), preview);
        assert!(Arc::ptr_eq(
            &cached_result,
            &session.cache.lock().unwrap().target.as_ref().unwrap().1
        ));
        outputs.push(preview);
    }
    assert_ne!(outputs[0], outputs[1], "fixture must exercise AA");
    assert_ne!(outputs[1], outputs[2]);
    assert_eq!(
        outputs[0], outputs[3],
        "switching back must not return stale AA"
    );
}

#[test]
fn validates_dimensions_and_cancellation_without_populating_cache() {
    let session = Session::new(Arc::new(RgbaImage::new(8, 6)));
    for (w, h) in [(0, 3), (4, 0), (9, 3), (4, 7)] {
        assert!(matches!(
            session.resize(w, h, GameAssetAa::default(), &CancellationToken::default()),
            Err(Error::InvalidDimensions)
        ));
    }
    let cancel = CancellationToken::default();
    cancel.cancel();
    for (w, h) in [(8, 6), (4, 3)] {
        assert!(matches!(
            session.resize(w, h, GameAssetAa::default(), &cancel),
            Err(Error::Cancelled)
        ));
    }
    assert!(session.cache.lock().unwrap().prepared.is_none());
}
#[test]
fn transparent_rgb_does_not_bleed_and_empty_sources_remain_empty() {
    let a = RgbaImage::from_fn(12, 8, |x, _| {
        image::Rgba(if x < 6 {
            [10, 80, 190, 255]
        } else {
            [255, 0, 70, 0]
        })
    });
    let mut b = a.clone();
    for p in b.pixels_mut() {
        if p[3] == 0 {
            *p = image::Rgba([0; 4]);
        }
    }
    let cancel = CancellationToken::default();
    assert_eq!(
        Session::new(Arc::new(a))
            .resize(4, 3, GameAssetAa::default(), &cancel)
            .unwrap(),
        Session::new(Arc::new(b))
            .resize(4, 3, GameAssetAa::default(), &cancel)
            .unwrap()
    );
    let empty = Session::new(Arc::new(RgbaImage::from_pixel(
        9,
        7,
        image::Rgba([255, 0, 70, 0]),
    )))
    .resize(3, 2, GameAssetAa::default(), &cancel)
    .unwrap();
    assert!(empty.pixels().all(|p| p.0 == [0; 4]));
}

#[test]
fn retained_ink_excludes_short_neighbors() {
    let mut trace = raster::Mask::new(24, 12);
    for x in 2..20 {
        trace.data[5 * 24 + x] = true;
    }
    for x in 8..10 {
        trace.data[7 * 24 + x] = true;
    }
    let cancel = CancellationToken::default();
    let contours = contours::Contours::new(&trace, &[], &cancel).unwrap();
    let retained = contours
        .retained_ink_mask(&trace, [1., 1.], &cancel)
        .unwrap();
    assert!(retained.data[5 * 24 + 8]);
    assert!(!retained.data[7 * 24 + 8]);
}

#[test]
fn aa_zero_foreground_ink_keeps_owned_core_black_and_uses_exact_darkened_donor_colours() {
    let original = RgbaImage::from_fn(64, 64, |x, y| {
        if (12..52).contains(&x) && (12..52).contains(&y) {
            // Long enough to stay drawn as an interior line at tiny targets.
            if y == 31 && (15..49).contains(&x) {
                image::Rgba([0, 0, 0, 255])
            } else {
                image::Rgba([150, 110, 80, 255])
            }
        } else {
            image::Rgba([245, 240, 230, 255])
        }
    });
    let foreground = RgbaImage::from_fn(64, 64, |x, y| {
        if (12..52).contains(&x) && (12..52).contains(&y) {
            image::Rgba([200, 150, 100, 255])
        } else {
            image::Rgba([0, 0, 0, 0])
        }
    });
    let cancel = CancellationToken::default();
    let prepared = Prepared::new(&original, &cancel).unwrap();
    let foreground_linear = color::LinearImage::from_rgba(&foreground);
    let request = ForegroundInkResize {
        w: 32,
        h: 32,
        aa: GameAssetAa::new(0),
        brightness: 0.,
    };
    let black_target = prepared
        .target_contours_with_ink_source(
            &original,
            request.w,
            request.h,
            request.aa,
            InkSource {
                linear: &foreground_linear,
                suppress_unsupported: true,
                brightness: Some(request.brightness),
            },
            &cancel,
        )
        .unwrap();
    let black = prepared
        .resize_with_foreground_ink(&original, &foreground, request, &cancel)
        .unwrap();
    let owned_core: Vec<_> = black_target
        .strokes
        .core
        .as_raw()
        .iter()
        .zip(&black_target.strokes.owners)
        .enumerate()
        .filter_map(|(i, (&core, owner))| (core != 0 && owner.is_some()).then_some(i))
        .collect();
    assert!(
        !owned_core.is_empty(),
        "fixture retains a narrow contour core"
    );
    let owner = black_target.strokes.owners[owned_core[0]].unwrap();
    let intrinsic = opacity::calculate(
        &prepared.widths,
        &black_target.strokes.core,
        &black_target.strokes.owners,
    )[owner];
    let strengths: Vec<_> = [0, 1, 50, 99, 100]
        .into_iter()
        .map(|aa| {
            prepared.foreground_ink_strengths(&black_target.strokes, GameAssetAa::new(aa))[owner]
        })
        .collect();
    assert_eq!(strengths[0], 1.);
    assert_eq!(strengths[4], intrinsic);
    assert!(strengths.windows(2).all(|pair| pair[0] >= pair[1]));
    assert!(strengths[1] < 1. && strengths[3] > intrinsic);
    for i in &owned_core {
        assert_eq!(
            &black.as_raw()[i * 4..i * 4 + 4],
            &[0, 0, 0, 255],
            "AA0 owned core {i} must be solid black"
        );
    }

    let dark_target = prepared
        .target_contours_with_ink_source(
            &original,
            32,
            32,
            GameAssetAa::new(0),
            InkSource {
                linear: &foreground_linear,
                suppress_unsupported: true,
                brightness: Some(0.8),
            },
            &cancel,
        )
        .unwrap();
    let dark = prepared
        .resize_with_foreground_ink(
            &original,
            &foreground,
            ForegroundInkResize {
                w: 32,
                h: 32,
                aa: GameAssetAa::new(0),
                brightness: 0.8,
            },
            &cancel,
        )
        .unwrap();
    let clean = dark_target
        .strokes
        .coverage
        .as_raw()
        .iter()
        .enumerate()
        .find_map(|(i, &coverage)| (coverage == 0 && black.as_raw()[i * 4 + 3] == 255).then_some(i))
        .expect("fixture retains an opaque unpainted interior");
    assert_eq!(
        &black.as_raw()[clean * 4..clean * 4 + 4],
        &dark.as_raw()[clean * 4..clean * 4 + 4],
        "foreground ink brightness leaves the finished fill unchanged"
    );
    for i in owned_core {
        let expected = color::rgba([
            dark_target.colors[i][0],
            dark_target.colors[i][1],
            dark_target.colors[i][2],
            1.,
        ]);
        assert_eq!(
            &dark.as_raw()[i * 4..i * 4 + 4],
            &expected.0,
            "AA0 owned core {i} must use its exact darkened foreground donor"
        );
    }
}

#[test]
fn canonical_cleanup_drops_short_components_without_aa_ghosts() {
    let cancel = CancellationToken::default();
    for aa in [0, 100] {
        let mut strokes = strokes::Strokes {
            core: image::GrayImage::new(8, 4),
            coverage: image::GrayImage::new(8, 4),
            owners: vec![None; 32],
        };
        // Two core pixels plus a formerly antialiased neighbor. The target
        // component is below the three-pixel cutoff and must leave no paint at
        // either AA endpoint.
        for (i, coverage) in [(8 + 1, 255), (8 + 2, 255), (8 + 3, 96)] {
            if coverage == 255 {
                strokes.core.as_mut()[i] = 255;
            }
            strokes.coverage.as_mut()[i] = coverage;
            strokes.owners[i] = Some(0);
        }
        let mut colors = vec![[0.2, 0.1, 0.05]; 32];
        Prepared::canonicalize_foreground_strokes(
            &mut strokes,
            &mut colors,
            &[false; 32],
            GameAssetAa::new(aa),
            &cancel,
        )
        .unwrap();
        for i in [8 + 1, 8 + 2, 8 + 3] {
            assert_eq!(strokes.core.as_raw()[i], 0, "AA{aa} core {i}");
            assert_eq!(strokes.coverage.as_raw()[i], 0, "AA{aa} coverage {i}");
            assert_eq!(strokes.owners[i], None, "AA{aa} owner {i}");
        }
    }
}

struct ForegroundInkBeforeHaloCleanup {
    image: RgbaImage,
    support: Vec<bool>,
    fill: color::LinearImage,
    core: image::GrayImage,
    coverage: image::GrayImage,
    owners: Vec<Option<usize>>,
    colors: Vec<[f64; 3]>,
}

/// Compose the canonical foreground-ink path immediately before halo cleanup.
/// This is an integration-test baseline, not a second halo implementation.
fn foreground_ink_before_halo_cleanup(
    prepared: &Prepared,
    original: &RgbaImage,
    foreground: &RgbaImage,
    request: &ForegroundInkResize,
    cancel: &dyn Cancellation,
) -> ForegroundInkBeforeHaloCleanup {
    let fill_linear = color::LinearImage::from_rgba(foreground);
    let fill_silhouette = silhouette::Silhouette::detect(foreground, cancel)
        .unwrap()
        .expect("the fixture has a foreground silhouette");
    let TargetContours {
        mut strokes,
        mut colors,
        ..
    } = prepared
        .target_contours_with_ink_source(
            original,
            request.w,
            request.h,
            request.aa,
            InkSource {
                linear: &fill_linear,
                suppress_unsupported: true,
                brightness: Some(request.brightness),
            },
            cancel,
        )
        .unwrap();
    let FinishedFill {
        linear: fill,
        foreground_support,
        isolated,
    } = prepared
        .fill_for_target(
            &fill_linear,
            Some(&fill_silhouette),
            &strokes,
            FillTarget {
                target: (request.w, request.h),
                aa: request.aa,
                foreground_support: true,
            },
            cancel,
        )
        .unwrap();
    let support = foreground_support.expect("foreground support was requested");
    Prepared::canonicalize_foreground_strokes(
        &mut strokes,
        &mut colors,
        &support,
        request.aa,
        cancel,
    )
    .unwrap();
    let strength = prepared.foreground_ink_strengths(&strokes, request.aa);
    let paint = opacity::apply(&strokes.coverage, &strokes.owners, &strength);
    let clean = deink::apply(
        isolated.as_ref().unwrap_or(&fill_linear),
        &prepared.mask,
        fill.clone(),
        &strokes.core,
        &support,
        cancel,
    )
    .unwrap();
    let baseline = paint::composite(&clean, &colors, &paint);
    ForegroundInkBeforeHaloCleanup {
        image: baseline,
        support,
        fill,
        core: strokes.core,
        coverage: strokes.coverage,
        owners: strokes.owners,
        colors,
    }
}

fn halo_foreground_fixture(semitransparent_original: bool) -> (RgbaImage, RgbaImage) {
    let mut original = RgbaImage::from_fn(64, 64, |x, y| {
        let inside = (12..52).contains(&x) && (12..52).contains(&y);
        let ink = y == 31 && (15..49).contains(&x);
        image::Rgba(if ink {
            [0, 0, 0, 255]
        } else if inside {
            [150, 110, 80, 255]
        } else {
            [220, 230, 245, 255]
        })
    });
    if semitransparent_original {
        original.put_pixel(0, 0, image::Rgba([220, 230, 245, 254]));
    }
    let foreground = RgbaImage::from_fn(64, 64, |x, y| {
        let inside = (12..52).contains(&x) && (12..52).contains(&y);
        let ink = y == 31 && (15..49).contains(&x);
        let fringe = (14..50).contains(&x) && (27..31).contains(&y);
        image::Rgba(if ink {
            [0, 0, 0, 255]
        } else if fringe {
            // A residual from an extracted foreground remains source-supported
            // but falls below the compositor's 0.5 support threshold after
            // resampling, producing a low-alpha exterior next to original ink.
            [255, 255, 255, 20]
        } else if inside {
            [200, 150, 100, 255]
        } else {
            [255, 0, 255, 0]
        })
    });
    (original, foreground)
}

fn assert_foreground_ink_halo_cleanup(aa: u8) {
    let cancel = CancellationToken::default();
    let request = ForegroundInkResize {
        w: 31,
        h: 31,
        aa: GameAssetAa::new(aa),
        brightness: 1.,
    };
    let (w, h) = (request.w, request.h);
    let (original, foreground) = halo_foreground_fixture(false);
    let prepared = Prepared::new(&original, &cancel).unwrap();
    let before =
        foreground_ink_before_halo_cleanup(&prepared, &original, &foreground, &request, &cancel);
    let after = prepared
        .resize_with_foreground_ink(&original, &foreground, request, &cancel)
        .unwrap();
    let fringe = (0..before.support.len())
        .find(|&i| {
            !before.support[i]
                && (0. < before.fill.pixels[i][3] && before.fill.pixels[i][3] < 0.25)
                && before.core.as_raw()[i] == 0
                && before.owners[i].is_none()
                && before.image.as_raw()[i * 4] > 150
                && (-2isize..=2).any(|dy| {
                    (-2isize..=2).any(|dx| {
                        let xx = i % w as usize;
                        let yy = i / w as usize;
                        let xx = xx as isize + dx;
                        let yy = yy as isize + dy;
                        xx >= 0
                            && yy >= 0
                            && xx < w as isize
                            && yy < h as isize
                            && before.core.as_raw()[yy as usize * w as usize + xx as usize] != 0
                            && before.owners[yy as usize * w as usize + xx as usize].is_some()
                    })
                })
        })
        .expect("fixture must produce a bright low-alpha exterior fringe");
    let x = (fringe % w as usize) as u32;
    let y = (fringe / w as usize) as u32;
    let mut donor = None;
    for dy in -2isize..=2 {
        for dx in -2isize..=2 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let xx = x as isize + dx;
            let yy = y as isize + dy;
            if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                continue;
            }
            let i = yy as usize * w as usize + xx as usize;
            let Some(owner) = before.owners[i] else {
                continue;
            };
            if before.core.as_raw()[i] == 0 {
                continue;
            }
            let rank = (
                dx * dx + dy * dy,
                std::cmp::Reverse(before.coverage.as_raw()[i]),
                owner,
                i,
            );
            if donor.is_none_or(|(best, _)| rank < best) {
                donor = Some((rank, i));
            }
        }
    }
    let donor = donor
        .map(|(_, i)| i)
        .expect("the fringe must be within two pixels of an owned core donor");
    let expected = color::rgba([
        before.colors[donor][0],
        before.colors[donor][1],
        before.colors[donor][2],
        1.,
    ]);
    assert_eq!(
        &after.as_raw()[fringe * 4..fringe * 4 + 3],
        &expected.0[..3],
        "opaque originals replace exterior fringe RGB with the core donor"
    );
    assert_eq!(
        after.as_raw()[fringe * 4 + 3],
        before.image.as_raw()[fringe * 4 + 3],
        "cleanup retains the baseline alpha byte"
    );
    assert_ne!(
        &after.as_raw()[fringe * 4..fringe * 4 + 3],
        &before.image.as_raw()[fringe * 4..fringe * 4 + 3],
        "the fixture's bright fill fringe is actually repaired"
    );
    for (i, &core) in before.core.as_raw().iter().enumerate() {
        if core != 0 {
            assert_eq!(
                &after.as_raw()[i * 4..i * 4 + 4],
                &before.image.as_raw()[i * 4..i * 4 + 4],
                "core {i} stays frozen"
            );
        }
        if before.support[i] && before.coverage.as_raw()[i] == 0 {
            assert_eq!(
                &after.as_raw()[i * 4..i * 4 + 4],
                &before.image.as_raw()[i * 4..i * 4 + 4],
                "supported interior {i} stays unchanged"
            );
        }
    }

    let (transparent_original, transparent_foreground) = halo_foreground_fixture(true);
    let transparent_prepared = Prepared::new(&transparent_original, &cancel).unwrap();
    let transparent_before = foreground_ink_before_halo_cleanup(
        &transparent_prepared,
        &transparent_original,
        &transparent_foreground,
        &ForegroundInkResize {
            w,
            h,
            aa: GameAssetAa::new(aa),
            brightness: 1.,
        },
        &cancel,
    );
    let transparent_after = transparent_prepared
        .resize_with_foreground_ink(
            &transparent_original,
            &transparent_foreground,
            ForegroundInkResize {
                w,
                h,
                aa: GameAssetAa::new(aa),
                brightness: 1.,
            },
            &cancel,
        )
        .unwrap();
    assert_eq!(transparent_after, transparent_before.image);
}

#[test]
fn foreground_ink_cleans_only_opaque_originals_low_alpha_exterior_fringe() {
    for aa in [0, 100] {
        assert_foreground_ink_halo_cleanup(aa);
    }
}

#[test]
fn cancelled_cached_requests_and_working_set_preflight_are_rejected() {
    let session = Session::new(Arc::new(RgbaImage::new(12, 8)));
    session
        .resize(6, 4, GameAssetAa::default(), &CancellationToken::default())
        .unwrap();
    let cancel = CancellationToken::default();
    cancel.cancel();
    assert!(matches!(
        session.resize(6, 4, GameAssetAa::default(), &cancel),
        Err(Error::Cancelled)
    ));
    assert_eq!(working_set_estimate(2560, 1440, 1280, 720), 2_034_892_800);
    assert!(check_working_set_budget(2560, 1440, 1280, 720, DEFAULT_MEMORY_LIMIT).is_ok());
    let large = Session::new(Arc::new(RgbaImage::new(4096, 4096)));
    let error = large
        .resize(
            128,
            128,
            GameAssetAa::default(),
            &CancellationToken::default(),
        )
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Game Asset scaling would exceed the configured 4294967296 byte working-memory limit"
    );
    assert!(matches!(
        error,
        Error::GameAssetMemoryLimit {
            limit_bytes: DEFAULT_MEMORY_LIMIT
        }
    ));
    assert!(large.cache.lock().unwrap().prepared.is_none());
}

#[test]
fn crowded_owner_rerender_restores_crossing_geometry_and_its_owner_color() {
    let cancel = CancellationToken::default();
    let mut source = RgbaImage::from_pixel(48, 48, image::Rgba([255, 255, 255, 255]));
    // Each owner has a distinct source colour. B wins the initial C crossing,
    // then crowding removes it; C must receive C's blue donor when retained
    // geometry is rendered again.
    for y in 4..44 {
        source.put_pixel(4, y, image::Rgba([220, 30, 30, 255]));
        source.put_pixel(6, y, image::Rgba([20, 210, 30, 255]));
    }
    for x in 6..44 {
        source.put_pixel(x, 24, image::Rgba([30, 60, 230, 255]));
    }
    let mut prepared = Prepared::new(&source, &cancel).unwrap();
    prepared.widths = vec![8., 6., 1.];
    let line = |a: [f64; 2], b: [f64; 2]| -> coverage::Quadratic {
        [a, [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5], b]
    };
    let curves = vec![
        line([4., 4.], [4., 43.]),
        line([6., 4.], [6., 43.]),
        line([6., 24.], [43., 24.]),
    ];
    let donors = vec![
        [4., 4., 1., 0., 0., 0., 0., 0., 39., 1.],
        [6., 4., 1., 0., 0., 0., 0., 0., 39., 1.],
        [6., 24., 0., 1., 0., 0., 0., -37., 0., 1.],
    ];
    let source_linear = color::LinearImage::from_rgba(&source);
    let blue: [f64; 3] = source_linear.pixels[24 * 48 + 30][..3].try_into().unwrap();
    let crossing = 24 * 48 + 6;

    // Exercise both ordinary retained models and the production fitted-source
    // donor path. The latter is the reduced contour branch; brightness must be
    // reapplied after a rerender instead of retaining a prior owner's colour.
    for (fitted, brightness) in [(false, None), (true, Some(0.5))] {
        let geometry = TargetGeometry {
            curves: curves.clone(),
            owners: vec![0, 1, 2],
            donors: donors.clone(),
            donor_owners: vec![0, 1, 2],
            fitted_source: fitted.then(|| FittedSource {
                curves: curves.clone(),
                owners: vec![0, 1, 2],
                trace_donors: vec![
                    ([4., 4.], 0),
                    ([4., 43.], 0),
                    ([6., 4.], 1),
                    ([6., 43.], 1),
                    ([6., 24.], 2),
                    ([43., 24.], 2),
                ],
            }),
            scale: [1., 1.],
        };
        let ink = InkSource {
            linear: &source_linear,
            suppress_unsupported: false,
            brightness,
        };
        let expected_blue = brightness.map_or(blue, |value| color::scale_srgb(blue, value));

        for aa in [GameAssetAa::new(0), GameAssetAa::new(100)] {
            let all = vec![true; 3];
            let (strokes, colors) = prepared
                .render_target_geometry(
                    &geometry,
                    &all,
                    TargetRender {
                        target: (48, 48),
                        aa,
                        ink_source: ink,
                        cancel: &cancel,
                    },
                )
                .unwrap();
            assert_eq!(
                strokes.owners[crossing],
                Some(1),
                "fitted={fitted}, AA{aa:?}: B must initially own the crossing"
            );
            let mut contours = TargetContours {
                strokes,
                colors,
                geometry: TargetGeometry {
                    curves: geometry.curves.clone(),
                    owners: geometry.owners.clone(),
                    donors: donors.clone(),
                    donor_owners: vec![0, 1, 2],
                    fitted_source: geometry.fitted_source.clone(),
                    scale: [1., 1.],
                },
                suppress_unsupported: false,
                brightness,
            };
            let rejected = prepared
                .rerender_crowded_target(
                    &mut contours,
                    &[true; 48 * 48],
                    &[1., 0.1, 10.],
                    TargetRender {
                        target: (48, 48),
                        aa,
                        ink_source: ink,
                        cancel: &cancel,
                    },
                )
                .unwrap();
            assert!(rejected, "fitted={fitted}, AA{aa:?} setup must remove B");
            assert!(
                contours
                    .strokes
                    .owners
                    .iter()
                    .all(|&owner| owner != Some(1)),
                "fitted={fitted}, AA{aa:?}: rejected B must have no pixels"
            );

            let (expected, expected_colors) = prepared
                .render_target_geometry(
                    &geometry,
                    &[true, false, true],
                    TargetRender {
                        target: (48, 48),
                        aa,
                        ink_source: ink,
                        cancel: &cancel,
                    },
                )
                .unwrap();
            assert_eq!(
                contours.strokes.core, expected.core,
                "fitted={fitted}, AA{aa:?} core"
            );
            assert_eq!(
                contours.strokes.coverage, expected.coverage,
                "fitted={fitted}, AA{aa:?} coverage"
            );
            assert_eq!(
                contours.strokes.owners, expected.owners,
                "fitted={fitted}, AA{aa:?} owners"
            );
            assert_eq!(
                contours.colors, expected_colors,
                "fitted={fitted}, AA{aa:?} colours"
            );
            assert_eq!(
                contours.strokes.owners[crossing],
                Some(2),
                "fitted={fitted}, AA{aa:?}: retained C must recover B's crossing pixel"
            );
            assert_eq!(
                contours.colors[crossing], expected_blue,
                "fitted={fitted}, AA{aa:?}: recovered crossing must use C's donor and brightness"
            );
        }
    }
}

/// The approved smooth source vectors of the README elf, locked exactly.
///
/// Extraction and smoothing were tuned by eye to coloring-book quality; any
/// change to detection, skeleton cleanup, tracing or cubic fitting shows up
/// here. After an intentional change, review the new geometry and re-bless:
/// `ASSET_SCALER_BLESS=1 cargo test -p asset-scaler --lib locked_elf_source_vectors`
#[test]
fn locked_elf_source_vectors() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = image::open(directory.join("docs/images/elf-source.png"))
        .unwrap()
        .into_rgba8();
    let cancel = CancellationToken::default();
    let prepared = Prepared::new(&source, &cancel).unwrap();
    let mut svg = String::new();
    let (sw, sh) = source.dimensions();
    for (w, h) in [(sw, sh), (200, 200)] {
        use std::fmt::Write;
        let scale = [w as f64 / sw as f64, h as f64 / sh as f64];
        let fit = prepared
            .contours
            .polished(scale, contours::MAX_SHORT_PIXELS, &cancel)
            .unwrap();
        assert!(fit.max_error <= 2.25 + 1e-9, "fit error {}", fit.max_error);
        writeln!(
            svg,
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/><g fill="none" stroke="black" stroke-width="0.6" stroke-linecap="round">"#
        )
        .unwrap();
        for cubic in &fit.cubic_curves {
            let p = cubic.map(|p| [(p[0] + 0.5) * scale[0] - 0.5, (p[1] + 0.5) * scale[1] - 0.5]);
            writeln!(
                svg,
                r#"<path d="M {:.2} {:.2} C {:.2} {:.2} {:.2} {:.2} {:.2} {:.2}"/>"#,
                p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1], p[3][0], p[3][1]
            )
            .unwrap();
        }
        svg.push_str("</g></svg>\n");
    }
    let locked = directory.join("tests/fixtures/elf-source-vectors.svg");
    if std::env::var_os("ASSET_SCALER_BLESS").is_some() {
        std::fs::write(&locked, &svg).unwrap();
    }
    let expected = std::fs::read_to_string(&locked).expect("bless the locked vectors first");
    assert!(
        svg == expected,
        "source vectors changed; review and re-bless {}",
        locked.display()
    );
}
