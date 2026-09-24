//! Ignored geometry and line-quality review export.
//!
//! Writes, for one source image, the raw and thinned source masks, the fitted
//! cubics before their quadratic renderer approximation (so a raster grid
//! cannot hide a poor join), finished foreground-ink sprites with their
//! canonical cores, and a summary of bubble, fragment and fat-line counts:
//!
//! `cargo test --lib export_vector_review -- --ignored --nocapture`
//!
//! `ASSET_REVIEW_IMAGE`, `ASSET_REVIEW_OUT` and `ASSET_REVIEW_TAG` override the
//! reviewed female-elf source, `/tmp/diorama-spline-review` and `review`.
//! `ASSET_REVIEW_FOREGROUND` names the BiRefNet cutout of that source (from
//! Diorama's ignored `export_birefnet_cutout` test); sprites and contour
//! scoring use it as the removed-background foreground, as Diorama does.
use crate::{CancellationToken, cleanup, contours, detect, raster::Mask, source};
use std::fmt::Write as _;

fn save_mask(mask: &Mask, path: &str) {
    image::GrayImage::from_vec(
        mask.w as u32,
        mask.h as u32,
        mask.data
            .iter()
            .map(|&on| if on { 0 } else { 255 })
            .collect(),
    )
    .unwrap()
    .save(path)
    .unwrap();
}

fn svg(cubics: &[[[f64; 2]; 4]], w: usize, h: usize, scale: f64, width: f64) -> String {
    let mut out = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w}\" height=\"{h}\" viewBox=\"0 0 {w} {h}\">\
         <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\
         <g fill=\"none\" stroke=\"black\" stroke-width=\"{width}\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n"
    );
    let p = |q: [f64; 2]| [(q[0] + 0.5) * scale, (q[1] + 0.5) * scale];
    let mut open = false;
    let mut last = [f64::NAN; 2];
    for c in cubics {
        let [a, b, d, e] = c.map(p);
        if !open || (a[0] - last[0]).hypot(a[1] - last[1]) > 1e-6 {
            if open {
                out.push_str("\"/>\n");
            }
            write!(out, "<path d=\"M {:.3} {:.3}", a[0], a[1]).unwrap();
            open = true;
        }
        write!(
            out,
            " C {:.3} {:.3}, {:.3} {:.3}, {:.3} {:.3}",
            b[0], b[1], d[0], d[1], e[0], e[1]
        )
        .unwrap();
        last = e;
    }
    if open {
        out.push_str("\"/>\n");
    }
    out.push_str("</g></svg>\n");
    out
}

fn enclosed_holes(mask: &Mask) -> Vec<usize> {
    let (labels, sizes) = cleanup::labels(mask, false, false);
    let mut border = vec![false; sizes.len()];
    for y in 0..mask.h {
        for x in 0..mask.w {
            if x == 0 || y == 0 || x + 1 == mask.w || y + 1 == mask.h {
                border[labels[y * mask.w + x]] = true;
            }
        }
    }
    (1..sizes.len())
        .filter(|&id| !border[id])
        .map(|id| sizes[id])
        .collect()
}

/// Count visibly dark pixel defects in a finished sprite: fully dark 2x2
/// blocks (a line two pixels wide) and L corners, where a dark pixel joins a
/// horizontal and a vertical dark neighbour whose shared diagonal is light,
/// so a diagonal Bresenham step would have been enough.
fn line_defects(sprite: &image::RgbaImage) -> (usize, usize) {
    let (w, h) = (sprite.width() as i64, sprite.height() as i64);
    let dark = |x: i64, y: i64| {
        x >= 0 && y >= 0 && x < w && y < h && {
            let p = sprite.get_pixel(x as u32, y as u32);
            p[3] >= 128
                && (0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64) < 35.
        }
    };
    let mut fat = 0;
    let mut corners = 0;
    for y in 0..h {
        for x in 0..w {
            if !dark(x, y) {
                continue;
            }
            if dark(x + 1, y) && dark(x, y + 1) && dark(x + 1, y + 1) {
                fat += 1;
            }
            for (dx, dy) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
                if dark(x + dx, y) && dark(x, y + dy) && !dark(x + dx, y + dy) {
                    corners += 1;
                }
            }
        }
    }
    (fat, corners)
}

/// Defects attributable to the drawn contour: dark non-core pixels
/// 4-adjacent to the canonical core, fully dark 2x2 blocks containing a core
/// pixel, and core pixels whose only two core neighbours form an L (a
/// diagonal step would connect them without it).
fn core_defects(
    sprite: &image::RgbaImage,
    core: &image::GrayImage,
) -> (usize, usize, usize, usize) {
    let (w, h) = (sprite.width() as i64, sprite.height() as i64);
    let on = |x: i64, y: i64| {
        x >= 0 && y >= 0 && x < w && y < h && core.get_pixel(x as u32, y as u32)[0] == 0
    };
    let dark = |x: i64, y: i64| {
        x >= 0 && y >= 0 && x < w && y < h && {
            let p = sprite.get_pixel(x as u32, y as u32);
            p[3] >= 128
                && (0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64) < 35.
        }
    };
    let background = |x: i64, y: i64| {
        x >= 0 && y >= 0 && x < w && y < h && sprite.get_pixel(x as u32, y as u32)[3] < 128
    };
    // An outer core touches the removed background.
    let outer = |x: i64, y: i64| {
        on(x, y) && (-1..=1).any(|dy| (-1..=1).any(|dx| background(x + dx, y + dy)))
    };
    let (mut beside, mut beside_outer, mut fat, mut staircase) = (0, 0, 0, 0);
    for y in 0..h {
        for x in 0..w {
            if !on(x, y)
                && dark(x, y)
                && [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .any(|&(dx, dy)| on(x + dx, y + dy))
            {
                beside += 1;
            }
            if !on(x, y)
                && dark(x, y)
                && [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .any(|&(dx, dy)| outer(x + dx, y + dy))
            {
                beside_outer += 1;
            }
            let block = [(0, 0), (1, 0), (0, 1), (1, 1)];
            if block.iter().all(|&(dx, dy)| dark(x + dx, y + dy))
                && block.iter().any(|&(dx, dy)| on(x + dx, y + dy))
            {
                fat += 1;
            }
            if on(x, y) {
                let neighbours: Vec<_> = (-1..=1)
                    .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
                    .filter(|&(dx, dy)| (dx, dy) != (0, 0) && on(x + dx, y + dy))
                    .collect();
                if let [a, b] = neighbours[..]
                    && (a.0 == 0) != (b.0 == 0)
                    && (a.0 + b.0).abs() == 1
                    && (a.1 + b.1).abs() == 1
                {
                    staircase += 1;
                }
            }
        }
    }
    (beside, beside_outer, fat, staircase)
}

#[test]
#[ignore = "manual vector and line-quality review export"]
fn export_vector_review() {
    let image = std::env::var("ASSET_REVIEW_IMAGE").unwrap_or_else(|_| {
        "/home/mendrik/desk/mendrik/mule/assets/characters/01-female-elf.png".into()
    });
    let out =
        std::env::var("ASSET_REVIEW_OUT").unwrap_or_else(|_| "/tmp/diorama-spline-review".into());
    std::fs::create_dir_all(&out).unwrap();
    let tag = std::env::var("ASSET_REVIEW_TAG").unwrap_or_else(|_| "review".into());
    let image = image::open(image).unwrap().into_rgba8();
    let (w, h) = (image.width() as usize, image.height() as usize);
    let cancel = CancellationToken::default();
    let started = std::time::Instant::now();
    let samples = detect::detect(&image, &cancel).unwrap();
    let models = detect::fit_models(&samples, 0.012, &cancel).unwrap();
    let (raw, _) = source::rasterize(&models, w, h, &cancel).unwrap();
    let thinned = source::skeleton(&models, w, h, &cancel).unwrap();
    let contours = contours::Contours::new(&thinned, &models, &cancel).unwrap();
    let prepared = started.elapsed();
    save_mask(&raw, &format!("{out}/{tag}-raw.png"));
    save_mask(&thinned, &format!("{out}/{tag}-thinned.png"));
    let mut summary = String::new();
    let holes = enclosed_holes(&raw);
    writeln!(
        summary,
        "{tag}: samples {} models {} prepared {:.2?}\nraw mask enclosed holes: {} (<=16px: {}, <=48px: {})",
        samples.len(),
        models.len(),
        prepared,
        holes.len(),
        holes.iter().filter(|&&a| a <= 16).count(),
        holes.iter().filter(|&&a| a <= 48).count(),
    )
    .unwrap();
    let lengths = &contours.lengths;
    writeln!(
        summary,
        "traced paths: {} (length < 4px: {}, < 8px: {}); total length {:.0}px",
        lengths.len(),
        lengths.iter().filter(|&&l| l < 4.).count(),
        lengths.iter().filter(|&&l| l < 8.).count(),
        lengths.iter().sum::<f64>(),
    )
    .unwrap();
    for target in [w, 256, 200] {
        let scale = target as f64 / w as f64;
        let fit = contours.polished([scale, scale], 0, &cancel).unwrap();
        std::fs::write(
            format!("{out}/{tag}-fitted-{target}.svg"),
            svg(
                &fit.cubic_curves,
                target,
                target * h / w,
                scale,
                if target == w { 0.9 } else { 0.5 },
            ),
        )
        .unwrap();
        writeln!(
            summary,
            "{target}px: {} cubic segments, {} quadratics, max fit error {:.3} source px",
            fit.cubic_curves.len(),
            fit.curves.len(),
            fit.max_error,
        )
        .unwrap();
    }
    if std::env::var("ASSET_REVIEW_OWNERS").is_ok() {
        let prepared = crate::Prepared::new(&image, &cancel).unwrap();
        let support = prepared.silhouette.as_ref().map(|s| s.support.clone());
        let mut dump = String::new();
        for (id, trace) in prepared.contours.traces().enumerate() {
            let outer = support.as_ref().map_or(0., |support| {
                trace
                    .iter()
                    .filter(|p| {
                        (-3..=3).any(|dy| {
                            (-3..=3).any(|dx| !support.at(p[0] as isize + dx, p[1] as isize + dy))
                        })
                    })
                    .count() as f64
                    / trace.len().max(1) as f64
            });
            let pts: Vec<String> = trace
                .iter()
                .map(|p| format!("{:.0},{:.0}", p[0], p[1]))
                .collect();
            writeln!(
                dump,
                "{id} {:.1} {:.2} {:.2} {}",
                prepared.contours.lengths[id],
                prepared.widths[id],
                outer,
                pts.join(";")
            )
            .unwrap();
        }
        std::fs::write(format!("{out}/{tag}-owners.txt"), dump).unwrap();
    }
    // Contours are scored against the real background removal, exactly as
    // Diorama composes them: the foreground is a BiRefNet cutout of this
    // image (Diorama's ignored `export_birefnet_cutout` test writes one).
    let foreground_path = std::env::var("ASSET_REVIEW_FOREGROUND")
        .unwrap_or_else(|_| "/tmp/diorama-spline-review/female-elf-birefnet-foreground.png".into());
    let foreground = image::open(&foreground_path)
        .unwrap_or_else(|error| {
            panic!("a BiRefNet foreground is required at {foreground_path}: {error}")
        })
        .into_rgba8();
    assert_eq!(foreground.dimensions(), image.dimensions());
    let session = crate::Session::new(std::sync::Arc::new(image.clone()));
    for target in [128u32, 200, 256, 512] {
        for aa in [0u8, 50] {
            let th = (target as usize * h / w) as u32;
            let started = std::time::Instant::now();
            let sprite = session
                .resize_with_foreground_ink(
                    &foreground,
                    target,
                    th,
                    crate::GameAssetAa::new(aa),
                    0.8,
                    &cancel,
                )
                .unwrap();
            let elapsed = started.elapsed();
            let (fat, corners) = line_defects(&sprite);
            let core = session
                .foreground_contour_mask(
                    &foreground,
                    target,
                    th,
                    crate::GameAssetAa::new(aa),
                    &cancel,
                )
                .unwrap();
            let (beside, beside_outer, core_fat, staircase) = core_defects(&sprite, &core);
            writeln!(
                summary,
                "sprite {target}px AA{aa} ({elapsed:.2?}): dark 2x2 blocks {fat}, dark L corners {corners}; \
                 core: dark non-core 4-neighbours {beside} (of outer cores {beside_outer}), dark 2x2 on core {core_fat}, core L steps {staircase}"
            )
            .unwrap();
            sprite
                .save(format!("{out}/{tag}-sprite-{target}-aa{aa}.png"))
                .unwrap();
            core.save(format!("{out}/{tag}-core-{target}-aa{aa}.png"))
                .unwrap();
        }
    }
    std::fs::write(format!("{out}/{tag}-summary.txt"), &summary).unwrap();
    print!("{summary}");
}
