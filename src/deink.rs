//! Remove resampled outline ink beside the drawn silhouette edge.
//!
//! The direct Lanczos base still contains the painted outline, spread over
//! two or three target pixels. Where a drawn core edges the removed
//! background, that spread doubles the outline into fat pixels or L-shaped
//! steps. Only pixels whose nearest core is such an outer core are repaired:
//! an interior pixel whose footprint holds enough ink-free foreground takes
//! that local colour, and one covered entirely by the outline takes the
//! painted interior just beyond it. A pixel on the silhouette edge is never
//! refilled; it becomes transparent only when it holds nothing but outline
//! ink and background, so the drawn core becomes the edge without gaps.
//!
//! Interior contours are never de-inked. Their surroundings are painted
//! detail (faces, folds, fills) that the resampled base must keep; a pixel
//! beside any interior core stays untouched even when an outline is near.
use crate::{Cancellation, Result, color, raster::Mask};
use image::{ImageBuffer, Rgba, imageops::FilterType};

/// Target pixels within this Chebyshev distance of an outer core are
/// eligible: the Lanczos3 base smears an outline up to two pixels inward.
const REACH: isize = 2;
/// How far an entirely inked pixel looks away from the outline for the
/// nearest ink-free fill colour.
const BORROW: isize = 3;
/// Ink fraction of a target footprint below which the base is left as is.
const MIN_INK: f32 = 0.02;
/// Ink-free foreground weight needed before its local colour is trusted.
const MIN_EVIDENCE: f32 = 0.2;
/// Largest ink-free foreground share of a pixel that may become transparent.
/// Anything more is painted fill (a thin bow limb, a strap) and stays.
const MAX_CLEARED_FILL: f32 = 0.05;

type Plane = ImageBuffer<Rgba<f32>, Vec<f32>>;

/// `support` is the target foreground support: `false` is removed background.
pub(crate) fn apply(
    source: &color::LinearImage,
    ink: &Mask,
    mut base: color::LinearImage,
    core: &image::GrayImage,
    support: &[bool],
    cancel: &dyn Cancellation,
) -> Result<color::LinearImage> {
    cancel.check()?;
    assert_eq!((source.w, source.h), (ink.w, ink.h));
    assert_eq!((base.w as u32, base.h as u32), core.dimensions());
    let (w, h) = (base.w, base.h);
    assert_eq!(support.len(), w * h);
    let core = core.as_raw();
    let inside = |x: isize, y: isize| x >= 0 && y >= 0 && x < w as isize && y < h as isize;
    let is_core = |x: isize, y: isize| inside(x, y) && core[y as usize * w + x as usize] != 0;
    // The image border is not removed background; only unsupported pixels are.
    let unsupported = |x: isize, y: isize| inside(x, y) && !support[y as usize * w + x as usize];
    let near = |x: isize, y: isize, test: &dyn Fn(isize, isize) -> bool| {
        (-1..=1).any(|dy| (-1..=1).any(|dx| (dx, dy) != (0, 0) && test(x + dx, y + dy)))
    };
    // An outer core touches removed background; every other core is interior.
    let outer: Vec<bool> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            core[i] != 0 && near(x, y, &unsupported)
        })
        .collect();
    let is_outer = |x: isize, y: isize| inside(x, y) && outer[y as usize * w + x as usize];
    // Each eligible pixel remembers the unit step leading away from its
    // nearest outer core, towards the painted interior.
    let mut away: Vec<Option<(isize, isize)>> = vec![None; w * h];
    let mut any = false;
    for y in 0..h as isize {
        cancel.check()?;
        for x in 0..w as isize {
            let i = y as usize * w + x as usize;
            if core[i] != 0 || base.pixels[i][3] <= 1e-8 {
                continue;
            }
            if near(x, y, &|x, y| is_core(x, y) && !is_outer(x, y)) {
                continue;
            }
            let mut nearest = None;
            for dy in -REACH..=REACH {
                for dx in -REACH..=REACH {
                    if is_core(x + dx, y + dy)
                        && nearest.is_none_or(|(d, _, _)| dx * dx + dy * dy < d)
                    {
                        nearest = Some((dx * dx + dy * dy, x + dx, y + dy));
                    }
                }
            }
            let Some((_, cx, cy)) = nearest else {
                continue;
            };
            if !is_outer(cx, cy) {
                continue;
            }
            // Ink ahead of a core endpoint continues the drawn line; only ink
            // beside a core duplicates it.
            let mut neighbours = (-1..=1)
                .flat_map(|dy| (-1..=1).map(move |dx| (dx, dy)))
                .filter(|&(nx, ny)| (nx, ny) != (0, 0) && is_core(cx + nx, cy + ny));
            if let (Some((nx, ny)), None) = (neighbours.next(), neighbours.next())
                && (x - cx) * -nx + (y - cy) * -ny > 0
            {
                continue;
            }
            away[i] = Some(((x - cx).signum(), (y - cy).signum()));
            any = true;
        }
    }
    if !any {
        return Ok(base);
    }
    // Premultiplied ink-free colour and its weight, and the ink weight.
    // Transparent pixels contribute to neither, so the estimate never
    // borrows hidden RGB or the canvas.
    let mut clean = Vec::with_capacity(source.pixels.len() * 4);
    let mut inked = Vec::with_capacity(source.pixels.len() * 4);
    for (i, p) in source.pixels.iter().enumerate() {
        if i.is_multiple_of(4096) {
            cancel.check()?;
        }
        let a = p[3] as f32;
        let (clean_a, ink_a) = if ink.data[i] { (0., a) } else { (a, 0.) };
        clean.extend([
            p[0] as f32 * clean_a,
            p[1] as f32 * clean_a,
            p[2] as f32 * clean_a,
            clean_a,
        ]);
        inked.extend([0., 0., 0., ink_a]);
    }
    let clean = Plane::from_vec(source.w as u32, source.h as u32, clean)
        .expect("LinearImage pixel dimensions match its storage");
    let inked = Plane::from_vec(source.w as u32, source.h as u32, inked)
        .expect("LinearImage pixel dimensions match its storage");
    // Measure ink presence with the base's own Lanczos3 support; the local
    // split into ink, fill and background uses a positive footprint.
    let ink_fraction = image::imageops::resize(&inked, w as u32, h as u32, FilterType::Lanczos3);
    let clean = image::imageops::resize(&clean, w as u32, h as u32, FilterType::Triangle);
    let inked = image::imageops::resize(&inked, w as u32, h as u32, FilterType::Triangle);
    let fill_at = |x: isize, y: isize| {
        (inside(x, y) && support[y as usize * w + x as usize])
            .then(|| clean.get_pixel(x as u32, y as u32).0)
            .filter(|e| e[3] >= MIN_EVIDENCE)
    };
    for (i, pixel) in base.pixels.iter_mut().enumerate() {
        if i.is_multiple_of(4096) {
            cancel.check()?;
        }
        let Some((sx, sy)) = away[i] else {
            continue;
        };
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        if ink_fraction.get_pixel(x as u32, y as u32).0[3].abs() < MIN_INK {
            continue;
        }
        let e = clean.get_pixel(x as u32, y as u32).0;
        let exterior = (1. - e[3] - inked.get_pixel(x as u32, y as u32).0[3]).max(0.);
        let edge = near(x, y, &unsupported);
        let estimate = if edge {
            // A silhouette-edge pixel is the visible outline wherever the
            // core runs just inside it. It is never refilled: pure outline
            // ink beside the background goes, anything else stays.
            if e[3] <= MAX_CLEARED_FILL && exterior > e[3] {
                *pixel = [0.; 4];
            }
            continue;
        } else if e[3] >= MIN_EVIDENCE {
            Some(e)
        } else {
            // A wide outline covers this pixel entirely: its colour is the
            // painted interior just beyond the outline.
            (1..=BORROW).find_map(|k| fill_at(x + k * sx, y + k * sy))
        };
        if let Some(e) = estimate {
            for c in 0..3 {
                pixel[c] = f64::from(e[c] / e[3]).clamp(0., 1.);
            }
        }
    }
    Ok(base)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CancellationToken, lanczos};

    const FILL: [f64; 4] = [0.5, 0.2, 0.05, 1.];

    /// Fill, then a 6px outline at source x 30..36, then `right` beyond it.
    /// The outline ends on a target pixel boundary at half resolution.
    fn outlined(right: [f64; 4]) -> (color::LinearImage, Mask) {
        let (w, h) = (64, 32);
        let source = color::LinearImage {
            w,
            h,
            pixels: (0..w * h)
                .map(|i| match i % w {
                    30..36 => [0.005, 0.005, 0.005, 1.],
                    36.. => right,
                    _ => FILL,
                })
                .collect(),
        };
        let mask = Mask {
            w,
            h,
            data: (0..w * h).map(|i| (30..36).contains(&(i % w))).collect(),
        };
        (source, mask)
    }

    fn column_core(w: u32, h: u32, x: u32) -> image::GrayImage {
        let mut core = image::GrayImage::new(w, h);
        for y in 0..h {
            core.put_pixel(x, y, image::Luma([255]));
        }
        core
    }

    fn support(base: &color::LinearImage) -> Vec<bool> {
        base.pixels.iter().map(|p| p[3] >= 0.5).collect()
    }

    #[test]
    fn interior_strokes_and_their_painted_surroundings_are_never_touched() {
        let cancel = CancellationToken::default();
        let (source, mask) = outlined(FILL);
        let base = lanczos::resize(&source, 32, 16, &cancel).unwrap();
        let before = base.clone();
        let support = support(&base);
        let cleaned = apply(
            &source,
            &mask,
            base,
            &column_core(32, 16, 16),
            &support,
            &cancel,
        )
        .unwrap();
        assert!(cleaned.pixels == before.pixels);
    }

    #[test]
    fn outer_outline_keeps_only_its_drawn_core() {
        let cancel = CancellationToken::default();
        let (source, mask) = outlined([0.; 4]);
        let base = lanczos::resize(&source, 32, 16, &cancel).unwrap();
        let support = support(&base);
        // The canonical exterior core is the outermost supported pixel.
        let edge = (0..32).rev().find(|&x| support[8 * 32 + x]).unwrap();
        let before = base.clone();
        let cleaned = apply(
            &source,
            &mask,
            base,
            &column_core(32, 16, edge as u32),
            &support,
            &cancel,
        )
        .unwrap();
        for y in 0..16 {
            let i = y * 32 + edge;
            assert_eq!(cleaned.pixels[i], before.pixels[i], "core changed");
            assert!(
                before.pixels[i - 1][0] < 0.3,
                "fixture must smear ink inward"
            );
            for (value, expected) in cleaned.pixels[i - 1][..3].iter().zip(FILL) {
                assert!((value - expected).abs() < 0.02, "inner ink left");
            }
            assert_eq!(cleaned.pixels[i - 1][3], before.pixels[i - 1][3]);
            if before.pixels[i + 1][3] > 0. {
                assert_eq!(cleaned.pixels[i + 1], [0.; 4], "outer ink half must go");
            }
            // Beyond the outline's reach nothing changes.
            for x in (0..edge - 2).chain(edge + 3..32) {
                assert_eq!(cleaned.pixels[y * 32 + x], before.pixels[y * 32 + x]);
            }
        }
    }

    #[test]
    fn a_pixel_beside_an_interior_core_is_kept_even_next_to_an_outline() {
        let cancel = CancellationToken::default();
        let (source, mask) = outlined([0.; 4]);
        let base = lanczos::resize(&source, 32, 16, &cancel).unwrap();
        let support = support(&base);
        let edge = (0..32).rev().find(|&x| support[8 * 32 + x]).unwrap();
        let mut core = column_core(32, 16, edge as u32);
        for y in 0..16 {
            core.put_pixel(edge as u32 - 2, y, image::Luma([255]));
        }
        let before = base.clone();
        let cleaned = apply(&source, &mask, base, &core, &support, &cancel).unwrap();
        for y in 0..16 {
            let i = y * 32 + edge - 1;
            assert_eq!(cleaned.pixels[i], before.pixels[i]);
        }
    }
}
