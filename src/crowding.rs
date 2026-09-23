//! Whole-contour arbitration for crowded target-resolution ink.
//!
//! This pass never changes a pixel owner or colour.  It returns one decision
//! per source contour so its caller can remove a rejected contour coherently,
//! before target-union cleanup borrows any antialias fringe ownership.

use crate::{Cancellation, Error, Result, strokes::Strokes};

/// The target-pixel reach used to identify contours that collapse together.
/// It stays fixed in target space, so the same source detail is tested more
/// aggressively as the requested output gets smaller.
const CROWD_RADIUS: isize = 2;
/// A contact must cover this fraction of a contour's rendered core to remove
/// it.  This prevents a single junction or endpoint touch from deleting a
/// complete branch.
const MIN_CROWDED_FRACTION_NUMERATOR: usize = 3;
const MIN_CROWDED_FRACTION_DENOMINATOR: usize = 5;
const MIN_CROWDED_PIXELS: usize = 4;

/// Select the target contour owners that should remain visible.
///
/// `widths` and `source_strengths` are per-source-contour measurements.  A
/// larger finite value in either input increases an owner's deterministic
/// rank. Pixels adjacent to unsupported foreground (or the image edge) are
/// exterior contours and are always retained. Unsupported foreground includes
/// internal alpha holes and narrow transparent gaps, not only the canvas
/// exterior. Other owners are rejected only
/// when a stronger owner remains parallel and within [`CROWD_RADIUS`] for a
/// substantial part of their target core.
///
/// The work is O(target pixels) with a fixed 5x5 neighborhood per core pixel;
/// it does not compare arbitrary pairs of contours or pixels.
pub(crate) fn select(
    strokes: &Strokes,
    foreground_support: &[bool],
    widths: &[f64],
    source_strengths: &[f64],
    cancel: &dyn Cancellation,
) -> Result<Vec<bool>> {
    cancel.check()?;
    let (w, h) = (
        strokes.core.width() as usize,
        strokes.core.height() as usize,
    );
    let Some(len) = w.checked_mul(h) else {
        return Err(invalid_inputs());
    };
    if w == 0
        || h == 0
        || strokes.core.as_raw().len() != len
        || strokes.coverage.dimensions() != strokes.core.dimensions()
        || strokes.coverage.as_raw().len() != len
        || strokes.owners.len() != len
        || foreground_support.len() != len
        || widths.len() != source_strengths.len()
        || widths.iter().any(|value| !value.is_finite() || *value < 0.)
        || source_strengths
            .iter()
            .any(|value| !value.is_finite() || *value < 0.)
        || strokes
            .owners
            .iter()
            .flatten()
            .any(|&owner| owner >= widths.len())
    {
        return Err(invalid_inputs());
    }

    let mut owner_pixels = vec![Vec::new(); widths.len()];
    let mut outer = vec![false; widths.len()];
    let mut direction = vec![None; len];
    for (i, direction_at) in direction.iter_mut().enumerate() {
        if i.is_multiple_of(4096) {
            cancel.check()?;
        }
        if strokes.core.as_raw()[i] == 0 {
            continue;
        }
        let Some(owner) = strokes.owners[i] else {
            return Err(invalid_inputs());
        };
        owner_pixels[owner].push(i);
        outer[owner] |= touches_exterior(i, w, h, foreground_support);
        *direction_at = tangent(i, owner, w, h, &strokes.owners, strokes.core.as_raw());
    }

    // Empty owners have no core to crowd. Keeping them makes this a pure
    // selection decision; the existing renderer/canonicalizer owns their AA
    // handling and may already have cleared them.
    let mut order: Vec<_> = (0..widths.len()).collect();
    order.sort_by(|&a, &b| {
        outer[b]
            .cmp(&outer[a])
            .then_with(|| {
                source_rank(widths[b], source_strengths[b])
                    .total_cmp(&source_rank(widths[a], source_strengths[a]))
            })
            .then_with(|| owner_pixels[b].len().cmp(&owner_pixels[a].len()))
            .then_with(|| a.cmp(&b))
    });
    let mut keep = vec![false; widths.len()];
    for (position, owner) in order.into_iter().enumerate() {
        if position.is_multiple_of(4096) {
            cancel.check()?;
        }
        if outer[owner] || owner_pixels[owner].is_empty() {
            keep[owner] = true;
            continue;
        }
        let mut crowded = 0;
        for (pixel, &i) in owner_pixels[owner].iter().enumerate() {
            if pixel.is_multiple_of(4096) {
                cancel.check()?;
            }
            let Some(tangent) = direction[i] else {
                // A one-pixel fragment has no reliable direction, therefore
                // no geometric evidence to remove a whole contour.
                continue;
            };
            if has_kept_parallel_neighbor(
                i,
                owner,
                tangent,
                w,
                h,
                strokes.core.as_raw(),
                &strokes.owners,
                &direction,
                &keep,
            ) {
                crowded += 1;
            }
        }
        keep[owner] = crowded < MIN_CROWDED_PIXELS
            || crowded * MIN_CROWDED_FRACTION_DENOMINATOR
                < owner_pixels[owner].len() * MIN_CROWDED_FRACTION_NUMERATOR;
    }
    cancel.check()?;
    Ok(keep)
}

fn invalid_inputs() -> Error {
    Error::Scaling("Invalid contour crowding inputs".into())
}

/// Rank source evidence first and target length second. The logarithm keeps a
/// very thick detected band from overwhelming a substantially stronger edge.
fn source_rank(width: f64, source_strength: f64) -> f64 {
    (1. + width.ln_1p()) * (1. + 3. * source_strength.sqrt())
}

fn touches_exterior(i: usize, w: usize, h: usize, support: &[bool]) -> bool {
    let x = i % w;
    let y = i / w;
    if x == 0 || y == 0 || x + 1 == w || y + 1 == h || !support[i] {
        return true;
    }
    for dy in -1isize..=1 {
        for dx in -1isize..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let xx = x as isize + dx;
            let yy = y as isize + dy;
            if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                return true;
            }
            if !support[yy as usize * w + xx as usize] {
                return true;
            }
        }
    }
    false
}

fn tangent(
    i: usize,
    owner: usize,
    w: usize,
    h: usize,
    owners: &[Option<usize>],
    core: &[u8],
) -> Option<(i8, i8)> {
    let x = (i % w) as isize;
    let y = (i / w) as isize;
    let mut points = [(0i8, 0i8); 9];
    let mut count = 1;
    for dy in -1isize..=1 {
        for dx in -1isize..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let xx = x + dx;
            let yy = y + dy;
            if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                continue;
            }
            let j = yy as usize * w + xx as usize;
            if core[j] != 0 && owners[j] == Some(owner) {
                points[count] = (dx as i8, dy as i8);
                count += 1;
            }
        }
    }
    let mut best = None;
    for a in 0..count {
        for b in a + 1..count {
            let vector = (points[b].0 - points[a].0, points[b].1 - points[a].1);
            let length2 = vector.0 * vector.0 + vector.1 * vector.1;
            if best.is_none_or(|(_, old)| length2 > old) {
                best = Some((vector, length2));
            }
        }
    }
    best.map(|((dx, dy), _)| (dx, dy))
}

#[allow(clippy::too_many_arguments)]
fn has_kept_parallel_neighbor(
    i: usize,
    owner: usize,
    tangent: (i8, i8),
    w: usize,
    h: usize,
    core: &[u8],
    owners: &[Option<usize>],
    directions: &[Option<(i8, i8)>],
    keep: &[bool],
) -> bool {
    let x = (i % w) as isize;
    let y = (i / w) as isize;
    for dy in -CROWD_RADIUS..=CROWD_RADIUS {
        for dx in -CROWD_RADIUS..=CROWD_RADIUS {
            if dx == 0 && dy == 0 {
                continue;
            }
            let xx = x + dx;
            let yy = y + dy;
            if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                continue;
            }
            let j = yy as usize * w + xx as usize;
            let Some(other) = owners[j] else {
                continue;
            };
            if other != owner
                && keep[other]
                && core[j] != 0
                && directions[j].is_some_and(|other_tangent| parallel(tangent, other_tangent))
            {
                return true;
            }
        }
    }
    false
}

/// Treat directions up to 45 degrees apart as parallel.  The sign is ignored:
/// a contour may be rasterized in either direction.
fn parallel(a: (i8, i8), b: (i8, i8)) -> bool {
    let (ax, ay) = (i32::from(a.0), i32::from(a.1));
    let (bx, by) = (i32::from(b.0), i32::from(b.1));
    let dot = ax * bx + ay * by;
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    2 * dot * dot >= a2 * b2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CancellationToken;
    use image::{GrayImage, Luma};

    fn strokes(w: u32, h: u32, paths: &[(usize, &[(u32, u32)])]) -> Strokes {
        let mut core = GrayImage::new(w, h);
        let mut coverage = GrayImage::new(w, h);
        let mut owners = vec![None; (w * h) as usize];
        for &(owner, pixels) in paths {
            for &(x, y) in pixels {
                let i = (y * w + x) as usize;
                assert!(owners[i].is_none(), "test owners must not overlap");
                core.put_pixel(x, y, Luma([255]));
                coverage.put_pixel(x, y, Luma([255]));
                owners[i] = Some(owner);
            }
        }
        Strokes {
            core,
            coverage,
            owners,
        }
    }

    fn horizontal(y: u32, start: u32, end: u32) -> Vec<(u32, u32)> {
        (start..=end).map(|x| (x, y)).collect()
    }

    #[test]
    fn drops_a_weaker_parallel_inset_but_keeps_separated_lines() {
        let strong = horizontal(2, 1, 9);
        let crowded = horizontal(4, 1, 9);
        let image = strokes(11, 8, &[(0, &strong), (1, &crowded)]);
        let keep = select(
            &image,
            &[true; 88],
            &[6., 1.],
            &[1., 0.1],
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [true, false]);

        let separated = horizontal(7, 1, 9);
        let image = strokes(11, 10, &[(0, &strong), (1, &separated)]);
        let keep = select(
            &image,
            &[true; 110],
            &[6., 1.],
            &[1., 0.1],
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [true, true]);
    }

    #[test]
    fn exterior_keeps_two_nearby_silhouette_edges() {
        let upper = horizontal(2, 1, 9);
        let lower = horizontal(4, 1, 9);
        let image = strokes(11, 7, &[(0, &upper), (1, &lower)]);
        let mut support = vec![true; 77];
        for x in 0..11 {
            support[11 + x] = false;
            support[5 * 11 + x] = false;
        }
        let keep = select(
            &image,
            &support,
            &[6., 1.],
            &[1., 0.1],
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [true, true]);
    }

    #[test]
    fn junctions_do_not_count_as_parallel_crowding() {
        let horizontal = horizontal(4, 1, 9);
        let vertical: Vec<_> = (1..=7).filter(|&y| y != 4).map(|y| (5, y)).collect();
        let image = strokes(11, 9, &[(0, &horizontal), (1, &vertical)]);
        let keep = select(
            &image,
            &[true; 99],
            &[6., 1.],
            &[1., 0.1],
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [true, true]);
    }

    #[test]
    fn drops_crowded_u_inset_but_keeps_a_separate_closed_shape() {
        let boundary = horizontal(1, 1, 9);
        let mut inset = horizontal(3, 2, 8);
        // The short arms touch the outer line at both ends. Their contacts are
        // perpendicular; the sustained parallel top is the removal evidence.
        inset.extend([(2, 2), (8, 2)]);
        let loop_pixels = vec![
            (1, 8),
            (2, 8),
            (3, 8),
            (1, 9),
            (3, 9),
            (1, 10),
            (2, 10),
            (3, 10),
        ];
        let image = strokes(11, 12, &[(0, &boundary), (1, &inset), (2, &loop_pixels)]);
        let mut support = vec![true; 132];
        support[..11].fill(false);
        let keep = select(
            &image,
            &support,
            &[6., 1., 1.],
            &[1., 0.1, 0.1],
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [true, false, true]);
    }

    #[test]
    fn source_strength_and_thickness_are_both_priority_evidence() {
        let upper = horizontal(2, 1, 9);
        let lower = horizontal(4, 1, 9);
        let image = strokes(11, 8, &[(0, &upper), (1, &lower)]);
        let keep = select(
            &image,
            &[true; 88],
            &[2., 7.],
            &[1., 0.],
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [true, false]);

        let keep = select(
            &image,
            &[true; 88],
            &[2., 7.],
            &[0., 1.],
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [false, true]);
    }

    #[test]
    fn outer_edge_dominates_interior_and_rejected_contours_do_not_form_a_chain() {
        let outer = horizontal(2, 1, 9);
        let interior = horizontal(4, 1, 9);
        let image = strokes(11, 8, &[(0, &outer), (1, &interior)]);
        let mut support = vec![true; 88];
        for x in 0..11 {
            support[11 + x] = false;
        }
        let keep = select(
            &image,
            &support,
            &[1., 9.],
            &[0., 2.],
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [true, false]);

        let strongest = horizontal(2, 1, 11);
        let middle = horizontal(4, 1, 11);
        let weakest = horizontal(6, 1, 11);
        let image = strokes(13, 9, &[(0, &strongest), (1, &middle), (2, &weakest)]);
        let keep = select(
            &image,
            &[true; 117],
            &[9., 5., 1.],
            &[1., 0.5, 0.1],
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [true, false, true]);
    }

    #[test]
    fn enclosed_transparent_hole_boundary_beats_a_nearby_strong_interior_line() {
        let hole_boundary = horizontal(5, 2, 8);
        let interior = horizontal(7, 2, 8);
        let image = strokes(12, 12, &[(0, &hole_boundary), (1, &interior)]);
        // This unsupported area is completely inside the target. It models a
        // transparent hole rather than an image-border exterior.
        let mut support = vec![true; 12 * 12];
        for y in 3..=4 {
            for x in 3..=7 {
                support[y * 12 + x] = false;
            }
        }
        let keep = select(
            &image,
            &support,
            &[1., 8.],
            &[0., 1.],
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [true, false]);

        // With no alpha hole, ordinary source evidence determines the winner.
        let keep = select(
            &image,
            &[true; 12 * 12],
            &[1., 8.],
            &[0., 1.],
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [false, true]);
    }

    #[test]
    fn opposite_edges_of_an_internal_transparent_gap_are_both_retained() {
        let upper = horizontal(5, 2, 8);
        let lower = horizontal(7, 2, 8);
        let image = strokes(12, 12, &[(0, &upper), (1, &lower)]);
        let mut support = vec![true; 12 * 12];
        for x in 2..=8 {
            support[6 * 12 + x] = false;
        }
        let keep = select(
            &image,
            &support,
            &[8., 1.],
            &[1., 0.],
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [true, true]);

        let keep = select(
            &image,
            &[true; 12 * 12],
            &[8., 1.],
            &[1., 0.],
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [true, false]);
    }

    #[test]
    fn validates_inputs_and_observes_cancellation() {
        let image = strokes(2, 2, &[(0, &[(0, 0), (1, 0)])]);
        assert!(
            select(
                &image,
                &[true; 3],
                &[1.],
                &[1.],
                &CancellationToken::default(),
            )
            .is_err()
        );
        assert!(
            select(
                &image,
                &[true; 4],
                &[1.],
                &[f64::NAN],
                &CancellationToken::default(),
            )
            .is_err()
        );
        let cancelled = CancellationToken::default();
        cancelled.cancel();
        assert!(matches!(
            select(&image, &[true; 4], &[1.], &[1.], &cancelled),
            Err(Error::Cancelled)
        ));
    }
}
