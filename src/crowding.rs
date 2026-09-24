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
/// `widths` and `lengths` are per-source-contour measurements (source stroke
/// width and traced length). Ranking is purely geometric; see [`rank`].
/// Pixels adjacent to unsupported foreground (or the image edge) are
/// exterior contours and are always retained. Unsupported foreground includes
/// internal alpha holes and narrow transparent gaps, not only the canvas
/// exterior. Other owners are rejected only
/// when a higher-ranked owner remains parallel and within [`CROWD_RADIUS`] for a
/// substantial part of their target core.
///
/// The work is O(target pixels) with a fixed 5x5 neighborhood per core pixel;
/// it does not compare arbitrary pairs of contours or pixels.
pub(crate) fn select(
    strokes: &Strokes,
    foreground_support: &[bool],
    widths: &[f64],
    lengths: &[f64],
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
        || widths.len() != lengths.len()
        || widths.iter().any(|value| !value.is_finite() || *value < 0.)
        || lengths
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

    let isolation = isolation(&owner_pixels, w, h, &strokes.owners, strokes.core.as_raw());
    let continuity = continuity(&owner_pixels, w, h, &strokes.owners, strokes.core.as_raw());
    let ranks: Vec<_> = (0..widths.len())
        .map(|id| rank(lengths[id], isolation[id], widths[id], continuity[id]))
        .collect();

    // Empty owners have no core to crowd. Keeping them makes this a pure
    // selection decision; the existing renderer/canonicalizer owns their AA
    // handling and may already have cleared them.
    let mut order: Vec<_> = (0..widths.len()).collect();
    order.sort_by(|&a, &b| {
        outer[b]
            .cmp(&outer[a])
            .then_with(|| ranks[b].total_cmp(&ranks[a]))
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

/// Drawn core pixels an interior contour needs beyond the base minimum at
/// tiny targets, where only outlines and long interior lines should remain.
const TINY_INTERIOR_EXTRA: f64 = 10.;
/// The largest target side that receives the full extra length.
const TINY_TARGET: f64 = 128.;
/// The target side from which interior contours need no extra length.
const DETAILED_TARGET: f64 = 512.;

/// The drawn length an interior contour needs at a `w`×`h` target: 15
/// pixels at 128px or less, easing linearly to the base minimum at 512px.
pub(crate) fn interior_minimum(w: u32, h: u32) -> usize {
    let side = f64::from(w.max(h));
    let t = ((DETAILED_TARGET - side) / (DETAILED_TARGET - TINY_TARGET)).clamp(0., 1.);
    crate::contours::MIN_LINE_PIXELS + (TINY_INTERIOR_EXTRA * t).round() as usize
}

/// Reject kept interior contours drawn with fewer than `minimum` core
/// pixels. A contour is an outline when at least half of its core touches
/// the removed background or image edge; outlines are never rejected here.
/// A short interior contour whose core touches two different kept contours
/// is a connector and stays, so a long line is not interrupted.
pub(crate) fn drop_short_interior(
    strokes: &Strokes,
    foreground_support: &[bool],
    keep: &mut [bool],
    minimum: usize,
    cancel: &dyn Cancellation,
) -> Result<()> {
    cancel.check()?;
    let (w, h) = (
        strokes.core.width() as usize,
        strokes.core.height() as usize,
    );
    if foreground_support.len() != w * h || strokes.owners.len() != w * h {
        return Err(invalid_inputs());
    }
    let core = strokes.core.as_raw();
    let mut pixels = vec![Vec::new(); keep.len()];
    let mut touching = vec![0usize; keep.len()];
    for (i, owner) in strokes.owners.iter().enumerate() {
        if i.is_multiple_of(4096) {
            cancel.check()?;
        }
        if core[i] == 0 {
            continue;
        }
        let Some(owner) = *owner else {
            continue;
        };
        if owner >= keep.len() {
            return Err(invalid_inputs());
        }
        pixels[owner].push(i);
        touching[owner] += usize::from(touches_exterior(i, w, h, foreground_support));
    }
    let is_short: Vec<bool> = (0..keep.len())
        .map(|id| {
            keep[id]
                && !pixels[id].is_empty()
                && pixels[id].len() < minimum
                && touching[id] * 2 < pixels[id].len()
        })
        .collect();
    for id in 0..keep.len() {
        if !is_short[id] {
            continue;
        }
        let mut joined: Vec<usize> = Vec::new();
        for &i in &pixels[id] {
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (xx, yy) = (x + dx, y + dy);
                    if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                        continue;
                    }
                    let j = yy as usize * w + xx as usize;
                    if let Some(other) = strokes.owners[j]
                        && core[j] != 0
                        && other != id
                        && keep[other]
                        && !is_short[other]
                        && !joined.contains(&other)
                    {
                        joined.push(other);
                    }
                }
            }
        }
        if joined.len() < 2 {
            keep[id] = false;
        }
    }
    Ok(())
}

fn invalid_inputs() -> Error {
    Error::Scaling("Invalid contour crowding inputs".into())
}

/// Geometric priority among interior contours: longer first, then isolated
/// over lines packed among others, thicker source strokes over thinner, and
/// continuous over interrupted. Length is the base; each further property
/// scales it by a bounded factor (below 1.5, 1.3 and 1.15), so a contour more
/// than 2.25 times longer always wins however the others compare.
fn rank(length: f64, isolation: f64, width: f64, continuity: f64) -> f64 {
    length * (1. + 0.5 * isolation) * (1. + 0.3 * width / (width + 4.)) * (1. + 0.15 * continuity)
}

/// Share of each owner's core pixels with no other owner's core within
/// [`CROWD_RADIUS`]. A line in a tangle of contours scores near zero.
fn isolation(
    owner_pixels: &[Vec<usize>],
    w: usize,
    h: usize,
    owners: &[Option<usize>],
    core: &[u8],
) -> Vec<f64> {
    owner_pixels
        .iter()
        .enumerate()
        .map(|(owner, pixels)| {
            if pixels.is_empty() {
                return 0.;
            }
            let alone = pixels
                .iter()
                .filter(|&&i| {
                    let (x, y) = ((i % w) as isize, (i / w) as isize);
                    !(-CROWD_RADIUS..=CROWD_RADIUS).any(|dy| {
                        (-CROWD_RADIUS..=CROWD_RADIUS).any(|dx| {
                            let (xx, yy) = (x + dx, y + dy);
                            xx >= 0 && yy >= 0 && xx < w as isize && yy < h as isize && {
                                let j = yy as usize * w + xx as usize;
                                core[j] != 0 && owners[j].is_some_and(|other| other != owner)
                            }
                        })
                    })
                })
                .count();
            alone as f64 / pixels.len() as f64
        })
        .collect()
}

/// One over the number of 8-connected pieces of each owner's core: 1 for an
/// uninterrupted target line, less where crossings or overlap broke it.
fn continuity(
    owner_pixels: &[Vec<usize>],
    w: usize,
    h: usize,
    owners: &[Option<usize>],
    core: &[u8],
) -> Vec<f64> {
    let mut seen = vec![false; owners.len()];
    let mut stack = Vec::new();
    owner_pixels
        .iter()
        .enumerate()
        .map(|(owner, pixels)| {
            let mut pieces = 0usize;
            for &start in pixels {
                if seen[start] {
                    continue;
                }
                pieces += 1;
                seen[start] = true;
                stack.push(start);
                while let Some(i) = stack.pop() {
                    let (x, y) = ((i % w) as isize, (i / w) as isize);
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let (xx, yy) = (x + dx, y + dy);
                            if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                                continue;
                            }
                            let j = yy as usize * w + xx as usize;
                            if !seen[j] && core[j] != 0 && owners[j] == Some(owner) {
                                seen[j] = true;
                                stack.push(j);
                            }
                        }
                    }
                }
            }
            if pieces == 0 { 0. } else { 1. / pieces as f64 }
        })
        .collect()
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
            // Parallel within the crowd radius, or touching at any angle:
            // at a tiny target both merge into one unreadable blob.
            let touching = dx.abs() <= 1 && dy.abs() <= 1;
            if other != owner
                && keep[other]
                && core[j] != 0
                && (touching
                    || directions[j].is_some_and(|other_tangent| parallel(tangent, other_tangent)))
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
    fn touching_counts_at_any_angle_but_distant_crossings_need_parallels() {
        let horizontal = horizontal(4, 1, 9);
        let vertical: Vec<_> = (6..=8).map(|y| (5, y)).collect();
        let image = strokes(11, 10, &[(0, &horizontal), (1, &vertical)]);
        let directions: Vec<_> = (0..110)
            .map(|i| {
                image.owners[i]
                    .and_then(|owner| tangent(i, owner, 11, 10, &image.owners, image.core.as_raw()))
            })
            .collect();
        let test = |x: usize, y: usize| {
            let i = y * 11 + x;
            has_kept_parallel_neighbor(
                i,
                1,
                directions[i].unwrap(),
                11,
                10,
                image.core.as_raw(),
                &image.owners,
                &directions,
                &[true, false],
            )
        };
        // Perpendicular and two pixels away: a crossing, not crowding.
        assert!(!test(5, 6));
        let touching: Vec<_> = (5..=7).map(|y| (5, y)).collect();
        let image = strokes(11, 10, &[(0, &horizontal), (1, &touching)]);
        let directions: Vec<_> = (0..110)
            .map(|i| {
                image.owners[i]
                    .and_then(|owner| tangent(i, owner, 11, 10, &image.owners, image.core.as_raw()))
            })
            .collect();
        let i = 5 * 11 + 5;
        assert!(has_kept_parallel_neighbor(
            i,
            1,
            directions[i].unwrap(),
            11,
            10,
            image.core.as_raw(),
            &image.owners,
            &directions,
            &[true, false],
        ));
    }

    #[test]
    fn tiny_targets_keep_outlines_long_interiors_and_connectors_only() {
        assert_eq!(interior_minimum(128, 96), 15);
        assert_eq!(interior_minimum(64, 64), 15);
        assert_eq!(interior_minimum(320, 320), 10);
        assert_eq!(interior_minimum(512, 400), 5);
        // Row 0 of a 24x12 target is removed background. Owner 0 is an
        // outline along it; 1 is a long interior line; 2 a short interior
        // stub; 3 a short interior connector between 1 and a second long
        // interior line 4.
        let outline = horizontal(1, 1, 8);
        let long = horizontal(5, 1, 20);
        let stub = horizontal(9, 2, 6);
        let connector: Vec<_> = (6..=7).map(|y| (21, y)).collect();
        let lower = horizontal(8, 22, 23)
            .into_iter()
            .chain((1..=20).map(|x| (x, 11)))
            .collect::<Vec<_>>();
        let lower: Vec<_> = lower
            .into_iter()
            .filter(|&(x, _)| x != 22 && x != 23)
            .collect();
        let lower = [lower, vec![(22, 8), (22, 9), (22, 10), (21, 11)]].concat();
        let image = strokes(
            24,
            12,
            &[
                (0, &outline),
                (1, &long),
                (2, &stub),
                (3, &connector),
                (4, &lower[..lower.len() - 1]),
            ],
        );
        let mut support = vec![true; 24 * 12];
        support[..24].fill(false);
        let mut keep = vec![true; 5];
        drop_short_interior(
            &image,
            &support,
            &mut keep,
            15,
            &CancellationToken::default(),
        )
        .unwrap();
        assert_eq!(keep, [true, true, false, true, true]);
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
    fn length_outranks_thickness_and_thickness_breaks_equal_lengths() {
        let upper = horizontal(2, 1, 9);
        let lower = horizontal(4, 1, 9);
        let image = strokes(11, 8, &[(0, &upper), (1, &lower)]);
        let select = |widths: &[f64], lengths: &[f64]| {
            select(
                &image,
                &[true; 88],
                widths,
                lengths,
                &CancellationToken::default(),
            )
            .unwrap()
        };
        // A much longer source contour beats a thicker short one.
        assert_eq!(select(&[2., 7.], &[40., 10.]), [true, false]);
        assert_eq!(select(&[2., 7.], &[10., 40.]), [false, true]);
        // At equal length the thicker source stroke is kept.
        assert_eq!(select(&[2., 7.], &[20., 20.]), [false, true]);
        assert_eq!(select(&[7., 2.], &[20., 20.]), [true, false]);
    }

    #[test]
    fn isolated_and_continuous_lines_beat_tangled_or_broken_equals() {
        // Owner 1 runs beside owner 0 and also beside a third line, so it is
        // less isolated; owner 0 is also interrupted into two pieces.
        let broken: Vec<_> = (1..=12).filter(|&x| x != 6).map(|x| (x, 4)).collect();
        let tangled = horizontal(6, 1, 12);
        let neighbour = horizontal(8, 1, 12);
        let image = strokes(14, 11, &[(0, &broken), (1, &tangled), (2, &neighbour)]);
        let isolation = isolation(
            &[
                broken.iter().map(|&(x, y)| (y * 14 + x) as usize).collect(),
                tangled
                    .iter()
                    .map(|&(x, y)| (y * 14 + x) as usize)
                    .collect(),
                neighbour
                    .iter()
                    .map(|&(x, y)| (y * 14 + x) as usize)
                    .collect(),
            ],
            14,
            11,
            &image.owners,
            image.core.as_raw(),
        );
        assert!(isolation.iter().all(|&v| v == 0.), "all touch a neighbour");
        let pixels: Vec<Vec<usize>> = [&broken, &tangled]
            .iter()
            .map(|line| line.iter().map(|&(x, y)| (y * 14 + x) as usize).collect())
            .collect();
        let continuity = continuity(&pixels, 14, 11, &image.owners, image.core.as_raw());
        assert_eq!(continuity, [0.5, 1.]);
        assert!(rank(10., 1., 3., 1.) > rank(10., 0., 3., 1.));
        assert!(rank(10., 0., 3., 1.) > rank(10., 0., 3., 0.5));
        assert!(rank(10., 0., 6., 0.5) > rank(10., 0., 3., 0.5));
        // Bounded factors: 2.25 times the length beats any better shorter line.
        assert!(rank(22.5, 0., 0., 0.) > rank(10., 1., 1e9, 1.));
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
