//! Game Asset reduction: direction-merged contours, source-width opacity,
//! tight antialiasing, and Lanczos3 source fill with bounded halo tone-down.

#![doc = include_str!("../README.md")]

mod api;
pub use api::{Cancellation, CancellationToken, Error, GameAssetAa, Result};
use image::{GrayImage, RgbaImage};
use std::sync::{Arc, Mutex};
mod antialias;
#[cfg(test)]
mod benchmarks;
mod cleanup;
mod color;
mod contours;
mod coverage;
mod crowding;
mod deink;
mod detect;
mod field;
mod foreground_halo;
mod halo;
mod ink;
mod lanczos;
mod opacity;
mod paint;
mod raster;
mod silhouette;
mod smoothing;
mod source;
mod strokes;
mod target_cleanup;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod vector_review;
pub const DEFAULT_MEMORY_LIMIT: u64 = 4 * 1024 * 1024 * 1024;
// The source phase keeps the established 512 B/pixel allowance for analysis,
// contour storage and source resampling. Target data is not concurrent
// with analysis, but can contain two linear images, contour ownership/colors,
// support/opacity projections, the output and cache; 160 B/pixel conservatively
// accounts for that composition peak. This phase-aware estimate is capped at
// four GiB, independently of the decoder and canvas safety limits.
const SOURCE_PHASE_BYTES: u64 = 512;
const TARGET_PHASE_BYTES: u64 = 160;
const FOREGROUND_SOURCE_BYTES: u64 = 96;

/// Colour to use when painting contours over an externally prepared fill.
///
/// `OriginalInk` samples the ink from the unedited source.  `DarkenedFill`
/// samples the already-downscaled fill at the contour pixel and multiplies its
/// linear-light RGB channels by `luminance`.  This deliberately leaves alpha
/// to the normal contour compositor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OutlineColor {
    OriginalInk,
    DarkenedFill { luminance: f64 },
}

/// Controls how ordinary resize calls treat a flat, opaque source canvas.
///
/// The default keeps the established behavior: boundary-connected pixels that
/// match a flat opaque canvas become transparent.  Select
/// [`Self::preserve_opaque_background`] when the source is a complete opaque
/// image, such as a game backdrop.  Source alpha is always respected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResizeOptions {
    remove_opaque_background: bool,
}

impl ResizeOptions {
    /// Preserve a flat opaque source canvas during resize.
    #[must_use]
    pub const fn preserve_opaque_background() -> Self {
        Self {
            remove_opaque_background: false,
        }
    }
}

impl Default for ResizeOptions {
    fn default() -> Self {
        Self {
            remove_opaque_background: true,
        }
    }
}

fn working_set_estimate(sw: u32, sh: u32, w: u32, h: u32) -> u64 {
    u64::from(sw)
        .saturating_mul(u64::from(sh))
        .saturating_mul(SOURCE_PHASE_BYTES)
        .saturating_add(
            u64::from(w)
                .saturating_mul(u64::from(h))
                .saturating_mul(TARGET_PHASE_BYTES),
        )
}

fn check_working_set_budget(sw: u32, sh: u32, w: u32, h: u32, limit: u64) -> Result<()> {
    if working_set_estimate(sw, sh, w, h) > limit {
        return Err(Error::GameAssetMemoryLimit { limit_bytes: limit });
    }
    Ok(())
}

fn check_foreground_working_set_budget(sw: u32, sh: u32, w: u32, h: u32, limit: u64) -> Result<()> {
    let estimate = working_set_estimate(sw, sh, w, h).saturating_add(
        u64::from(sw)
            .saturating_mul(u64::from(sh))
            .saturating_mul(FOREGROUND_SOURCE_BYTES),
    );
    if estimate > limit {
        return Err(Error::GameAssetMemoryLimit { limit_bytes: limit });
    }
    Ok(())
}

fn binary_contour_image(mask: &raster::Mask) -> Result<GrayImage> {
    let Some(image) = GrayImage::from_vec(
        mask.w as u32,
        mask.h as u32,
        mask.data
            .iter()
            .map(|&contour| if contour { 0 } else { u8::MAX })
            .collect(),
    ) else {
        return Err(Error::Scaling("Invalid contour mask dimensions".into()));
    };
    Ok(image)
}

struct Prepared {
    models: Vec<detect::Model>,
    widths: Vec<f64>,
    contours: contours::Contours,
    mask: raster::Mask,
    linear: color::LinearImage,
    silhouette: Option<silhouette::Silhouette>,
    original_is_opaque: bool,
}
struct TargetContours {
    strokes: strokes::Strokes,
    colors: Vec<[f64; 3]>,
    geometry: TargetGeometry,
    suppress_unsupported: bool,
    brightness: Option<f64>,
}
struct TargetGeometry {
    curves: Vec<coverage::Quadratic>,
    owners: Vec<usize>,
    donors: Vec<detect::Model>,
    donor_owners: Vec<usize>,
    fitted_source: Option<FittedSource>,
    scale: [f64; 2],
}
#[derive(Clone)]
struct FittedSource {
    curves: Vec<coverage::Quadratic>,
    owners: Vec<usize>,
    trace_donors: Vec<([f64; 2], usize)>,
}
#[derive(Clone, Copy)]
struct InkSource<'a> {
    linear: &'a color::LinearImage,
    suppress_unsupported: bool,
    brightness: Option<f64>,
}
#[derive(Clone, Copy)]
struct TargetRender<'source, 'cancel> {
    target: (u32, u32),
    aa: GameAssetAa,
    ink_source: InkSource<'source>,
    cancel: &'cancel dyn Cancellation,
}
struct ForegroundInkResize {
    w: u32,
    h: u32,
    aa: GameAssetAa,
    brightness: f64,
}
struct FinishedFill {
    linear: color::LinearImage,
    foreground_support: Option<Vec<bool>>,
    /// The silhouette-isolated fill source, when a silhouette removed a
    /// canvas. Ink removal must not borrow that canvas as fill colour.
    isolated: Option<color::LinearImage>,
}
struct FillTarget {
    target: (u32, u32),
    aa: GameAssetAa,
    foreground_support: bool,
}
struct FillResize {
    w: u32,
    h: u32,
    aa: GameAssetAa,
    outline_color: OutlineColor,
}
impl Prepared {
    fn new(image: &RgbaImage, cancel: &dyn Cancellation) -> Result<Self> {
        Self::with_options(image, ResizeOptions::default(), cancel)
    }

    fn with_options(
        image: &RgbaImage,
        options: ResizeOptions,
        cancel: &dyn Cancellation,
    ) -> Result<Self> {
        cancel.check()?;
        let samples = detect::detect(image, cancel)?;
        cancel.check()?;
        let models = detect::fit_models(&samples, 0.012, cancel)?;
        let thinned = source::skeleton(
            &models,
            image.width() as usize,
            image.height() as usize,
            cancel,
        )?;
        cancel.check()?;
        let contours = contours::Contours::new(&thinned, &models, cancel)?;
        let mask = ink::ink_mask(image, &samples, &thinned, cancel)?;
        cancel.check()?;
        let widths = opacity::widths(image, &mask, &samples, &contours, cancel)?;
        let linear = color::LinearImage::from_rgba(image);
        let mut original_is_opaque = true;
        for (i, pixel) in image.pixels().enumerate() {
            if i.is_multiple_of(4096) {
                cancel.check()?;
            }
            original_is_opaque &= pixel[3] == 255;
        }
        let silhouette = silhouette::Silhouette::detect_with_opaque_background_removal(
            image,
            options.remove_opaque_background,
            cancel,
        )?;
        Ok(Self {
            models,
            widths,
            contours,
            mask,
            linear,
            silhouette,
            original_is_opaque,
        })
    }
    fn target_contours(
        &self,
        image: &RgbaImage,
        w: u32,
        h: u32,
        aa: GameAssetAa,
        cancel: &dyn Cancellation,
    ) -> Result<TargetContours> {
        self.target_contours_with_ink_source(
            image,
            w,
            h,
            aa,
            InkSource {
                linear: &self.linear,
                suppress_unsupported: false,
                brightness: None,
            },
            cancel,
        )
    }

    fn polished_contour_mask(
        &self,
        source: &RgbaImage,
        w: u32,
        h: u32,
        cancel: &dyn Cancellation,
    ) -> Result<GrayImage> {
        let scale = [
            w as f64 / source.width() as f64,
            h as f64 / source.height() as f64,
        ];
        let fitted = self
            .contours
            .polished(scale, contours::MAX_SHORT_PIXELS, cancel)?;
        let curves: Vec<coverage::Quadratic> = fitted
            .curves
            .iter()
            .map(|curve| {
                curve.map(|p| [(p[0] + 0.5) * scale[0] - 0.5, (p[1] + 0.5) * scale[1] - 0.5])
            })
            .collect();
        let strokes = strokes::render(
            &curves,
            &fitted.owners,
            &self.widths,
            w,
            h,
            GameAssetAa::new(0),
            cancel,
        )?;
        GrayImage::from_raw(w, h, strokes.core.into_raw())
            .ok_or_else(|| Error::Scaling("Invalid polished contour dimensions".into()))
    }

    /// Keep original contour geometry while optionally coloring it from an
    /// aligned foreground. This remains private because ordinary resize calls
    /// must retain their established original-ink behavior.
    fn target_contours_with_ink_source(
        &self,
        image: &RgbaImage,
        w: u32,
        h: u32,
        aa: GameAssetAa,
        ink_source: InkSource<'_>,
        cancel: &dyn Cancellation,
    ) -> Result<TargetContours> {
        let suppress_unsupported = ink_source.suppress_unsupported;
        let brightness = ink_source.brightness;
        let scale = [
            w as f64 / image.width() as f64,
            h as f64 / image.height() as f64,
        ];
        // Every path drops lines shorter than the minimum target length;
        // short connectors between longer lines survive selection, so a line
        // split at junctions stays continuous.
        let cutoff = contours::MAX_SHORT_PIXELS;
        let (retained, owners) = self.contours.retain_with_ids(&self.models, scale, cutoff);
        let use_splines = scale[0] < 1. || scale[1] < 1.;
        let fitted = use_splines
            .then(|| self.contours.polished(scale, cutoff, cancel))
            .transpose()?;
        let (curves, curve_owners): (Vec<coverage::Quadratic>, Vec<usize>) = if let Some(fitted) =
            &fitted
        {
            // A path with no original local model has no established colour
            // donor. Preserve the existing rule by not redrawing it in the
            // colour pipeline; diagnostics can still display its geometry.
            let supported: std::collections::HashSet<_> = owners.iter().copied().collect();
            fitted
                .curves
                .iter()
                .zip(&fitted.owners)
                .filter(|(_, owner)| supported.contains(owner))
                .map(|(curve, &owner)| {
                    (
                        curve.map(|p| {
                            [(p[0] + 0.5) * scale[0] - 0.5, (p[1] + 0.5) * scale[1] - 0.5]
                        }),
                        owner,
                    )
                })
                .unzip::<_, _, Vec<coverage::Quadratic>, Vec<usize>>()
        } else {
            let smoothed = &retained;
            (
                smoothed
                    .iter()
                    .map(|m| {
                        detect::controls(m, 1.)
                            .map(|p| [(p[0] + 0.5) * scale[0] - 0.5, (p[1] + 0.5) * scale[1] - 0.5])
                    })
                    .collect::<Vec<coverage::Quadratic>>(),
                owners.clone(),
            )
        };
        let geometry = TargetGeometry {
            curves,
            owners: curve_owners,
            donors: retained,
            donor_owners: owners,
            fitted_source: fitted.map(|fit| FittedSource {
                curves: fit.curves,
                owners: fit.owners,
                trace_donors: fit.trace_donors,
            }),
            scale,
        };
        let keep = vec![true; self.widths.len()];
        let (strokes, colors) = self.render_target_geometry(
            &geometry,
            &keep,
            TargetRender {
                target: (w, h),
                aa,
                ink_source,
                cancel,
            },
        )?;
        Ok(TargetContours {
            strokes,
            colors,
            geometry,
            suppress_unsupported,
            brightness,
        })
    }

    fn render_target_geometry(
        &self,
        geometry: &TargetGeometry,
        keep: &[bool],
        render: TargetRender<'_, '_>,
    ) -> Result<(strokes::Strokes, Vec<[f64; 3]>)> {
        let TargetRender {
            target: (w, h),
            aa,
            ink_source,
            cancel,
        } = render;
        if keep.len() != self.widths.len() {
            return Err(Error::Scaling("Invalid retained contour owners".into()));
        }
        let (curves, owners): (Vec<_>, Vec<_>) = geometry
            .curves
            .iter()
            .zip(&geometry.owners)
            .filter(|&(_, &owner)| keep[owner])
            .map(|(&curve, &owner)| (curve, owner))
            .unzip();
        let mut strokes = strokes::render(&curves, &owners, &self.widths, w, h, aa, cancel)?;
        let (mut colors, visible) = if let Some(FittedSource {
            curves: source_curves,
            owners: source_owners,
            trace_donors,
        }) = &geometry.fitted_source
        {
            let (source_curves, source_owners): (Vec<_>, Vec<_>) = source_curves
                .iter()
                .zip(source_owners)
                .filter(|&(_, &owner)| keep[owner])
                .map(|(&curve, &owner)| (curve, owner))
                .unzip();
            paint::ink_colors_for_curves(
                ink_source.linear,
                &source_curves,
                &source_owners,
                &geometry.donors,
                &geometry.donor_owners,
                trace_donors,
                &strokes,
                geometry.scale,
                cancel,
            )?
        } else {
            let (donors, owners): (Vec<_>, Vec<_>) = geometry
                .donors
                .iter()
                .zip(&geometry.donor_owners)
                .filter(|&(_, &owner)| keep[owner])
                .map(|(&model, &owner)| (model, owner))
                .unzip();
            paint::ink_colors_with_visibility(
                ink_source.linear,
                &donors,
                &donors,
                &owners,
                &strokes,
                geometry.scale,
                cancel,
            )?
        };
        if !ink_source.suppress_unsupported {
            if let Some((i, _)) = strokes
                .coverage
                .as_raw()
                .iter()
                .zip(&visible)
                .enumerate()
                .find(|&(_, (&coverage, &visible))| coverage != 0 && !visible)
            {
                return Err(Error::Scaling(format!(
                    "No visible source ink donor for target pixel {i}"
                )));
            }
        } else {
            for (i, &supported) in visible.iter().enumerate() {
                if i.is_multiple_of(4096) {
                    cancel.check()?;
                }
                if !supported {
                    strokes.coverage.as_mut()[i] = 0;
                    strokes.core.as_mut()[i] = 0;
                    strokes.owners[i] = None;
                }
            }
        }
        if let Some(brightness) = ink_source.brightness {
            for (i, color) in colors.iter_mut().enumerate() {
                if i.is_multiple_of(4096) {
                    cancel.check()?;
                }
                *color = color::scale_srgb(*color, brightness);
            }
        }
        Ok((strokes, colors))
    }

    /// Apply whole-contour crowding decisions by rendering the surviving
    /// geometry again. Pixel-level arbitration happens inside `strokes::render`,
    /// so this restores a retained contour wherever a rejected owner had won
    /// their initial overlap. Colors and visibility are recomputed from the
    /// same-owner source donors for that new ownership map.
    fn rerender_crowded_target(
        &self,
        contours: &mut TargetContours,
        foreground_support: &[bool],
        lengths: &[f64],
        render: TargetRender<'_, '_>,
    ) -> Result<bool> {
        let mut keep = crowding::select(
            &contours.strokes,
            foreground_support,
            &self.widths,
            lengths,
            render.cancel,
        )?;
        // Tiny targets keep only outlines and long interior lines.
        crowding::drop_short_interior(
            &contours.strokes,
            foreground_support,
            &mut keep,
            crowding::interior_minimum(render.target.0, render.target.1),
            render.cancel,
        )?;
        let rejected = keep.iter().any(|&keep| !keep);
        if rejected {
            let (strokes, colors) =
                self.render_target_geometry(&contours.geometry, &keep, render)?;
            contours.strokes = strokes;
            contours.colors = colors;
        }
        Ok(rejected)
    }

    /// Return the binary target core used by the baseline foreground-ink
    /// compositor before opacity, antialias coverage, colour, and halo work.
    fn foreground_contour_mask(
        &self,
        original: &RgbaImage,
        foreground: &RgbaImage,
        w: u32,
        h: u32,
        aa: GameAssetAa,
        cancel: &dyn Cancellation,
    ) -> Result<GrayImage> {
        let fill_linear = color::LinearImage::from_rgba(foreground);
        let fill_silhouette = silhouette::Silhouette::detect(foreground, cancel)?;
        let mut contours = self.target_contours_with_ink_source(
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
        )?;
        let fill = self.fill_for_target(
            &fill_linear,
            fill_silhouette.as_ref(),
            &contours.strokes,
            FillTarget {
                target: (w, h),
                aa,
                foreground_support: true,
            },
            cancel,
        )?;
        let support = fill
            .foreground_support
            .as_deref()
            .expect("foreground support was requested");
        let ink_source = InkSource {
            linear: &fill_linear,
            suppress_unsupported: contours.suppress_unsupported,
            brightness: contours.brightness,
        };
        self.rerender_crowded_target(
            &mut contours,
            support,
            &self.contours.lengths,
            TargetRender {
                target: (w, h),
                aa,
                ink_source,
                cancel,
            },
        )?;
        let raw = raster::Mask {
            w: contours.strokes.core.width() as usize,
            h: contours.strokes.core.height() as usize,
            data: contours
                .strokes
                .core
                .as_raw()
                .iter()
                .map(|&value| value != 0)
                .collect(),
        };
        let canonical = target_cleanup::thin_outer(&raw, support, cancel)?;
        binary_contour_image(&canonical)
    }

    fn resize_with_fill(
        &self,
        contour_source: &RgbaImage,
        fill_linear: &color::LinearImage,
        fill_silhouette: Option<&silhouette::Silhouette>,
        target: (u32, u32),
        aa: GameAssetAa,
        cancel: &dyn Cancellation,
    ) -> Result<RgbaImage> {
        let (w, h) = target;
        let contours = self.target_contours(contour_source, w, h, aa, cancel)?;
        self.resize_with_fill_from_target(
            fill_linear,
            fill_silhouette,
            contours,
            target,
            aa,
            cancel,
        )
    }

    /// Shared fill, silhouette, halo and compositor path after contour colors
    /// have been selected.
    fn resize_with_fill_from_target(
        &self,
        fill_linear: &color::LinearImage,
        fill_silhouette: Option<&silhouette::Silhouette>,
        TargetContours {
            strokes, colors, ..
        }: TargetContours,
        target: (u32, u32),
        aa: GameAssetAa,
        cancel: &dyn Cancellation,
    ) -> Result<RgbaImage> {
        let strength = opacity::calculate(&self.widths, &strokes.core, &strokes.owners);
        let paint = opacity::apply(&strokes.coverage, &strokes.owners, &strength);
        let fill = self.fill_for_target(
            fill_linear,
            fill_silhouette,
            &strokes,
            FillTarget {
                target,
                aa,
                foreground_support: false,
            },
            cancel,
        )?;
        let support: Vec<bool> = fill.linear.pixels.iter().map(|p| p[3] >= 0.5).collect();
        let fill = deink::apply(
            fill.isolated.as_ref().unwrap_or(fill_linear),
            &self.mask,
            fill.linear,
            &strokes.core,
            &support,
            cancel,
        )?;
        let result = paint::composite(&fill, &colors, &paint);
        cancel.check()?;
        Ok(result)
    }

    /// Produce the finished fill once, including ordinary halo and silhouette
    /// work. The foreground compositor uses this support to canonicalize its
    /// target contour core before painting.
    fn fill_for_target(
        &self,
        fill_linear: &color::LinearImage,
        fill_silhouette: Option<&silhouette::Silhouette>,
        strokes: &strokes::Strokes,
        request: FillTarget,
        cancel: &dyn Cancellation,
    ) -> Result<FinishedFill> {
        let (w, h) = request.target;
        let scale = [
            w as f64 / self.linear.w as f64,
            h as f64 / self.linear.h as f64,
        ];
        let mut retained_mask = self.contours.retained_ink_mask(&self.mask, scale, cancel)?;
        cancel.check()?;
        let isolated = if let Some(silhouette) = fill_silhouette {
            for (i, (masked, &supported)) in retained_mask
                .data
                .iter_mut()
                .zip(&silhouette.support.data)
                .enumerate()
            {
                if i % 4096 == 0 {
                    cancel.check()?;
                }
                *masked &= supported;
            }
            Some(silhouette.isolated(fill_linear, cancel)?)
        } else {
            None
        };
        let fill_source = isolated.as_ref().unwrap_or(fill_linear);
        let base = lanczos::resize(fill_source, w as usize, h as usize, cancel)?;
        let target_support = if request.foreground_support {
            let mut support = Vec::with_capacity(base.pixels.len());
            for (i, pixel) in base.pixels.iter().enumerate() {
                if i.is_multiple_of(4096) {
                    cancel.check()?;
                }
                support.push(pixel[3] >= 0.5);
            }
            Some(support)
        } else {
            None
        };
        let base = halo::apply(fill_source, &retained_mask, base, &strokes.core, cancel)?;
        let fill = if let Some(silhouette) = fill_silhouette {
            let source_coverage = silhouette.coverage(w as usize, h as usize, cancel)?;
            let coverage = silhouette.target_coverage(&source_coverage, request.aa, cancel)?;
            let opacity = silhouette.intrinsic_opacity(
                fill_source,
                &source_coverage,
                w as usize,
                h as usize,
                cancel,
            )?;
            let mut pixels = Vec::with_capacity(base.pixels.len());
            for (i, ((pixel, support), intrinsic)) in
                base.pixels.iter().zip(coverage).zip(opacity).enumerate()
            {
                if i % 4096 == 0 {
                    cancel.check()?;
                }
                let alpha = (intrinsic * support).clamp(0., 1.);
                pixels.push(if alpha <= 1e-8 {
                    [0.; 4]
                } else {
                    [pixel[0], pixel[1], pixel[2], alpha]
                });
            }
            color::LinearImage {
                w: w as usize,
                h: h as usize,
                pixels,
            }
        } else {
            base
        };
        Ok(FinishedFill {
            linear: fill,
            foreground_support: target_support,
            isolated,
        })
    }

    fn resize(
        &self,
        image: &RgbaImage,
        w: u32,
        h: u32,
        aa: GameAssetAa,
        cancel: &dyn Cancellation,
    ) -> Result<RgbaImage> {
        self.resize_with_fill(
            image,
            &self.linear,
            self.silhouette.as_ref(),
            (w, h),
            aa,
            cancel,
        )
    }

    fn resize_with_foreground(
        &self,
        original: &RgbaImage,
        foreground: &RgbaImage,
        w: u32,
        h: u32,
        aa: GameAssetAa,
        cancel: &dyn Cancellation,
    ) -> Result<RgbaImage> {
        let fill_linear = color::LinearImage::from_rgba(foreground);
        let fill_silhouette = silhouette::Silhouette::detect(foreground, cancel)?;
        self.resize_with_fill(
            original,
            &fill_linear,
            fill_silhouette.as_ref(),
            (w, h),
            aa,
            cancel,
        )
    }

    fn resize_with_foreground_ink(
        &self,
        original: &RgbaImage,
        foreground: &RgbaImage,
        request: ForegroundInkResize,
        cancel: &dyn Cancellation,
    ) -> Result<RgbaImage> {
        let fill_linear = color::LinearImage::from_rgba(foreground);
        let fill_silhouette = silhouette::Silhouette::detect(foreground, cancel)?;
        let contours = self.target_contours_with_ink_source(
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
        )?;
        self.resize_with_fill_from_target_and_canonical_core(
            &fill_linear,
            fill_silhouette.as_ref(),
            contours,
            (request.w, request.h),
            request.aa,
            cancel,
        )
    }

    /// The foreground-ink-only compositor path reduces all rendered target
    /// cores to one canonical, outer-prioritized mask before painting them.
    fn resize_with_fill_from_target_and_canonical_core(
        &self,
        fill_linear: &color::LinearImage,
        fill_silhouette: Option<&silhouette::Silhouette>,
        mut contours: TargetContours,
        target: (u32, u32),
        aa: GameAssetAa,
        cancel: &dyn Cancellation,
    ) -> Result<RgbaImage> {
        let fill = self.fill_for_target(
            fill_linear,
            fill_silhouette,
            &contours.strokes,
            FillTarget {
                target,
                aa,
                foreground_support: true,
            },
            cancel,
        )?;
        let support = fill
            .foreground_support
            .as_deref()
            .expect("foreground support was requested");
        let ink_source = InkSource {
            linear: fill_linear,
            suppress_unsupported: contours.suppress_unsupported,
            brightness: contours.brightness,
        };
        let rejected = self.rerender_crowded_target(
            &mut contours,
            support,
            &self.contours.lengths,
            TargetRender {
                target,
                aa,
                ink_source,
                cancel,
            },
        )?;
        // Halo repair sees contour cores. Rebuild only when arbitration
        // removed an owner, so rejected interiors rely on scaled fill alone.
        let fill = if rejected {
            self.fill_for_target(
                fill_linear,
                fill_silhouette,
                &contours.strokes,
                FillTarget {
                    target,
                    aa,
                    foreground_support: true,
                },
                cancel,
            )?
        } else {
            fill
        };
        let support = fill
            .foreground_support
            .as_deref()
            .expect("foreground support was requested");
        Self::canonicalize_foreground_strokes(
            &mut contours.strokes,
            &mut contours.colors,
            support,
            aa,
            cancel,
        )?;
        let strength = self.foreground_ink_strengths(&contours.strokes, aa);
        let paint = opacity::apply(
            &contours.strokes.coverage,
            &contours.strokes.owners,
            &strength,
        );
        // The canonical outer core is the only outline: its one-pixel
        // neighbourhood loses the resampled outline ink.
        let fill_linear_clean = deink::apply(
            fill.isolated.as_ref().unwrap_or(fill_linear),
            &self.mask,
            fill.linear.clone(),
            &contours.strokes.core,
            support,
            cancel,
        )?;
        let baseline = paint::composite(&fill_linear_clean, &contours.colors, &paint);
        foreground_halo::clean(
            foreground_halo::Inputs {
                support,
                original_is_opaque: self.original_is_opaque,
                fill: &fill.linear,
                baseline,
                core: &contours.strokes.core,
                coverage: &contours.strokes.coverage,
                owners: &contours.strokes.owners,
                colors: &contours.colors,
            },
            cancel,
        )
    }

    /// Convert all target contour owners into one exterior-prioritized core.
    /// Removed cores cannot retain full AA coverage; a bounded fringe may only
    /// borrow the nearest surviving canonical core's owner and colour.
    fn canonicalize_foreground_strokes(
        strokes: &mut strokes::Strokes,
        colors: &mut [[f64; 3]],
        support: &[bool],
        aa: GameAssetAa,
        cancel: &dyn Cancellation,
    ) -> Result<()> {
        let raw = raster::Mask {
            w: strokes.core.width() as usize,
            h: strokes.core.height() as usize,
            data: strokes
                .core
                .as_raw()
                .iter()
                .map(|&value| value != 0)
                .collect(),
        };
        if colors.len() != raw.data.len() || strokes.owners.len() != raw.data.len() {
            return Err(Error::Scaling(
                "Invalid foreground contour ownership".into(),
            ));
        }
        let canonical = target_cleanup::thin_outer(&raw, support, cancel)?;
        let previous_coverage = strokes.coverage.as_raw().to_vec();
        let previous_owners = strokes.owners.clone();
        for (i, &keep) in canonical.data.iter().enumerate() {
            if i.is_multiple_of(4096) {
                cancel.check()?;
            }
            if raw.data[i] && !keep {
                strokes.core.as_mut()[i] = 0;
            }
        }
        for i in 0..raw.data.len() {
            if i.is_multiple_of(4096) {
                cancel.check()?;
            }
            if !canonical.data[i] {
                strokes.coverage.as_mut()[i] = 0;
                strokes.owners[i] = None;
            }
        }
        if aa.percent() == 0 {
            return Ok(());
        }
        let fringe_cap = (85 * u16::from(aa.percent()) / 100) as u8;
        for (i, &old_coverage) in previous_coverage.iter().enumerate() {
            if i.is_multiple_of(4096) {
                cancel.check()?;
            }
            if canonical.data[i] || old_coverage == 0 {
                continue;
            }
            let x = i % raw.w;
            let y = i / raw.w;
            let donor = (-1isize..=1)
                .flat_map(|dy| (-1isize..=1).map(move |dx| (dx, dy)))
                .filter(|&(dx, dy)| dx != 0 || dy != 0)
                .filter_map(|(dx, dy)| {
                    let xx = x as isize + dx;
                    let yy = y as isize + dy;
                    (xx >= 0 && yy >= 0 && xx < raw.w as isize && yy < raw.h as isize)
                        .then(|| yy as usize * raw.w + xx as usize)
                })
                .filter(|&j| canonical.data[j] && previous_owners[j].is_some())
                .min_by_key(|&j| {
                    let dx = j % raw.w;
                    let dy = j / raw.w;
                    let ddx = dx.abs_diff(x);
                    let ddy = dy.abs_diff(y);
                    (ddx * ddx + ddy * ddy, j)
                });
            if let Some(j) = donor {
                strokes.coverage.as_mut()[i] = old_coverage.min(fringe_cap);
                strokes.owners[i] = previous_owners[j];
                colors[i] = colors[j];
            }
        }
        Ok(())
    }

    /// Foreground ink deliberately hardens its own core as antialiasing is
    /// reduced. The intrinsic contour strength remains the AA100 endpoint.
    fn foreground_ink_strengths(&self, strokes: &strokes::Strokes, aa: GameAssetAa) -> Vec<f64> {
        let intrinsic = opacity::calculate(&self.widths, &strokes.core, &strokes.owners);
        if aa.percent() == 100 {
            return intrinsic;
        }
        let fraction = f64::from(aa.percent()) / 100.;
        intrinsic
            .into_iter()
            .map(|strength| 1. - fraction * (1. - strength))
            .collect()
    }

    fn resize_over_fill(
        &self,
        original: &RgbaImage,
        fill_source: &RgbaImage,
        request: FillResize,
        cancel: &dyn Cancellation,
    ) -> Result<RgbaImage> {
        let TargetContours {
            strokes, colors, ..
        } = self.target_contours(original, request.w, request.h, request.aa, cancel)?;
        let strength = opacity::calculate(&self.widths, &strokes.core, &strokes.owners);
        let paint = opacity::apply(&strokes.coverage, &strokes.owners, &strength);
        let fill_source = color::LinearImage::from_rgba(fill_source);
        let base = lanczos::resize(&fill_source, request.w as usize, request.h as usize, cancel)?;
        let fill = if let Some(silhouette) = &self.silhouette {
            let source_coverage =
                silhouette.coverage(request.w as usize, request.h as usize, cancel)?;
            let coverage = silhouette.target_coverage(&source_coverage, request.aa, cancel)?;
            let mut pixels = Vec::with_capacity(base.pixels.len());
            for (i, (pixel, &support)) in base.pixels.iter().zip(&coverage).enumerate() {
                if i % 4096 == 0 {
                    cancel.check()?;
                }
                pixels.push([pixel[0], pixel[1], pixel[2], pixel[3] * support]);
            }
            color::LinearImage {
                w: request.w as usize,
                h: request.h as usize,
                pixels,
            }
        } else {
            base
        };
        let colors = match request.outline_color {
            OutlineColor::OriginalInk => colors,
            OutlineColor::DarkenedFill { luminance } => {
                let luminance = if luminance.is_finite() {
                    luminance.clamp(0., 1.)
                } else {
                    0.
                };
                fill.pixels
                    .iter()
                    .map(|p| [p[0] * luminance, p[1] * luminance, p[2] * luminance])
                    .collect()
            }
        };
        cancel.check()?;
        Ok(paint::composite(&fill, &colors, &paint))
    }
}
#[derive(Default)]
struct Cache {
    prepared: Option<Arc<Prepared>>,
    target: Option<((u32, u32, GameAssetAa), Arc<RgbaImage>)>,
}
/// One source analysis and one target result per preview session. Heavy work
/// stays outside the lock so an obsolete preview can be cancelled promptly.
pub struct Session {
    source: Arc<RgbaImage>,
    cache: Mutex<Cache>,
    memory_limit: u64,
    options: ResizeOptions,
}
impl Session {
    pub fn new(source: Arc<RgbaImage>) -> Self {
        Self::with_options(source, ResizeOptions::default())
    }

    /// Create a cached session with the selected resize behavior.
    #[must_use]
    pub fn with_options(source: Arc<RgbaImage>, options: ResizeOptions) -> Self {
        Self::with_memory_limit_and_options(source, DEFAULT_MEMORY_LIMIT, options)
    }

    /// Create a cached session with a caller-selected working-memory limit.
    pub fn with_memory_limit(source: Arc<RgbaImage>, memory_limit: u64) -> Self {
        Self::with_memory_limit_and_options(source, memory_limit, ResizeOptions::default())
    }

    /// Create a cached session with a caller-selected memory limit and resize
    /// behavior.
    #[must_use]
    pub fn with_memory_limit_and_options(
        source: Arc<RgbaImage>,
        memory_limit: u64,
        options: ResizeOptions,
    ) -> Self {
        Self {
            source,
            memory_limit,
            options,
            cache: Mutex::new(Cache::default()),
        }
    }

    fn prepared(&self, cancel: &dyn Cancellation) -> Result<Arc<Prepared>> {
        if let Some(prepared) = self
            .cache
            .lock()
            .expect("Game Asset cache poisoned")
            .prepared
            .clone()
        {
            return Ok(prepared);
        }
        let prepared = Arc::new(Prepared::with_options(&self.source, self.options, cancel)?);
        cancel.check()?;
        self.cache
            .lock()
            .expect("Game Asset cache poisoned")
            .prepared = Some(prepared.clone());
        Ok(prepared)
    }

    /// Analyze the original artwork before an external background remover
    /// changes its alpha or colours.
    pub fn prepare(&self, cancel: &dyn Cancellation) -> Result<()> {
        cancel.check()?;
        let (sw, sh) = self.source.dimensions();
        if sw == 0 || sh == 0 {
            return Err(Error::InvalidDimensions);
        }
        check_working_set_budget(sw, sh, 1, 1, self.memory_limit)?;
        self.prepared(cancel)?;
        cancel.check()
    }

    /// Return the source-resolution detected contour mask.
    ///
    /// Pixels are binary: `0` is a detected contour and `255` is background.
    /// This reads the prepared source analysis only; it does not require a
    /// removed foreground or render a scaled asset.
    pub fn source_contour_mask(&self, cancel: &dyn Cancellation) -> Result<GrayImage> {
        cancel.check()?;
        let (sw, sh) = self.source.dimensions();
        if sw == 0 || sh == 0 {
            return Err(Error::InvalidDimensions);
        }
        check_working_set_budget(sw, sh, 1, 1, self.memory_limit)?;
        let prepared = self.prepared(cancel)?;
        cancel.check()?;
        binary_contour_image(&prepared.mask)
    }

    /// Return a binary rendering of the ordered traces fitted as smooth
    /// cubics. Unlike [`Self::source_contour_mask`], this is a visual
    /// diagnostic and may move a contour fractionally within its source
    /// support. It is available at source size for editor inspection.
    pub fn polished_contour_mask(
        &self,
        w: u32,
        h: u32,
        cancel: &dyn Cancellation,
    ) -> Result<GrayImage> {
        cancel.check()?;
        let (sw, sh) = self.source.dimensions();
        if w == 0 || h == 0 || sw == 0 || sh == 0 || w > sw || h > sh {
            return Err(Error::InvalidDimensions);
        }
        check_working_set_budget(sw, sh, w, h, self.memory_limit)?;
        let prepared = self.prepared(cancel)?;
        let core = prepared.polished_contour_mask(&self.source, w, h, cancel)?;
        cancel.check()?;
        binary_contour_image(&raster::Mask {
            w: w as usize,
            h: h as usize,
            data: core.as_raw().iter().map(|&p| p != 0).collect(),
        })
    }

    /// Return the post-cleanup foreground target contour core as a binary mask.
    ///
    /// This follows the same target stroke and foreground-support path as
    /// [`Self::resize_with_foreground_ink`], then runs the established
    /// `target_cleanup::thin_outer` canonicalization. It does not composite,
    /// darken, or antialias the diagnostic image. Pixels are `0` for contour
    /// and `255` for background.
    pub fn foreground_contour_mask(
        &self,
        foreground: &RgbaImage,
        w: u32,
        h: u32,
        aa: GameAssetAa,
        cancel: &dyn Cancellation,
    ) -> Result<GrayImage> {
        cancel.check()?;
        let (sw, sh) = self.source.dimensions();
        if foreground.dimensions() != (sw, sh)
            || w == 0
            || h == 0
            || sw == 0
            || sh == 0
            || w > sw
            || h > sh
        {
            return Err(Error::InvalidDimensions);
        }
        check_foreground_working_set_budget(sw, sh, w, h, self.memory_limit)?;
        let prepared = self.prepared(cancel)?;
        let mask = prepared.foreground_contour_mask(&self.source, foreground, w, h, aa, cancel)?;
        cancel.check()?;
        Ok(mask)
    }
    pub fn resize(
        &self,
        w: u32,
        h: u32,
        aa: GameAssetAa,
        cancel: &dyn Cancellation,
    ) -> Result<RgbaImage> {
        cancel.check()?;
        let (sw, sh) = self.source.dimensions();
        if w == 0 || h == 0 || sw == 0 || sh == 0 || w > sw || h > sh {
            return Err(Error::InvalidDimensions);
        }
        if (w, h) == (sw, sh) {
            return Ok((*self.source).clone());
        }
        check_working_set_budget(sw, sh, w, h, self.memory_limit)?;
        let cached = {
            let cache = self.cache.lock().expect("Game Asset cache poisoned");
            if let Some((key, result)) = &cache.target
                && *key == (w, h, aa)
            {
                return Ok((**result).clone());
            }
            cache.prepared.clone()
        };
        let prepared = cached.map(Ok).unwrap_or_else(|| self.prepared(cancel))?;
        let result = prepared.resize(&self.source, w, h, aa, cancel)?;
        cancel.check()?;
        self.cache.lock().expect("Game Asset cache poisoned").target =
            Some(((w, h, aa), Arc::new(result.clone())));
        Ok(result)
    }

    /// Resize a background-removed foreground while retaining contours and
    /// source-width ink measured from the original session artwork.
    pub fn resize_with_foreground(
        &self,
        foreground: &RgbaImage,
        w: u32,
        h: u32,
        aa: GameAssetAa,
        cancel: &dyn Cancellation,
    ) -> Result<RgbaImage> {
        cancel.check()?;
        let (sw, sh) = self.source.dimensions();
        if foreground.dimensions() != (sw, sh)
            || w == 0
            || h == 0
            || sw == 0
            || sh == 0
            || w > sw
            || h > sh
        {
            return Err(Error::InvalidDimensions);
        }
        if (w, h) == (sw, sh) {
            return Ok(foreground.clone());
        }
        check_foreground_working_set_budget(sw, sh, w, h, self.memory_limit)?;
        let prepared = self.prepared(cancel)?;
        let result = prepared.resize_with_foreground(&self.source, foreground, w, h, aa, cancel)?;
        cancel.check()?;
        Ok(result)
    }

    /// Resize an aligned extracted foreground using its visible colours for
    /// original-geometry ink, while leaving the resampled fill unchanged.
    ///
    /// `brightness` scales only contour RGB in displayed sRGB space and must
    /// be finite in the inclusive range `0.0..=1.0`. Foreground pixels with
    /// no alpha-supported donor suppress their corresponding contour pixels.
    pub fn resize_with_foreground_ink(
        &self,
        foreground: &RgbaImage,
        w: u32,
        h: u32,
        aa: GameAssetAa,
        brightness: f64,
        cancel: &dyn Cancellation,
    ) -> Result<RgbaImage> {
        cancel.check()?;
        if !brightness.is_finite() || !(0.0..=1.0).contains(&brightness) {
            return Err(Error::Scaling(
                "foreground ink brightness must be finite and within 0.0..=1.0".into(),
            ));
        }
        let (sw, sh) = self.source.dimensions();
        if foreground.dimensions() != (sw, sh)
            || w == 0
            || h == 0
            || sw == 0
            || sh == 0
            || w > sw
            || h > sh
        {
            return Err(Error::InvalidDimensions);
        }
        if (w, h) == (sw, sh) {
            return Ok(foreground.clone());
        }
        check_foreground_working_set_budget(sw, sh, w, h, self.memory_limit)?;
        let prepared = self.prepared(cancel)?;
        let result = prepared.resize_with_foreground_ink(
            &self.source,
            foreground,
            ForegroundInkResize {
                w,
                h,
                aa,
                brightness,
            },
            cancel,
        )?;
        cancel.check()?;
        Ok(result)
    }
}

/// Reduce a borrowed image once, avoiding a cloned source buffer or session cache.
pub fn resize(
    image: &RgbaImage,
    width: u32,
    height: u32,
    aa: GameAssetAa,
    cancel: &dyn Cancellation,
) -> Result<RgbaImage> {
    resize_with_options(image, width, height, aa, cancel, ResizeOptions::default())
}

/// Reduce a borrowed image once with the selected resize behavior.
pub fn resize_with_options(
    image: &RgbaImage,
    width: u32,
    height: u32,
    aa: GameAssetAa,
    cancel: &dyn Cancellation,
    options: ResizeOptions,
) -> Result<RgbaImage> {
    resize_with_memory_limit_and_options(
        image,
        width,
        height,
        aa,
        cancel,
        DEFAULT_MEMORY_LIMIT,
        options,
    )
}

/// Single-shot reduction with a caller-selected working-memory budget.
pub fn resize_with_memory_limit(
    image: &RgbaImage,
    width: u32,
    height: u32,
    aa: GameAssetAa,
    cancel: &dyn Cancellation,
    memory_limit: u64,
) -> Result<RgbaImage> {
    resize_with_memory_limit_and_options(
        image,
        width,
        height,
        aa,
        cancel,
        memory_limit,
        ResizeOptions::default(),
    )
}

/// Single-shot reduction with a caller-selected working-memory budget and
/// resize behavior.
pub fn resize_with_memory_limit_and_options(
    image: &RgbaImage,
    width: u32,
    height: u32,
    aa: GameAssetAa,
    cancel: &dyn Cancellation,
    memory_limit: u64,
    options: ResizeOptions,
) -> Result<RgbaImage> {
    cancel.check()?;
    let (sw, sh) = image.dimensions();
    if sw == 0 || sh == 0 || width == 0 || height == 0 || width > sw || height > sh {
        return Err(Error::InvalidDimensions);
    }
    if (sw, sh) == (width, height) {
        return Ok(image.clone());
    }
    check_working_set_budget(sw, sh, width, height, memory_limit)?;
    let prepared = Prepared::with_options(image, options, cancel)?;
    let output = prepared.resize(image, width, height, aa, cancel)?;
    cancel.check()?;
    Ok(output)
}

/// Downscale an externally generated, outline-free fill while tracing contours
/// from `original` and repainting them on top.
///
/// The two sources must have identical dimensions.  This keeps Qwen or another
/// editor's colour-only result aligned with contours traced from the untouched
/// artwork.  The fill itself is resampled in linear light with Lanczos3.
pub fn resize_with_outline_free_fill(
    original: &RgbaImage,
    outline_free_fill: &RgbaImage,
    width: u32,
    height: u32,
    aa: GameAssetAa,
    outline_color: OutlineColor,
    cancel: &dyn Cancellation,
) -> Result<RgbaImage> {
    cancel.check()?;
    if original.dimensions() != outline_free_fill.dimensions() {
        return Err(Error::InvalidDimensions);
    }
    let (sw, sh) = original.dimensions();
    if sw == 0 || sh == 0 || width == 0 || height == 0 || width > sw || height > sh {
        return Err(Error::InvalidDimensions);
    }
    check_working_set_budget(sw, sh, width, height, DEFAULT_MEMORY_LIMIT)?;
    let prepared = Prepared::new(original, cancel)?;
    prepared.resize_over_fill(
        original,
        outline_free_fill,
        FillResize {
            w: width,
            h: height,
            aa,
            outline_color,
        },
        cancel,
    )
}
