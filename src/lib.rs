//! Game Asset reduction: direction-merged contours, source-width opacity,
//! tight antialiasing, and Lanczos3 source fill with bounded halo tone-down.

#![doc = include_str!("../README.md")]

mod api;
pub use api::{Cancellation, CancellationToken, Error, GameAssetAa, Result};
use image::RgbaImage;
use std::sync::{Arc, Mutex};
mod antialias;
#[cfg(test)]
mod benchmarks;
mod cleanup;
mod color;
mod contours;
mod coverage;
mod detect;
mod field;
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
#[cfg(test)]
mod tests;
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
struct Prepared {
    models: Vec<detect::Model>,
    widths: Vec<f64>,
    contours: contours::Contours,
    mask: raster::Mask,
    linear: color::LinearImage,
    silhouette: Option<silhouette::Silhouette>,
}
struct TargetContours {
    strokes: strokes::Strokes,
    colors: Vec<[f64; 3]>,
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
        let (raw, distance) = source::rasterize(
            &models,
            image.width() as usize,
            image.height() as usize,
            cancel,
        )?;
        let thinned = cleanup::thin(&raw, &distance, cancel)?;
        cancel.check()?;
        let contours = contours::Contours::new(&thinned, &models, cancel)?;
        let mask = ink::ink_mask(image, &samples, &thinned, cancel)?;
        cancel.check()?;
        let widths = opacity::widths(image, &mask, &samples, &contours, cancel)?;
        let linear = color::LinearImage::from_rgba(image);
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
        let scale = [
            w as f64 / image.width() as f64,
            h as f64 / image.height() as f64,
        ];
        let (retained, owners) =
            self.contours
                .retain_with_ids(&self.models, scale, contours::MAX_SHORT_PIXELS);
        let smoothed = smoothing::smooth(&retained, scale[0].min(scale[1]), cancel)?;
        let curves: Vec<_> = smoothed
            .iter()
            .map(|m| {
                detect::controls(m, 1.)
                    .map(|p| [(p[0] + 0.5) * scale[0] - 0.5, (p[1] + 0.5) * scale[1] - 0.5])
            })
            .collect();
        cancel.check()?;
        let strokes = strokes::render(&curves, &owners, &self.widths, w, h, aa, cancel)?;
        let colors = paint::ink_colors(
            &self.linear,
            &retained,
            &smoothed,
            &owners,
            &strokes,
            scale,
            cancel,
        )?;
        Ok(TargetContours { strokes, colors })
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
        let TargetContours { strokes, colors } =
            self.target_contours(contour_source, w, h, aa, cancel)?;
        let strength = opacity::calculate(&self.widths, &strokes.core, &strokes.owners);
        let paint = opacity::apply(&strokes.coverage, &strokes.owners, &strength);
        let scale = [
            w as f64 / contour_source.width() as f64,
            h as f64 / contour_source.height() as f64,
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
        let base = halo::apply(fill_source, &retained_mask, base, &strokes.core, cancel)?;
        let fill = if let Some(silhouette) = fill_silhouette {
            let source_coverage = silhouette.coverage(w as usize, h as usize, cancel)?;
            let coverage = silhouette.target_coverage(&source_coverage, aa, cancel)?;
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
        let result = paint::composite(&fill, &colors, &paint);
        cancel.check()?;
        Ok(result)
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

    fn resize_over_fill(
        &self,
        original: &RgbaImage,
        fill_source: &RgbaImage,
        request: FillResize,
        cancel: &dyn Cancellation,
    ) -> Result<RgbaImage> {
        let TargetContours { strokes, colors } =
            self.target_contours(original, request.w, request.h, request.aa, cancel)?;
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
