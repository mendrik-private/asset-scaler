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
impl Prepared {
    fn new(image: &RgbaImage, cancel: &dyn Cancellation) -> Result<Self> {
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
        let silhouette = silhouette::Silhouette::detect(image, cancel)?;
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
}
impl Session {
    pub fn new(source: Arc<RgbaImage>) -> Self {
        Self::with_memory_limit(source, DEFAULT_MEMORY_LIMIT)
    }
    /// Create a cached session with a caller-selected working-memory limit.
    pub fn with_memory_limit(source: Arc<RgbaImage>, memory_limit: u64) -> Self {
        Self {
            source,
            memory_limit,
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
        let prepared = Arc::new(Prepared::new(&self.source, cancel)?);
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
    resize_with_memory_limit(image, width, height, aa, cancel, DEFAULT_MEMORY_LIMIT)
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
    cancel.check()?;
    let (sw, sh) = image.dimensions();
    if sw == 0 || sh == 0 || width == 0 || height == 0 || width > sw || height > sh {
        return Err(Error::InvalidDimensions);
    }
    if (sw, sh) == (width, height) {
        return Ok(image.clone());
    }
    check_working_set_budget(sw, sh, width, height, memory_limit)?;
    let prepared = Prepared::new(image, cancel)?;
    let output = prepared.resize(image, width, height, aa, cancel)?;
    cancel.check()?;
    Ok(output)
}
