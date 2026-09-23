//! Manual documentation exporter, using the production pipeline's private stages.
use super::*;
use image::{GrayImage, Luma, imageops};

#[test]
#[ignore = "regenerate README images from docs/images/elf-source.png"]
fn generate_readme_images() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/images");
    let source = image::open(directory.join("elf-source.png"))?.into_rgba8();
    assert_eq!(source.dimensions(), (800, 800));
    let cancel = CancellationToken::default();
    let prepared = Prepared::new(&source, &cancel)?;
    let aa = GameAssetAa::default();
    let target = prepared.target_contours(&source, 200, 200, aa, &cancel)?;
    target
        .strokes
        .core
        .save(directory.join("elf-contours.png"))?;
    target
        .strokes
        .coverage
        .save(directory.join("elf-contour-coverage.png"))?;

    // Match the silhouette support coverage used by resize.
    let silhouette = prepared.silhouette.as_ref().expect("elf has source alpha");
    let source_coverage = silhouette.coverage(200, 200, &cancel)?;
    let coverage = silhouette.target_coverage(&source_coverage, aa, &cancel)?;
    GrayImage::from_fn(200, 200, |x, y| {
        Luma([(coverage[(y * 200 + x) as usize] * 255.).round() as u8])
    })
    .save(directory.join("elf-fill-mask.png"))?;

    // This is the actual color resample before halo correction and composition.
    let isolated = silhouette.isolated(&prepared.linear, &cancel)?;
    let base = lanczos::resize(&isolated, 200, 200, &cancel)?;
    RgbaImage::from_fn(200, 200, |x, y| {
        color::rgba(base.pixels[(y * 200 + x) as usize])
    })
    .save(directory.join("elf-fill.png"))?;

    let mut outputs = Vec::new();
    for percent in [0, 50, 100] {
        let output = prepared.resize(&source, 200, 200, GameAssetAa::new(percent), &cancel)?;
        assert_eq!(output.dimensions(), (200, 200));
        output.save(directory.join(format!("elf-aa-{percent}.png")))?;
        // Bake nearest-neighbor enlargement into PNGs: GitHub removes inline CSS.
        let detail = imageops::crop_imm(&output, 78, 44, 50, 60).to_image();
        imageops::resize(&detail, 200, 240, imageops::FilterType::Nearest)
            .save(directory.join(format!("elf-aa-{percent}-detail.png")))?;
        outputs.push(output);
    }
    assert_ne!(
        outputs[0], outputs[1],
        "AA comparison must show distinct pixels"
    );
    assert_ne!(
        outputs[1], outputs[2],
        "AA comparison must show distinct pixels"
    );
    Ok(())
}

/// Reproducible geometry review for the supplied female-elf source.  This
/// exports the authoritative fitted cubics before their quadratic renderer
/// approximation, so a raster grid cannot hide a poor curve join.
#[test]
#[ignore = "manual spline review export"]
fn export_female_elf_spline_svg() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::fmt::Write;

    let source =
        image::open("/home/mendrik/desk/mendrik/mule/assets/characters/01-female-elf.png")?
            .into_rgba8();
    let output = std::path::Path::new("/tmp/diorama-spline-review");
    std::fs::create_dir_all(output)?;
    let cancel = CancellationToken::default();
    let prepared = Prepared::new(&source, &cancel)?;
    for (w, h) in [(source.width(), source.height()), (200, 200)] {
        let scale = [
            w as f64 / source.width() as f64,
            h as f64 / source.height() as f64,
        ];
        let fit = prepared
            .contours
            .polished(scale, contours::MAX_SHORT_PIXELS, &cancel)?;
        let mut svg = format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/><g fill="none" stroke="black" stroke-width="0.55" stroke-linecap="round">"#
        );
        for cubic in fit.cubic_curves {
            let p = cubic.map(|p| [(p[0] + 0.5) * scale[0] - 0.5, (p[1] + 0.5) * scale[1] - 0.5]);
            writeln!(
                svg,
                r#"<path d="M {:.3} {:.3} C {:.3} {:.3}, {:.3} {:.3}, {:.3} {:.3}"/>"#,
                p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1], p[3][0], p[3][1]
            )?;
        }
        svg.push_str("</g></svg>");
        std::fs::write(output.join(format!("female-elf-fitted-{w}.svg")), svg)?;
    }
    Ok(())
}
