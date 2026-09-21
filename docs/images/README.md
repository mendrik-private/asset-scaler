# README image assets

`elf-source.png` is the supplied 800×800 transparent elf illustration, copied
unchanged from `mule/assets/generated/characters/elf.png`. All other PNGs are
generated from that source by the scaler, without retouching. The README displays
the transparent assets inside `*-preview.svg` frames with a mid-gray
(`#808080`) background. Each SVG embeds the original PNG unchanged; the
background is only for presentation and does not alter the downloadable PNG.

From the repository root, regenerate them with:

```sh
cargo test -p asset-scaler --release --lib generate_readme_images -- --ignored
python3 scripts/generate-readme-previews.py
```

| File | Contents |
| --- | --- |
| `elf-contours.png` | 200×200 retained and smoothed contour cores (`strokes.core`), white on black. |
| `elf-contour-coverage.png` | 200×200 contour AA coverage at 50%, before source-width opacity. |
| `elf-fill-mask.png` | 200×200 silhouette `target_coverage` at 50%, rounded to 8-bit grayscale for display. White means full coverage; black means exterior. This is coverage, not the final image's alpha or an inverse ink mask. |
| `elf-fill.png` | 200×200 linear-light premultiplied Lanczos3 resample of the isolated source, converted to sRGB for display, before halo correction and final silhouette coverage. |
| `elf-aa-{0,50,100}.png` | Actual 200×200 outputs of `Prepared::resize` at the three AA settings. Transparency is preserved. |
| `elf-aa-{0,50,100}-detail.png` | Crop at `(78,44)` with size 50×60 from each output, enlarged to 200×240 with nearest-neighbor. |

The SVG frames are generated using Python's standard library. They are
self-contained, with no external image references or dependency on inline HTML
styles. Diagnostic masks retain their black-to-white scale so their values
remain readable.

The detail crops encode their 4× pixel blocks directly in the PNG files.
The full sprites are displayed at their native 200×200 size.
