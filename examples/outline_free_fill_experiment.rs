use asset_scaler::{CancellationToken, GameAssetAa, OutlineColor, resize_with_outline_free_fill};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: outline_free_fill_experiment ORIGINAL_DIR FILL_DIR OUTPUT_DIR TARGET_PX".into(),
        );
    }
    let source = Path::new(&args[1]);
    let fills = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    let target: u32 = args[4].parse()?;
    std::fs::create_dir_all(output.join("original-ink"))?;
    std::fs::create_dir_all(output.join("fill-25-luminance"))?;
    let mut inputs = std::fs::read_dir(source)?.collect::<Result<Vec<_>, _>>()?;
    inputs.sort_by_key(|entry| entry.file_name());
    let cancel = CancellationToken::default();
    for entry in inputs {
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("png") {
            continue;
        }
        let name = entry.file_name();
        let fill_path = fills.join(&name);
        if !fill_path.is_file() {
            return Err(format!("missing Qwen fill: {}", fill_path.display()).into());
        }
        let original = image::open(&path)?.into_rgba8();
        let fill = image::open(&fill_path)?.into_rgba8();
        let original_ink = resize_with_outline_free_fill(
            &original,
            &fill,
            target,
            target,
            GameAssetAa::new(30),
            OutlineColor::OriginalInk,
            &cancel,
        )?;
        let fill_ink = resize_with_outline_free_fill(
            &original,
            &fill,
            target,
            target,
            GameAssetAa::new(30),
            OutlineColor::DarkenedFill { luminance: 0.25 },
            &cancel,
        )?;
        original_ink.save(output.join("original-ink").join(&name))?;
        fill_ink.save(output.join("fill-25-luminance").join(name))?;
    }
    Ok(())
}
