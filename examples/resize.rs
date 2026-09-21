use asset_scaler::{CancellationToken, GameAssetAa, resize};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: resize INPUT.png OUTPUT.png WIDTH HEIGHT".into());
    }
    let image = image::open(&args[1])?.into_rgba8();
    resize(
        &image,
        args[3].parse()?,
        args[4].parse()?,
        GameAssetAa::default(),
        &CancellationToken::default(),
    )?
    .save(&args[2])?;
    Ok(())
}
