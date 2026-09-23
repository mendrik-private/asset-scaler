#!/usr/bin/env python3
"""Create outline-free Qwen Image 2.1 fill layers with local sd.cpp GGUFs."""

import argparse
import os
import subprocess
import sys
import zlib
from pathlib import Path

PROMPT = """Remove all dark outlines, ink contours, black or brown line art, and
the anti-aliased dark fringe surrounding those lines. Preserve every colorful
fill region, silhouette, pose, proportions, clothing, hair, face, palette,
lighting, texture, transparent background, and composition exactly. Return
flat clean color fills only: no dark contour remnants, halos, edge bleed,
new objects, text, watermark, frame, shadow, or background."""


def pngs(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.suffix.lower() == ".png")


def edit(source: Path, destination: Path, args: argparse.Namespace) -> None:
    from PIL import Image

    with Image.open(source) as opened:
        original = opened.convert("RGBA")
    width, height = original.size
    if width % 32 or height % 32:
        raise ValueError(f"{source} is {width}x{height}; Qwen needs dimensions divisible by 32")
    command = [
        str(args.sd_cli), "--diffusion-model", str(args.diffusion_model),
        "--vae", str(args.vae), "--llm", str(args.llm),
        "--llm_vision", str(args.llm_vision), "-r", str(source),
        "-p", PROMPT, "--width", str(width), "--height", str(height),
        "--steps", str(args.steps), "--strength", str(args.strength),
        "--cfg-scale", str(args.cfg_scale), "--sampling-method", "euler",
        "--offload-to-cpu", "--diffusion-fa", "-s",
        str(zlib.crc32(source.name.encode())), "-o", str(destination),
    ]
    environment = os.environ.copy()
    library_path = "/home/mendrik/ai/therock-7.12/lib:/home/mendrik/ai/therock-7.12/lib/llvm/lib"
    environment["LD_LIBRARY_PATH"] = library_path + ":" + environment.get("LD_LIBRARY_PATH", "")
    subprocess.run(command, check=True, env=environment)
    with Image.open(destination) as opened:
        rendered = opened.convert("RGBA")
    if rendered.size != original.size:
        rendered = rendered.resize(original.size, Image.Resampling.LANCZOS)
    rendered.putalpha(original.getchannel("A"))
    rendered.save(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--sd-cli", type=Path, required=True)
    parser.add_argument("--diffusion-model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--llm-vision", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        import PIL  # noqa: F401
    except ImportError as error:
        parser.error("Pillow is required: " + str(error))
    inputs = pngs(args.source_dir)
    if not inputs:
        parser.error(f"no PNG files in {args.source_dir}")
    paths = (args.sd_cli, args.diffusion_model, args.vae, args.llm, args.llm_vision)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        parser.error("missing required model/runtime path(s): " + ", ".join(map(str, missing)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in inputs[:args.limit]:
        destination = args.output_dir / source.name
        if destination.exists() and not args.overwrite:
            print(f"skip {source.name}", flush=True)
            continue
        print(f"render {source.name}", flush=True)
        edit(source, destination, args)
        print(f"wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
