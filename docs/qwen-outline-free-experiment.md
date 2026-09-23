# Qwen outline-free fill experiment

This non-destructive experiment uses the local Qwen-Image-2.1 GGUF diffusion
model to create a clean fill layer, then traces the untouched source contours
back over a linear-light Lanczos downscale of that fill.

The result contains two 256px sets:

| Directory | Contour colour | AA |
| --- | --- | --- |
| `original-ink/` | sampled from the source | 30% |
| `fill-25-luminance/` | Qwen fill beneath each contour, at 25% linear luminance | 30% |

## Local Qwen GGUF run

The local `sd.cpp` HIP build drives the requested
`abenzerps/Qwen-Image-2.1-GGUF` Q4_K_M diffusion model. Qwen Image editing
also needs its local Qwen3-VL text/vision encoder GGUFs and VAE. Run
resumably with the existing local downloads:

```sh
python3 scripts/qwen_outline_free_fills.py \
  /home/mendrik/desk/mendrik/mule/assets/generated/characters \
  experiments/qwen-outline-free/fills \
  --sd-cli /home/mendrik/sd.cpp/build/bin/sd-cli \
  --diffusion-model /home/mendrik/.cache/huggingface/hub/models--abenzerps--Qwen-Image-2.1-GGUF/snapshots/c4de66efa2183fb25ecbc185bac509ee37952b41/qwen-image-2.1-Q4_K_M.gguf \
  --vae /home/mendrik/.cache/huggingface/hub/models--abenzerps--Qwen-Image-2.1-GGUF/snapshots/c4de66efa2183fb25ecbc185bac509ee37952b41/vae/qwen_image_2.1_vae_bf16.safetensors \
  --llm /home/mendrik/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct-GGUF/snapshots/f982a07559d4a2f6c8744d840bf6fccab30eea96/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
  --llm-vision /home/mendrik/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct-GGUF/snapshots/f982a07559d4a2f6c8744d840bf6fccab30eea96/mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf
```

The runner preserves the source alpha after each render, aligns any generated
size back to the source canvas, uses a deterministic filename-derived seed,
and skips completed files. `--steps 20` is the quality default; use a lower
step count only for a quick smoke test.

Then make both target-size variants:

```sh
cargo run --release --example outline_free_fill_experiment -- \
  /home/mendrik/desk/mendrik/mule/assets/generated/characters \
  experiments/qwen-outline-free/fills \
  experiments/qwen-outline-free/outputs 256
```

Neither stage overwrites source assets.
