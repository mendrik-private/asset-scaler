# Local InSPyReNet runtime

`asset-background-removal` uses the upstream
[`transparent-background`](https://github.com/plemeri/transparent-background)
InSPyReNet **base** checkpoint at its documented 1024 × 1024 inference size.
The local runner keeps one loaded model while its Python runtime, checkpoint
contents, device selection, and PyTorch backend identity are unchanged. It
restores each raw saliency mask to the original image canvas, and does not
download a checkpoint or apply matting.

Each source is decoded and inferred one at a time using FP32 `torch.no_grad()`.
The runner communicates over bounded JSON lines, emits diagnostics only on
standard error, and atomically publishes a raw grayscale mask before it is
acknowledged by the bounded CPU consumer. Cancellation, a protocol error, a
timeout, or a changed runtime identity stops and reaps the local process; a
later operation starts a fresh process.

The default checkpoint path is:

```text
~/.transparent-background/ckpt_base.pth
```

Install the runner in its shared user-local location. The venv may reuse an
existing compatible PyTorch installation:

```sh
python3 -m venv --system-site-packages ~/.local/share/asset-scaler/inspyrenet-venv
~/.local/share/asset-scaler/inspyrenet-venv/bin/pip install transparent-background==1.3.4
```

Set these variables in the environment of the application or server only when
using non-default local paths or forcing a device:

```sh
export ASSET_SCALER_INSPYRENET_PYTHON="$HOME/.local/share/asset-scaler/inspyrenet-venv/bin/python"
export ASSET_SCALER_INSPYRENET_CHECKPOINT="$HOME/.transparent-background/ckpt_base.pth"
export ASSET_SCALER_INSPYRENET_DEVICE=auto # auto, cpu, or cuda
```

For migration compatibility, `SPRITE_STUDIO_INSPYRENET_PYTHON`,
`SPRITE_STUDIO_INSPYRENET_CHECKPOINT`, and `SPRITE_STUDIO_INSPYRENET_DEVICE`
are used only when their matching `ASSET_SCALER_*` variable is unset. If no
Python variable is set, the runtime prefers the new venv above and then falls
back to `~/.local/share/sprite-studio/inspyrenet-venv/bin/python` when it
exists, then to `python3`. Missing or incompatible runtime files stop
processing; this runtime never downloads a model or selects another
background-removal implementation.
