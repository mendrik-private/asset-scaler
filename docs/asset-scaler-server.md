# Asset scaler HTTP server

`asset-scaler-server` is the loopback HTTP service for the shared Game Asset
scaler and InSPyReNet background remover. Start it from this workspace:

```sh
cd ~/desk/mendrik/asset-scaler
./run-server.sh
```

`./run-server.sh` works from any working directory and builds in release mode
with `-C target-cpu=native` to use the local CPU's SIMD features. The resulting
binary is intended for this machine. Explicit `RUSTFLAGS` or
`CARGO_ENCODED_RUSTFLAGS` take precedence; `RUSTFLAGS='' ./run-server.sh`
uses the portable compiler defaults. Direct `cargo run --release -p
asset-scaler-server` also retains the ordinary build defaults.
The server listens on `http://127.0.0.1:47831`, keeps the local InSPyReNet
model warm, and accepts one image at a time. Set
`ASSET_SCALER_SERVER_ADDRESS` to use another loopback bind address; the legacy
`SPRITE_STUDIO_ASSET_SERVER_ADDRESS` is accepted only when the shared variable
is unset. Non-loopback addresses are rejected before the model starts. See
[the local runtime guide](inspyrenet-runtime.md) for the model environment.

Before model inference, the server applies a conservative model-memory
preflight. Set `ASSET_SCALER_PROCESSING_MEMORY_BUDGET_MIB` to a positive MiB
budget; `SPRITE_STUDIO_PROCESSING_MEMORY_BUDGET_MIB` is used when the shared
variable is unset. The default is 2048 MiB. This setting does not change the
Game Asset reduction limit, which remains fixed at 1 GiB.

`POST /process` accepts raw PNG, JPEG, or WebP bytes and returns transparent
`image/png` on an exact 200 × 200 canvas by default. Supply `width` and
`height` together to request another exact canvas from 8 through 512 pixels on
each axis. Uploads are limited to 8 MiB and source images to 3840 × 2160
pixels. The default controls are `aa=20` and `feather=0`.

```sh
curl --fail-with-body --data-binary @character.webp \
  -H 'Content-Type: image/webp' \
  'http://127.0.0.1:47831/process?width=128&height=256&aa=20&feather=0' \
  -o character-128x256.png
```

`aa` and `feather` are independent optional integers from 0 through 100. `aa`
controls Game Asset anti-aliasing during downscaling only; identity-sized
images are copied and enlargements use bicubic scaling. `feather` remaps the
source-sized model confidence before it multiplies source alpha: `0` is a hard
mask (`127` is background, `128` foreground), `100` preserves the raw mask,
and an intermediate percentage is a linear confidence band centered at 0.5.
Game Asset scaling can still create semi-transparent edge pixels from a hard
source mask. `mask=model` is the default. `mask=edge-matte` instead removes a
connected, edge-coloured matte and bypasses model inference for that request;
the server still validates and warms the InSPyReNet runtime at startup.

`GET /health` returns `ok`. Browser requests from `localhost`, `127.0.0.1`, or
`::1` receive CORS headers; other origins do not. A second concurrent
`POST /process` receives HTTP 429 instead of waiting in memory. Once accepted,
a job completes even if its client disconnects, so its model and temporary
files cannot overlap a later request.

```js
const url = new URL("http://127.0.0.1:47831/process");
url.searchParams.set("width", "128");
url.searchParams.set("height", "256");
url.searchParams.set("aa", "20");
url.searchParams.set("feather", "0");
const response = await fetch(url, {
  method: "POST",
  headers: { "Content-Type": file.type || "image/png" },
  body: file,
});
if (!response.ok) throw new Error(await response.text());
const sprite = await response.blob();
```

For a detached server, find its PID with `pgrep -af asset-scaler-server` and
stop it gracefully with `kill -INT <pid-from-pgrep>`.
