# asset-scaler

Shared contour-aware **Game Asset** downscaler for Diorama and Sprite Studio
(the `spritesheet` repository). Rust 1.92+, GPL-3.0-only.

The implementation was extracted from Diorama's Game Asset renderer and compared
with Sprite Studio's port. Both applications now consume this crate rather than
maintaining separate algorithms. Source analysis, direction-merged contours,
source-width ink opacity, premultiplied linear Lanczos3 fill, bounded halo
correction, and adjustable antialiasing live here.

## Usage

```rust
use asset_scaler::{CancellationToken, GameAssetAa, resize};
use image::RgbaImage;

let source = RgbaImage::new(64, 64);
let small = resize(&source, 32, 32, GameAssetAa::default(), &CancellationToken::default())?;
# Ok::<(), asset_scaler::Error>(())
```

`resize` borrows a source for single-shot frame processing. `Session` owns an
`Arc<RgbaImage>` and caches source analysis and one output for interactive
previews. Both run the same algorithm. Dimensions must be nonzero and no larger
than the source on either axis; enlargement belongs to the host application.
AA is clamped to 0–100%, defaulting to 50%.

`Cancellation` accepts native tokens or `Sync` closures, for example
`&|| tokio_token.is_cancelled()`. Cancellation is checked even for identity and
cached requests. `resize_with_memory_limit` and `Session::with_memory_limit`
allow callers to preserve their own working-set limits (Sprite Studio uses
1 GiB, Diorama 4 GiB). This is a conservative preflight estimate, not an OS
allocation quota.

## Silhouette fixes

Explicit source alpha owns transparent asset geometry. Internal color ridges
must not carve holes into opaque source-supported boots or limbs. Opaque,
flat-background thin components without a reliable fill interior retain their
terminal rows. Regression tests cover both cases, including the previously
ignored off-grid-line test in Sprite Studio. Background removal itself is a
separate operation and is not performed by this crate.

## Development

```sh
cargo test --all-targets
cargo test --doc
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
cargo run --example resize -- input.png output.png 200 200
```

Keep `Cargo.lock` committed for repeatable CI. The applications initially pin a
Git revision of this repository; update both pins together after a shared fix.
For local development, Cargo's `--config` can patch the Git dependency to the
sibling checkout without committing an absolute path into either application.

## GitHub identity

Origin: `https://github.com/mendrik-private/asset-scaler.git`.
After cloning, run `scripts/configure-git-identity.sh`. It configures **only this
repository** with the `mendrik-private` commit identity and a credential helper
that explicitly reads that account from `gh`, ignoring globally active accounts
and environment tokens. It fails if that account is unavailable; it never falls
back to another account. Tokens are not stored in this repository.

## Publishing

`.github/workflows/publish.yml` publishes a matching `v<VERSION>` tag pushed by
`mendrik-private`, after formatting, Clippy, tests and packaging pass. Manual
reruns must also select that tag. Publishing uses short-lived
[crates.io trusted-publishing tokens](https://github.com/rust-lang/crates-io-auth-action),
not the GitHub credential helper.

Before the first release, use the intended crates.io owner account to register a
trusted publisher for owner `mendrik-private`, repository `asset-scaler`, workflow
`publish.yml`, environment `crates-io`. If initial publication requires a manual
bootstrap, use that same owner's narrowly scoped crates.io token. No release tag
or actual crates.io publication is performed by extraction/setup.

See the [Cargo publishing guide](https://doc.rust-lang.org/cargo/reference/publishing.html).
