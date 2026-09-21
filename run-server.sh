#!/usr/bin/env sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

# This launcher builds for the machine running the local server. Honor explicit
# compiler flags, including empty flags when a portable build is wanted.
if [ "${RUSTFLAGS+x}" != x ] && [ "${CARGO_ENCODED_RUSTFLAGS+x}" != x ]; then
    export RUSTFLAGS="-C target-cpu=native"
fi

exec cargo run --manifest-path "$script_dir/Cargo.toml" --release -p asset-scaler-server "$@"
