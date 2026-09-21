# CPU SIMD opportunities

Research date: 2026-09-21. Includes a local release-build comparison; results are specific to this machine and these workloads.

## Recommendation

Use a local `-C target-cpu=native` build as the first optimization experiment: the comparison below found lower source-analysis latency. The current Gaussian convolution already exposes independent, contiguous `f64` pixel operations to LLVM, so wider instructions may help without rewriting the algorithm. Keep the ordinary build portable; add runtime-selected kernels only if measurement justifies the complexity.

## Hardware and compiler

Local `lscpu` reports **AMD RYZEN AI MAX+ PRO 395**, 16 cores / 32 threads, and `avx`, `avx2`, `fma`, `avx512f`, `avx512dq`, `avx512bw`, and `avx512vl`. The relevant term is AVX, not “AVIX.” AVX2 and AVX-512 are x86 SIMD extensions; they are not exclusively AMD features. AMD describes their vector widths as 256 and 512 bits respectively. [AMD AVX-512 overview](https://www.amd.com/en/blogs/2026/understanding-avx-512---validating-usage-on-amd-epyc-.html)

`rustc -C target-cpu=native` selects the build machine's processor and its default feature set. It is appropriate for a binary built and run on this machine, but not a general distribution baseline. `rustc --print target-cpus` and `rustc --print target-features` list compiler-supported choices. Enabling features globally assumes the runtime CPU supports them. [rustc codegen options](https://doc.rust-lang.org/rustc/codegen-options/)

AVX-512 support alone does not predict a doubling in application speed. AMD's Zen 4 server design executes through 256-bit paths, while Zen 5 EPYC has full 512-bit paths; actual wins also depend on instruction selection and workload. These architectural references describe EPYC, not a benchmark of this Ryzen machine. [AMD Zen 4 architecture](https://www.amd.com/content/dam/amd/en/documents/epyc-business-docs/white-papers/221704010-B_en_4th-Gen-AMD-EPYC-Processor-Architecture---White-Paper_pdf.pdf), [AMD Zen 5 EPYC announcement](https://ir.amd.com/news-events/press-releases/detail/1219/amd-launches-5th-gen-amd-epyc-cpus-maintaining-leadership-performance-and-features-for-the-modern-data-center)

## Project-specific priorities

The following conclusions are based on repository inspection:

1. **Gaussian convolution is the best first candidate.** `src/field.rs` accumulates each tap across a contiguous row. Pixels are independent SIMD lanes, while each pixel retains tap order. Its existing `shared_derivatives_match_scalar_taps_and_reflection_exactly` test checks bit equality against the scalar reference, including signed zero and narrow widths. Inspect generated assembly before assuming this is scalar today.
2. **Preserve numerical semantics.** Do not introduce global fast math or replace multiply-plus-add with fused operations as a prerequisite for SIMD. The existing loop organization can vectorize across pixels without reordering a pixel's reduction. LLVM explains that floating-point reduction vectorization commonly needs permission to reorder operations, which changes results. [LLVM vectorizers](https://llvm.org/docs/Vectorizers.html)
3. **Lanczos merits profiling, not an immediate rewrite.** `src/lanczos.rs` delegates its main resampling to `image::imageops::resize`, with conversion loops around it. Replacing the resampler can change output and should be checked against image fixtures and alpha handling.
4. **Color decoding is already optimized algorithmically.** `src/color.rs` uses a 256-entry lookup for byte-to-linear conversion. Encoding still uses `powf`, but the report has no timing evidence that this dominates. Avoid spending effort on explicit gathers until profiling identifies a need.
5. **Irregular contour/spatial work is less obvious SIMD work.** Hash lookups, variable-length candidate lists, branching, and deterministic ordering in `Spatial` and contour code deserve algorithm/data-layout profiling before intrinsics.

## Portable implementation options

Rust documents runtime dispatch using `is_x86_feature_detected!("avx2")`, an appropriately annotated `#[target_feature]` function, and a fallback. Detect the actual features a kernel requires, not the AMD vendor name. [Rust architecture intrinsics and dispatch](https://doc.rust-lang.org/stable/core/arch/)

This crate forbids unsafe code in `Cargo.toml`. A handwritten intrinsic/dispatch implementation would conflict with that policy; preserving it requires relying on compiler vectorization or a dependency that exposes a safe abstraction. Standard `std::simd` is still nightly-only according to the current official API documentation, so it is not an immediate fit for this stable Rust project. [Rust portable SIMD status](https://doc.rust-lang.org/std/simd/index.html)

Rayon is a separate optimization axis: it distributes work across threads. It does not itself turn scalar arithmetic into AVX. Row-level parallelism could coexist with vectorized inner loops, but thread overhead and existing server request concurrency must be measured. The root crate currently has no Rayon dependency. [Rayon project documentation](https://github.com/rayon-rs/rayon)

## Suggested comparison

Use the existing manual benchmark and separate Cargo output directories so both variants remain available:

```sh
CARGO_TARGET_DIR=target/cpu-baseline RUSTFLAGS='' cargo test -p asset-scaler --release --lib game_asset_benchmark -- --ignored --nocapture --test-threads=1
CARGO_TARGET_DIR=target/cpu-native RUSTFLAGS='-C target-cpu=native' cargo test -p asset-scaler --release --lib game_asset_benchmark -- --ignored --nocapture --test-threads=1
CARGO_TARGET_DIR=target/cpu-native RUSTFLAGS='-C target-cpu=native' cargo test -p asset-scaler --release --lib
```

Repeat timed runs sequentially under similar load, compare stage medians and full first-preview latency, and preserve fixture output. A native speedup cannot by itself be attributed specifically to AVX-512: CPU tuning and multiple features change together. Follow up with an AVX2-only target and assembly inspection if identifying the cause matters. Prefer measurements over an assumed speedup from lane count.

## Local measurements

Compiler: rustc 1.94.1 (LLVM 21.1.8), x86_64-unknown-linux-gnu. Same source, release defaults, baseline without RUSTFLAGS versus `RUSTFLAGS='-C target-cpu=native'`. Saved executables were run sequentially in baseline/native/native/baseline order, with `DIORAMA_BENCH_SAMPLES=7` and the existing `game_asset_benchmark --ignored --nocapture --test-threads=1` harness. Each case has its existing untimed pilot. CPU affinity and frequency were not fixed; this is a local screening measurement, not a controlled laboratory benchmark. Image decoding is excluded. Source analysis plus the first target is reported as first resize.

Values below are the two run medians (ms), not pooled medians. Reduction compares the arithmetic means of those medians.

| Case / stage | Baseline run medians | Native run medians | Time reduction |
| --- | ---: | ---: | ---: |
| elf / analysis | 182.222, 181.496 | 156.539, 155.891 | 14.1% |
| elf / first_resize | 492.954, 489.506 | 476.821, 465.065 | 4.1% |
| odd / analysis | 11.274, 11.154 | 9.619, 9.586 | 14.4% |
| odd / first_resize | 16.821, 16.542 | 15.045, 14.973 | 10.0% |
| large / analysis | 201.152, 198.904 | 175.471, 169.138 | 13.9% |
| large / first_resize | 380.784, 371.296 | 355.227, 345.092 | 6.9% |

Cases: elf 800×800 → first target 128×128; odd synthetic 257×193 → 64×48; large synthetic 1024×768 → 128×96. Source analysis takes approximately 14–15% less time; first resize takes approximately 4–10% less time. Resizes reusing analysis show small or mixed changes, including slower native results on some targets. This does not support promising a large overall speedup or a uniform improvement for cached-source previews.

Disassembly (`objdump -Cd`) of `Field::convolve_*` confirms baseline packed double operations use XMM `mulpd`/`addpd`, while native uses ZMM `vmulpd`/`vaddpd` (512-bit, eight f64 lanes). Wider SIMD is real here, but this experiment does not isolate AVX-512 from all other native CPU tuning.

Both release library suites passed: 48 tests each, 3 manual tests ignored. This includes the scalar-reference bitwise Gaussian test and image regressions. All four benchmark runs passed their determinism checks. First-target CRC32s matched across both builds: elf `929c4772`, odd `92c453f8`, large `7fdee5a6`. These checks are not an exhaustive cross-build pixel comparison of all possible inputs or every benchmark target. No integration/workspace-wide test claim is made.

No production source or permanent build configuration was changed. Build commands temporarily resolved the in-progress workspace lockfile; that benchmark-induced lockfile change was restored. Existing user changes were preserved.

For a locally deployed Rust application, test `RUSTFLAGS='-C target-cpu=native' cargo build --release` from that application's workspace. A library consumer's final build controls CPU targeting. Do not distribute the resulting binary to CPUs lacking the selected features. For portable distribution, the next experiment is a safely abstracted runtime-selected convolution kernel, retaining the generic path and numerical order. Profile Lanczos and contour work before undertaking further SIMD rewrites.

The server's background-removal model runs in a separate Python/PyTorch runtime (`asset-background-removal/scripts/inspyrenet_video.py`); Rust CPU flags do not retune that runtime. The measurements above cover the Rust scaler only.

## Applied local build and follow-up profile

The local `run-server.sh` launcher now defaults to `RUSTFLAGS='-C target-cpu=native'`.
Explicit `RUSTFLAGS` or `CARGO_ENCODED_RUSTFLAGS`, including an empty value,
override that default. Ordinary Cargo builds and CI retain their existing
target settings. Use `RUSTFLAGS='' ./run-server.sh` for portable defaults.

Profiled the current native release library test executable with Linux perf:

```sh
DIORAMA_BENCH_CASE=elf DIORAMA_BENCH_SAMPLES=7 perf record \
  -e cycles:u -F 499 -o /tmp/asset-scaler-profile-elf.data -- \
  /tmp/asset-scaler-profile-native game_asset_benchmark \
  --ignored --nocapture --test-threads=1
perf report --stdio --no-children --percent-limit 1 \
  -i /tmp/asset-scaler-profile-elf.data --sort symbol
```

Repeated with `DIORAMA_BENCH_CASE=large`. These are flat sampled user-space
cycle shares for the entire harness (including setup, pilot, source analysis,
three output sizes and comparisons), not isolated first-resize timings or
inclusive call-tree costs. Elf collected 3,839 samples; large collected 2,068;
neither reported lost samples. Inlining affects symbol attribution.

| Symbol / group | Elf | Large |
| --- | ---: | ---: |
| `Spatial::radius` | 19.03% | 7.62% |
| `smoothing::smooth` | 18.39% | 11.51% |
| Unstable quicksort + small sort network | 19.87% | 10.88% |
| `coverage::rasterize` | 4.24% | 3.41% |
| Gaussian horizontal + vertical convolution | 3.04% | 6.08% |
| `image::imageops::sample::resize` | 1.29% | 4.72% |

The next optimization target is contour neighborhood selection and smoothing,
ahead of rewriting Lanczos or adding Gaussian intrinsics. `Spatial::radius`
allocates a result vector and sorts IDs on every query; `smooth` queries it
for every model and then rejects orientation/distance-incompatible candidates.
Investigate reusing query/observation buffers and filtering candidates before
sorting, while preserving accepted model order and exact numerical behavior.
The flat profile cannot assign all sorting samples to radius queries, so
validate attribution with a focused experiment before changing the algorithm.
No contour algorithm changes are part of this launcher change.

Validation: native release library tests passed (48 passed, four manual tests
ignored), both profiling benchmark runs passed, and launcher checks covered
shell syntax, native defaults, empty/custom/encoded flag overrides, execution
from a different directory, and argument forwarding. Native release server
tests also passed (nine tests).
