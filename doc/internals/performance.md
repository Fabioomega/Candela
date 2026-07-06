# Performance - Roadmap and Notes

This is a forward-looking document, not a punch list. It exists so that when you
do come back to performance work, the thinking from previous design discussions
isn't lost - and you can pick the items that actually matter for the workloads
you care about, rather than the ones that *look* like wins.

**The bias of this file:** features (CUDA, autograd) before micro-optimization,
and architectural wins before instruction-level wins. A 5% gain from hand-tuned
SIMD is interesting in isolation; it's irrelevant if the library can't compute
gradients.

---

## Where Candela is today

Two CPU backends - `CpuPure` (pure Rust, the default) and `CpuMkl` (Intel
MKL/VML for elementwise math, CBLAS for matmul, behind `--features mkl`) - each
generic over `f32` and `f64`. The graph fuses scalar chains via `FusedScalar`
and the planner reuses buffers, so memory pressure is already better than naive
eager evaluation.

What's *not* yet optimized:

- The contiguous `FusedScalar` path makes one full pass over the buffer per op
  in the chain - three ops means three trips through main memory.
- Fusion does not cross op-kind boundaries: `Add → Exp → AxBy` is two graph
  nodes (`Add` + `FusedScalar`), executed back-to-back with no shared tile.
- Everything is single-threaded. On large tensors that leaves 4–16× on the table
  from a `cargo add rayon` away.

The non-contiguous scalar path already does tiled fusion (see
[`compute_non_cont_scalar_op`](../src/tensor/backend/cpu_mkl/kernels.rs)) -
each packed chunk runs the entire op chain before the next chunk starts. That's
the model the contiguous path should grow into.

---

## The big wins, roughly in order

### 1. Parallelism

The cheapest large speedup available. `rayon::par_chunks_mut` on the output
buffer, with each thread running the existing kernel on its slice. Works for
elementwise ops trivially; reductions need a tree-style combine; matmul is
already parallelized inside MKL.

Realistic gain: 4–8× on a desktop CPU, near-linear up to physical core count for
memory-bound ops. The catch is that for small tensors the thread launch overhead
dominates - needs a size threshold. The `tracing` instrumentation already in
place will tell you where it kicks in.

### 2. Tiled fusion on the contiguous path

Window the contiguous buffer into L1-sized tiles (~1–4K f64 elements), run the
full `FusedScalar` chain per tile before moving on. Same structure as the
non-cont path, no packing copy needed. This collapses N memory passes into one
for any FusedScalar chain longer than a single op.

Realistic gain: 2–3× on chains of 3+ ops over buffers that exceed L3. Negligible
for tiny tensors that fit in L1 already.

### 3. Cross-kind fusion

The structural change: extend the fusion pass so that `Add`, `Mul`, `Sub`, `Div`
can absorb adjacent scalar chains, and emit a single tiled kernel that handles
both. The planner already has the dependency information; what's missing is the
op representation that can express "two strided inputs, run this pipeline, write
output."

This is where the design gets interesting. The cleanest framing is probably an
`OpKind::FusedPipeline` that holds a small bytecode of micro-ops (load, exp,
axby, mul, store) and a tiled kernel that interprets it. That's effectively
numexpr's design - a tiny interpreter beats not fusing at all by a wide margin.

Concepts worth studying before starting this:
[loop fusion](https://en.wikipedia.org/wiki/Loop_fission_and_fusion),
[loop tiling](https://en.wikipedia.org/wiki/Loop_nest_optimization), and
[polyhedral compilation](https://en.wikipedia.org/wiki/Polytope_model) (the
theory behind XLA and Polly). You don't need to implement polyhedral analysis;
you need to understand what XLA is doing so you know which corners to cut.

### 4. Vectorized scalar math

Only relevant once #3 lands - until then, MKL's `vdExp` running over a whole
buffer is your vectorized exp. With cross-kind fusion, the inner kernel needs
its own SIMD-friendly transcendentals. Options:

- **sleef** - C library, requires building. Most accurate option. `sleef-sys`
  crate exists.
- **Polynomial approximations** - 5th-order minimax exp, ~10 lines, vectorizes
  via `wide` (stable Rust SIMD) or `std::simd` (nightly). Gives ~1e-6 relative
  error, which is fine for ML.
- **libmvec** - glibc's vectorized math, Linux-only.

For ML workloads the polynomial path is the pragmatic choice. For a general
tensor library that promises IEEE-faithful results, sleef.

### 5. Dedicated activation ops

`Relu`, `Sigmoid`, `Tanh`, `Softmax` as their own `OpKind` variants. None of
these need `.where`. ReLU is `x.max(0.0)` (one `vmaxpd`); sigmoid as a fused
chain through #3 once it exists; softmax decomposes into `(x - max).exp() /
sum(exp)` which is three reductions you already have. Worth doing after #3 so
they reuse the pipeline machinery instead of growing custom paths.

---

## The micro stuff (defer)

These are the "5% gains" that are tempting but rarely move the bar:

- AVX-512 intrinsics for specific kernels
- Custom aligned allocators
- Prefetch hints
- `#[inline(always)]` tuning
- Branch-prediction hints

Each is a half-day rabbit hole. Worth it once profiling says a specific kernel
is the bottleneck and the obvious wins above are done. Not worth it speculatively.

One exception: **always check that LLVM is auto-vectorizing the loops you think
it is.** `cargo asm` or `cargo show-asm` will print the generated assembly for a
function; look for `vmulpd`, `vmaxpd`, `vfmadd` and friends. A `for el in slice
{ *el = el.max(0.0) }` loop *should* vectorize. If it doesn't, that's a real bug
worth fixing - not a 5% gain, more like a 4× regression.

---

## Concepts worth knowing

| Concept | Why it matters here |
|---------|---------------------|
| [Roofline model](https://en.wikipedia.org/wiki/Roofline_model) | Tells you whether a kernel is memory- or compute-bound, which decides whether fusion or vectorization is the right lever |
| Cache hierarchy (L1/L2/L3 sizes, line size) | Determines tile sizes and the "fits in cache" thresholds |
| [SIMD intrinsics and auto-vectorization](https://doc.rust-lang.org/std/arch/index.html) | The whole reason MKL is fast - understanding this clarifies what you're competing with |
| Kernel fusion / loop fusion | The theory behind #2 and #3 above |
| [Numerical stability for transcendentals](https://en.wikipedia.org/wiki/Numerical_stability) | Why `1/(1+exp(-x))` is fine but `log(1+exp(x))` isn't - Kahan summation, range reduction, log-sum-exp trick |
| Memory bandwidth ceilings (STREAM benchmark) | The actual hardware limit for memory-bound work - if you're at 80% of STREAM, fusion is the only path to more |
| BLAS conventions (column-major, leading dimension, batched GEMM) | Already in the codebase, but the *why* is worth understanding |

---

## Libraries worth reading

### Rust ecosystem

- **[Candle](https://github.com/huggingface/candle)** (Hugging Face) - Similar
  shape to Candela. Less aggressive fusion, multiple backends (CPU, CUDA, Metal).
  Best Rust-native reference for backend architecture.
- **[burn](https://github.com/tracel-ai/burn)** - DL framework with a clean
  backend trait. Read for how they abstract over CPU/CUDA/WGPU.
- **[dfdx](https://github.com/coreylowman/dfdx)** - Type-level shape checking
  plus autograd. Read when you start backprop.
- **[ndarray](https://github.com/rust-ndarray/ndarray)** - The Rust array
  library. Read for layout abstractions and broadcasting.
- **[tch-rs](https://github.com/LaurentMazare/tch-rs)** - libtorch bindings.
  Read for "what API surface do users actually expect."

### Beyond Rust

- **[numexpr](https://github.com/pydata/numexpr)** - Tiny fused-elementwise
  interpreter for NumPy. **Read this first** - it's the smallest serious
  implementation of the cross-kind fusion idea, ~5KLoC of C.
- **[oneDNN](https://github.com/oneapi-src/oneDNN)** - Intel's primitive
  library. Read for how to compose vectorized primitives well.
- **[Eigen](https://eigen.tuxfamily.org/)** - C++ expression templates. The
  template-based equivalent of your graph approach.
- **[XLA](https://github.com/openxla/xla)** - Google's compiler. The reference
  implementation of "kernel fusion at scale." Read HLO docs, not the source.
- **[TVM](https://github.com/apache/tvm)** - Compiler with autotuned tile
  schedules. Read for the schedule-vs-algorithm split.
- **[Triton](https://github.com/openai/triton)** - GPU tile programming model.
  Worth reading when CUDA is on the table; the tile abstraction is the right
  mental model for GPU kernels.
- **[Halide](https://halide-lang.org/)** - Image processing DSL that separates
  *what* to compute from *how* (tile/parallel/vectorize). The cleanest
  presentation of these ideas.

---

## Benchmarking

### Tools

- **[criterion](https://github.com/bheisler/criterion.rs)** - Rust's de-facto
  microbenchmark harness. Statistical, regression-aware, plots.
- **[divan](https://github.com/nvzqz/divan)** - Newer alternative, nicer API,
  per-iteration timers. Either is fine; criterion has more ecosystem.
- **[cargo flamegraph](https://github.com/flamegraph-rs/flamegraph)** -
  Function-level profiling. First thing to run when something is slow.
- **Intel VTune** - Free for personal use. Gives cache miss data, vectorization
  reports, memory bandwidth utilization. Indispensable for the kind of work
  this document is about.
- **`cargo asm` / `cargo show-asm`** - Inspect generated assembly. Confirms
  whether LLVM vectorized a loop.

### Workloads to benchmark

Pick a set and stick with it - a perf regression suite is only useful if it
runs the same thing every time. A reasonable starting set:

| Workload | Why |
|---|---|
| `Add(a, b)` at sizes spanning L1 → L2 → L3 → DRAM | Shows the memory-bandwidth ceiling and where it hits |
| `(a + 1.0) * 2.0 - 3.0` (3-op scalar chain) | Measures `FusedScalar` quality |
| `1 / (1 + (-x).exp())` (sigmoid) | Measures transcendental + fusion combined |
| `softmax` along last axis | Composite op people actually care about |
| Matmul at M=N=K = 128, 512, 2048 | Sanity check vs MKL baseline |
| Strided slice + op | Stresses the non-contiguous path |

### What to compare against

- **MKL direct** - a hand-written C program calling `vdExp` etc. on the same
  data. This is your ceiling on the CPU side; if you're within 10–20% you're
  doing well.
- **NumPy** - easy to write a Python script that runs the same workload. Cross-
  language comparison, but tells you whether your abstraction tax is reasonable.
- **Candle** - closest peer. Useful for "does my architecture have a
  fundamental flaw vs another Rust library."
- **PyTorch CPU** - overkill for most ops, but the reference for what "fast"
  means in practice.

### Pitfalls to avoid

- Benchmarking only one tensor size. Performance changes character across cache
  levels - the answer at 1KB is different from the answer at 1GB.
- Forgetting to warm up. MKL's first call initializes dispatch tables; criterion
  handles this for you, but ad-hoc benchmarks don't.
- Measuring with `cargo run` instead of `cargo run --release`. Yes, people do
  this. The numbers are off by 50–100×.
- Comparing against unoptimized NumPy (no MKL). Make sure both sides are using
  the same BLAS or the comparison is meaningless.
- Ignoring variance. A 3% improvement inside ±5% noise is not an improvement.

---

## Order of operations, opinionated

1. ~~Finish the backend separation.~~ Done (ROADMAP Phase 7).
2. Build the benchmark suite. This moved ahead of rayon deliberately: everything
   below it is a performance claim, and an unmeasurable claim is a liability.
3. Add `rayon` for elementwise + reductions, threshold-gated. Easy and large.
4. Tile the contiguous `FusedScalar` path. Mechanical, follows the existing
   non-cont pattern.
5. Cross-kind fusion (the `FusedPipeline` design). This is the architecturally
   interesting one, and where you'll learn the most. It is the same work item as
   ROADMAP Phase 9's `FusedElementwise` (v0.4) - one design, don't build it twice.
6. Vectorized scalar math, if (5) shows transcendentals are still the bottleneck.
7. Micro-optimizations, only with profiling evidence.

Items (2)–(4) are the v0.3 arc (plus the per-skeleton buffer pool - see the
ROADMAP's Release Plan); item (5) is v0.4 on its own. Resist the temptation to do
(5) or (7) first. The leverage isn't there yet, and the work is much easier to
evaluate once the measurement infrastructure is in place.
