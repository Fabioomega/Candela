# Candela - Roadmap

Phases are ordered by dependency: complete earlier phases before starting later ones.
Versions group phases into releases - see the Release Plan below.

Status markers: `[ ]` not started · `[~]` in progress · `[x]` done

---

## Release Plan

Phases answer "what must exist before what." Versions answer "what ships together,
and what's the story." Every release should have a pitch that fits in one line.

| Version | Story | Contents |
|---------|-------|----------|
| 0.1 | A correct lazy tensor engine | Phases 1–7 (shipped) |
| 0.2 | Skeletons: compile once, run many, know your costs | Phase 8 + slot ergonomics + `memory_report` + skeleton cache (details below) |
| 0.3 | Measurement, and the speedups it unlocks | Benchmark suite, rayon, tiled contiguous fusion, skeleton memory arena, external output buffers (details below) |
| 0.4 | The fusion rewrite | Phase 9 |
| 0.5 | Runs a real model on CPU | Phases 10 + 11 (blocks + safetensors I/O) |
| 0.6 | CUDA | Phase 12 |
| 0.7 | fp16 / bf16 | Phase 13 |
| 0.8 | Gradients | Phase 14 (gradients + Stitch multi-output) |
| 0.9 | ONNX import | Phase 15 |
| 1.0 | Stability promise | No new features - the API stops moving |

### Versioning policy

Pre-1.0, Cargo treats `0.x → 0.(x+1)` as breaking and `0.x.y → 0.x.(y+1)` as
compatible. So: a **minor bump is a release arc** and is allowed to break the API;
a **patch bump is fixes and small additive features** between arcs. Don't agonize
over whether a feature is "major" - before 1.0 there's no such thing, and each
release's notes simply say whether it broke anything. CUDA possibly changing the
`Backend` trait is a normal 0.x bump like every other arc.

A feature that no phase mentions ships with the version whose *story* it serves:
benchmarks and rayon belong to 0.3 because that arc is about measurement; tensor
saving belongs to 0.5 because that arc is about real models. A feature that serves
no planned story waits, or goes out as a patch if it's small and additive.

`1.0` is a promise, not a feature: it lands after a full arc completes without
needing an API break. ONNX import is purely additive, so if the API is stable by
0.8, 1.0 can land before the importer rather than after - the table shows the
conservative order.

### v0.2 in detail (the current arc)

Beyond finishing Phase 8 (tests, rustdoc - see the phase):

- **Slot ergonomics.** `SkeletonSlot::from_shape(&[4])` locked to `DefaultBackend`
  (mirroring the tensor constructors) so `T` infers from use; a
  `B = DefaultBackend` type-parameter default so the type is writable as
  `SkeletonSlot<f64>`; settle the `into_skeleton` naming before the API freezes
  (`into_` conventionally consumes, but it borrows today).
- **`Skeleton::memory_report()`.** Peak bytes, allocation count, buffer sizes -
  read straight off the compiled plan, which already knows all three. Turns the
  README's "Candela can tell you the resources before it runs" into present tense.
- **Skeleton cache.** Dynamic shapes by keyed recompilation, built entirely on the
  public API as both a usable default and a reference for custom versions. Key is
  the tuple of *all* slot layouts (everything `run` validates), a builder closure
  fires on miss, and eviction is a small `EvictionPolicy` trait - LRU and
  unbounded ship; anything fancier is a user impl.

Memory *pooling* is deliberately not here: it's a performance claim, and 0.2 has
no benchmarks to back it. It moves to 0.3.

### v0.3 in detail

`doc/performance.md`'s order of operations, with the benchmark suite promoted to
first - nothing after it is provable without it:

1. Benchmark suite (`benches/`, criterion; workload set from `doc/performance.md`)
2. Rayon parallelism for elementwise + reductions, threshold-gated
3. Tiled fusion on the contiguous `FusedScalar` path
4. Skeleton memory arena (the pool deferred from 0.2, redesigned): plans pack
   their intermediates into one slab at compile time; slab *retention* is the
   opt-in part, per the allocation philosophy
5. External output buffers (`run_into`): the caller supplies the destination memory
   the root result is written into - the dual of the input-slot mechanism, and the
   same subsystem as the arena above seen from the user's end (the arena is the
   skeleton bringing reusable memory; `run_into` is the caller bringing it)

Each item lands with the benchmark that justifies it. The fusion *rewrite* is not
in this arc - it's the architecturally risky one and gets 0.4 to itself.

**The arena in detail.** `into_skeleton` always packs: greedy-by-size offset
assignment (Pisarchyk & Lee, *Efficient Memory Management for Deep Neural Network
Inference*) over the pre-planner's raw buffer lifetimes - bypassing the same-size
slot-reuse pass, which packing subsumes, and merging in-place chains and reference
lifetime extensions into single intervals. Packing computes offsets only and
retains no memory, so it is unconditional; `materialize()` keeps the current
allocation path untouched.

- **Escape rule.** Buffers with no reclaim point (the root result, cache fills -
  later, Stitch's outputs) are owned allocations, never slab regions. A region
  never leaves the executor; user-facing tensors are always owned.
- **Provenance.** `Storage` splits into owned (`Arc<Vec<T>>`) and region
  (slab + offset). The slab is raw-pointer-only inside, so disjoint reads plus
  the single writer never form aliasing `&`/`&mut`. Prerequisite refactor, its
  own PR, landing first: kernels take `&mut [T]` instead of owning `Vec<T>`,
  and the executor assembles outputs - `bind` needs the same split.
- **Slabs.** A skeleton stores required bytes plus offsets and owns no memory;
  slabs come from an `Arc<SlabPool>` - private by default, shareable by
  constructor. Default retention is zero: check out, run, free. Two independent
  knobs: `max_retained` (excess slabs freed silently) and a byte budget (hard
  cap; `run` errors). `DynamicSkeleton` owns one pool across all shape variants -
  slabs are fungible whenever capacity suffices, so one high-water slab serves
  every entry, and evicting an entry frees only its (tiny) plan.
- **Verification.** An always-on validator asserts no two regions overlap in
  both lifetime and address; `memory_report` gains `required_slab_bytes`, pool
  state, and the ratio of slab size to the live-bytes lower bound; a debug
  feature poisons the slab with NaN each run to enforce that kernels tolerate
  dirty output buffers.
- **Alignment.** Regions align to cache lines (parameterized; 256 B once CUDA
  needs it) so rayon-parallel kernels don't false-share at region boundaries.
- **Sequential execution is load-bearing.** Plan order = execution order
  underwrites the interval packing, the single-writer safety story, and the
  future CUDA mapping (one stream, stream order = plan order, so the arena
  transfers with zero synchronization). No inter-op threading, on CPU or via
  multiple streams.

**External buffers in detail.** `run_into` on a skeleton (and `materialize_into`
on a promise) moves a caller-owned buffer in, writes the root result into it, and
hands it back as an owned tensor. Ownership is the safety story: the buffer is
opaque while bound and returned only once the write is done, so there's no lock
and no aliasing window. Length is validated against the root layout; a pure-alias
root inserts the copy it implies. `Tensor::try_into_vec` recovers the buffer for
the next iteration, so with slab retention on, a steady-state `run_into` loop
performs zero allocations: inputs are the caller's tensors, intermediates live in
the slab, the output lands in the caller's buffer. `memory_report` grows an
internal-vs-external split: what the skeleton allocates, and what memory it
expects you to bring. Benchmark-gated like the rest of this arc.

Multi-output - several results or intermediates from one compiled run - is
deliberately *not* this feature: it lands with Stitch (Phase 14) as multi-root
skeletons. `into_skeleton` accepts several roots; outputs return positionally in
declaration order, exactly as slots bind today; every root is an owned output the
packer excludes. No per-node bind markers and no minted handles - a root is
observable by definition, so the fusion barrier falls out rather than being
policed. Multi-root skeletons are terminal-only: `compose` returns an error, and
the fix is to build the larger graph and compile *that*. Stitch degrades when
embedded because unplanned dead roots are elided for free; a compiled multi-root
plan replays whole, so degrading would silently compute and discard - it errors
instead. `CachedPromise` stays the eager-world way to share an intermediate
across materializations; a multi-root skeleton is its compiled-world dual,
sharing the subgraph within each run. `run_into` extends positionally to N
buffers when this lands.

---

## Phase 1 - Execution Planner (Done)

**Goal:** Replace the current ad-hoc buffer-reuse mechanism with a correct static
execution plan derived from the computation graph.

**Why first:** The existing `reusable: bool` approach produces wrong results for
non-commutative ops (`Sub`, `Div`) and does not scale. Everything that comes after
depends on execution being correct. Fix this before building on top of it.

**Known bugs to fix:**
- The `reusable` mechanism can pick the wrong buffer as the output of `Sub` and `Div`,
  silently reversing the operands. The result of `a - b` can come out as `b - a`.
- `Sub` in the scalar op path was copy-pasted from another arm and computes the wrong
  operation entirely.

### Tests
```rust
// Sub correctness (was silently computing addition before)
let t = ones(&[4]);
assert_eq!((t - 3.0).materialize().data(), &[-2.0; 4]);

// Div correctness
let t = Tensor::from_scalar(6.0, &[4]);
assert_eq!((t / 2.0).materialize().data(), &[3.0; 4]);

// Sub is non-commutative: (a - b) != (b - a)
let a = arange(4);   // [0, 1, 2, 3]
let b = ones(&[4]);
assert_ne!((a - b).materialize().data(), (b - a).materialize().data());

// Buffer reuse does not corrupt results in a longer chain
let t = arange(6);
let result = ((t * 2.0) - 1.0).materialize();
// expected: [-1, 1, 3, 5, 7, 9]
assert_eq!(result.data(), &[-1.0, 1.0, 3.0, 5.0, 7.0, 9.0]);

// A node used by two consumers is computed once and produces the same value for both
let t   = arange(4).as_promise();
let lhs = &t * 2.0;
let rhs = &t + 1.0;
let result = (lhs - rhs).materialize(); // (2x) - (x+1) = x - 1
assert_eq!(result.data(), &[-1.0, 0.0, 1.0, 2.0]);
```

### Documentation (Done)
- Document the execution plan struct and the liveness analysis algorithm
- Add a doc comment to `TensorGraphNode::compute` explaining the execution model

---

## Phase 2 - Cleanup and Test Foundation (Done)

**Why here:** Small, high-leverage housekeeping while the Phase 1 architecture is fresh.
The test suite must exist before Phase 3 so regressions are caught as they are introduced,
not discovered later.

### Implementation
- [X] Delete `src/tensor/tensor_old.rs` - dead code, never referenced
- [X] Remove unused `lapacke` dependency from `Cargo.toml`
- [X] Mark internal modules `pub(crate)`: `OpKind`, `NodeKind`, `Promising`,
      `TensorGraphEdge`, `TensorGraphNode`, `TensorGraphCacheNode`
- [X] Fix `Clone` semantics: make `Clone` a shallow Arc-bump; rename the current deep
      copy to `deep_clone()`. A `Vec<Tensor<T>>::clone()` should not silently copy every
      buffer
- [X] Replace remaining `unreachable!()` / `todo!()` in `graph.rs`, `impl_layout.rs`,
      and `impl_compute` with `Err(OpError::...)` or `unimplemented!()` with a clear
      message
- [X] Fix the three bugs in `Layout::broadcast_to_shape` before Phase 4 can use it:
      the `cfg_debug_only!` guard predicate is inverted; stride is built from `shape`
      instead of `self.stride`; `len` is computed from `self.shape` instead of the
      target shape
- [X] Set up `tests/` directory with an integration test module
- [X] Add a GitHub Actions workflow: `cargo fmt --check`, `cargo clippy -- -D warnings`,
      `cargo test`

### Tests
```rust
// Clone is shallow - both tensors share the same Arc, no allocation
let a = arange(1000);
let b = a.clone();
assert!(Arc::ptr_eq(&a.storage().buffer, &b.storage().buffer));

// deep_clone allocates a fresh buffer
let c = a.deep_clone();
assert!(!Arc::ptr_eq(&a.storage().buffer, &c.storage().buffer));
assert_eq!(a.data(), c.data());

// broadcast_to_shape produces correct strides
let layout = Layout::from_shape(&[4], 0);
let broadcast = layout.broadcast_to_shape(&[3, 4]).unwrap();
assert_eq!(broadcast.shape(), &[3, 4]);
assert_eq!(broadcast.stride()[0], 0); // broadcast dimension
assert_eq!(broadcast.len(), 12);      // total elements in broadcasted shape

// One test per OpKindScalar arm (catches copy-paste regressions)
let t = Tensor::from_scalar(3.0, &[4]);
assert_eq!((t + 2.0).materialize().data(), &[5.0; 4]);
assert_eq!((t - 2.0).materialize().data(), &[1.0; 4]);
assert_eq!((t * 2.0).materialize().data(), &[6.0; 4]);
assert_eq!((t / 2.0).materialize().data(), &[1.5; 4]);
```

### Documentation
- Add `# Examples` to the three public types: `Tensor`, `TensorPromise`,
  `CachedTensorPromise`
- Create `examples/lazy_eval.rs`: a walkthrough of building a graph and materializing it
- Create `examples/fusion.rs`: a demonstration of scalar fusion collapsing 20 additions

---

## Phase 3 - Complete Matmul (Done)

**Goal:** Make matrix multiplication actually compute the correct result.

**Why here:** Matmul is the single most important missing operation. Every model building
block in Phase 8 depends on it. Currently the implementation allocates a zeroed output
and returns it without computing anything.

**Known bugs to fix:**
- The `inputs.pop()` order is wrong: `raw_a` receives the second input (rhs) and `raw_b`
  receives the first (lhs). The labels are swapped, so any non-symmetric matmul will
  produce the wrong answer.

### Tests
```rust
// Basic 2x3 @ 3x4
let a = arange_2d(2, 3);         // [[0,1,2],[3,4,5]]
let b = Tensor::eye(3, 4);       // identity-ish
let c = a.matmul(&b)?.materialize();
assert_eq!(c.shape(), &[2, 4]);

// identity @ identity = identity
let i = Tensor::eye(3, 3);
let result = i.matmul(&i)?.materialize();
assert_approx_eq!(result.data(), Tensor::eye(3,3).data());

// Transposed matmul: A @ B^T where B is already transposed
let a = arange_2d(2, 3);
let b = arange_2d(4, 3).as_promise().transpose();
let c = a.matmul(&b)?.materialize();
assert_eq!(c.shape(), &[2, 4]);

// Batched: [2,3,4] @ [2,4,5] = [2,3,5]
let a = Tensor::from_scalar(1.0, &[2, 3, 4]);
let b = Tensor::from_scalar(1.0, &[2, 4, 5]);
assert_eq!(a.matmul(&b)?.materialize().shape(), &[2, 3, 5]);

// Shape mismatch returns error, does not panic
assert!(a_3x4.matmul(&b_3x4).is_err());
```

### Documentation
- Document the `Matmul` op in `def_op.rs` with the shape conventions used
- Add a `matmul` example in `examples/matmul.rs`

---

## Phase 4 - Broadcasting (Done)

**Goal:** Allow ops to work on tensors with compatible but non-identical shapes by
expanding dimensions implicitly rather than erroring.

**Why here:** Required for almost all real ops and for the batched matmul broadcast case.
The `broadcast_to_shape` bugs must be fixed in Phase 2 before this phase can proceed.

### Tests
```rust
// Column vector broadcast against row: [3,1] + [1,4] = [3,4]
let col = Tensor::from_scalar(1.0, &[3, 1]);
let row = Tensor::from_scalar(2.0, &[1, 4]);
let result = (col + row).materialize();
assert_eq!(result.shape(), &[3, 4]);
assert_eq!(result.data(), &[3.0; 12]);

// 1D against 2D: [4] + [3,4] = [3,4]
let v = arange(4);         // [0,1,2,3]
let m = ones(&[3, 4]);
let result = (v + m).materialize();
assert_eq!(result.shape(), &[3, 4]);
// each row of result should be [1,2,3,4]

// Scalar broadcast: [1] against [3,3]
let s = Tensor::from_scalar(5.0, &[1]);
let m = ones(&[3, 3]);
let result = (s * m).materialize();
assert_eq!(result.data(), &[5.0; 9]);

// Incompatible shapes still error
assert!((ones(&[3, 4]) + ones(&[2, 4])).materialize().is_err());
```

### Documentation
- Add broadcasting semantics to the Memory Layout section of `README.md`
- Document stride-0 broadcast in `Layout::broadcast_to_shape`

---

## Phase 5 - Reduction Ops (Done)

**Goal:** Add `sum`, `mean`, and `max` reductions along a specified axis, with optional
`keepdim` support.

**Why here:** Required for any loss function, normalization layer, or metric. Reductions
are also needed to implement `Softmax` in Phase 8.

### Tests
```rust
// 1D sum
let t = arange(5);  // [0,1,2,3,4]
assert_eq!(t.sum(0, false).materialize().data(), &[10.0]);

// 2D sum along axis 0
let t = Tensor::from_data(&[[1.0,2.0],[3.0,4.0]], &[2,2]);
assert_eq!(t.sum(0, false).materialize().data(), &[4.0, 6.0]);

// 2D sum along axis 1
assert_eq!(t.sum(1, false).materialize().data(), &[3.0, 7.0]);

// keepdim preserves rank
let result = t.sum(0, true).materialize();
assert_eq!(result.shape(), &[1, 2]);

// mean
let t = Tensor::from_scalar(4.0, &[3, 3]);
assert_eq!(t.mean(0, false).materialize().data(), &[4.0; 3]);

// max along axis
let t = Tensor::from_data(&[1.0, 5.0, 2.0, 4.0, 3.0], &[5]);
assert_eq!(t.max(0, false).materialize().data(), &[5.0]);
```

### Documentation
- Add `reduce` section to `README.md` features list
- Add doc examples to each reduction method

---

## Phase 6 - f32 Support and Activation Ops (Done, except `sigmoid`)

**Goal:** Support `f32` tensors in addition to `f64`, and add `relu`, `sigmoid`, and
`tanh` as first-class ops.

**Why here:** These are two independent, straightforward additions that together greatly
expand usability. Being scalar ops, activations automatically participate in fusion
without any new fusion rules.

**Status:** f32 support is in. `relu` and `tanh` landed (see
`examples/activations.rs`). `sigmoid` is deferred: it decomposes into a fused
chain once cross-kind fusion exists (`doc/performance.md`, item 5), so landing it
now would mean a custom path that gets rewritten anyway.

### Tests
```rust
// f32 basic ops
let t: Tensor<f32> = Tensor::from_scalar(2.0f32, &[4]);
assert_eq!((t * 3.0f32).materialize().data(), &[6.0f32; 4]);

// ReLU
let t = Tensor::from_data(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
assert_eq!(t.relu().materialize().data(), &[0.0, 0.0, 0.0, 1.0, 2.0]);

// Sigmoid output is in (0, 1)
let t = Tensor::from_data(&[-10.0, 0.0, 10.0], &[3]);
let s = t.sigmoid().materialize();
assert!(s.data().iter().all(|&x| x > 0.0 && x < 1.0));
assert_approx_eq!(s.data()[1], 0.5); // sigmoid(0) = 0.5

// Activation fuses with preceding scalar op
let t = arange(4).as_promise();
let p = (t * 2.0).relu();  // should be one fused node
let result = p.materialize();
assert_eq!(result.data(), &[0.0, 2.0, 4.0, 6.0]);
```

### Documentation
- Add f32 examples to `README.md`
- Create `examples/activations.rs`

---

## Phase 7 - Backend / Dtype Split and CI (Done)

**Goal:** Decouple the compute device from the element type, gate Intel MKL behind a
feature flag with a pure-Rust fallback, and stand up the CI that everything afterwards
will rely on.

**Why moved forward:** every later phase is shaped by whether the backend abstraction
exists. Today the project does not compile without an Intel toolchain, which gates
contributors and rules out non-x86 targets entirely. Doing this before autodiff or
skeletons also means those phases get to design against a clean trait rather than
retrofit one.

### Implementation

- Split `ComputeWrapperSpec` into a `Backend` trait (compute device - `CpuMkl`,
  `CpuNaive` to start) and a `Dtype` trait (element type - `f32`, `f64`). The pair
  `<B: Backend, T: Dtype>` selects a kernel set.
- Put MKL behind `--features mkl`. Default build uses the pure-Rust path:
  `matrixmultiply` (or a hand-rolled blocked kernel) for `MatMul`, straightforward
  loops for elementwise.
- Replace the `todo!()` arms in `cpu_f64.rs` and `cpu_f32.rs` with
  `OpError::UnsupportedInplace(op_name)` so an unsupported planner decision returns
  an error rather than panicking.
- Add `# Safety` doc comments to every `pub unsafe fn`, especially `iter_as_layout`.
- Remove the gratuitous `unsafe { layout.unwrap_unchecked() }` blocks in `graph.rs`
  and `impl_op.rs` - the optimiser already eliminates the dead panic branches.
- Decide and document: do `add/sub/mul/div` method variants return `Result` (per
  the README's "methods return Result" rule) or panic (per the current code)? Make
  one of the two true.
- Stand up `.github/workflows/ci.yml` running `cargo fmt --check`,
  `cargo clippy -- -D warnings`, and `cargo test` on the MKL-off build. Add an
  `--features mkl` job once an MKL-enabled runner is feasible.

### Tests
```rust
// Same op runs under either backend with identical results.
let t = Tensor::<CpuNaive, f32>::from_scalar(2.0, &[4]);
assert_eq!((t * 3.0).materialize().data(), &[6.0; 4]);

// Default build (no MKL) compiles and runs the full test suite.
// `cargo test` on a non-Intel machine is green.
```

### Documentation
- Add a "Backends" section to `README.md` covering the feature flag and what
  `mkl` vs default builds enable.
- Create `doc/backends.md` describing the `Backend` trait and what it takes to
  add a new one.

---

## Phase 8 - Skeletons (v0.2) (Done)

**Goal:** Pre-compile the execution plan once for a fixed graph topology and reuse it
across iterations with different data, avoiding the cost of re-planning on every call.

**Why moved forward:** every benchmark right now measures planner overhead + compute.
Skeletons let benchmarks measure compute alone - necessary for honest numbers before
autodiff piles more nodes onto every graph. Skeletons also unlock real production
patterns (preprocessing pipelines, repeated inference) that don't require autograd.

**What shipped** (the API grew past the original `PromiseSkeleton` sketch - see
`doc/skeleton.md` for the full story):

- `SkeletonSlot` - a typed hole in the graph: a `Layout` with no data behind it.
  Built from a `Layout` directly or borrowed from an existing tensor/promise via
  `.as_slot()`.
- `SkeletonPromise` - what any op chain containing a slot becomes. Enforced at
  the type level (the "taint algebra"): an expression with a slot anywhere in
  its lineage has no `.materialize()`; the only exit is `into_skeleton`.
- `Skeleton` - the compiled template: an owned plan plus the declared slot list.
  `run(&[Tensor])` executes it; slots bind by identity, in declaration order.
- `BakedPromise` - a skeleton bound over concrete inputs via `Skeleton::compose`,
  embedding the compiled plan as a single `Baked` node inside a larger graph.

**Still open:** caching the fusion-pass result and amortising buffer allocation via
a per-skeleton memory pool. Right now the *plan* is reused across `run` calls but
the buffers are re-allocated each time. Plan introspection (peak memory, allocation
count - the plan already knows both) would make the README's "Candela can tell you
the resources before it runs" promise real, nearly for free.

### Tests (to do)
- [X] Every error path: `IncorrectSlotAmount`, `NotSameSlot`, `NotSameLayoutAtSlot`
- [X] Multi-slot expressions; the same slot used twice in one expression
- [X] Re-running one skeleton with different data reuses the plan and matches
      `.materialize()` on the equivalent graph
- [X] A skeleton containing matmul and a reduction, not just scalar chains
- [X] `compose` + `BakedPromise::as_promise().materialize()` round-trip
- [X] Layout strictness: a same-shape but transposed input is rejected at `run`

```rust
// Skeleton and materialize produce identical results (real API)
let slot = SkeletonSlot::<f64, _>::new(Layout::from_shape(&[4], 0));
let skeleton = (&slot * 2.0 + 1.0).into_skeleton(std::slice::from_ref(&slot))?;

let input = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4]);
let from_skeleton    = skeleton.run(std::slice::from_ref(&input))?;
let from_materialize = (input.as_promise() * 2.0 + 1.0).materialize();
assert_eq!(from_skeleton.data(), from_materialize.data());

// Re-running with different data uses the same plan
let input2  = Tensor::from_scalar(5.0, &[4]);
let result2 = skeleton.run(std::slice::from_ref(&input2))?;
assert_eq!(result2.data(), &[11.0; 4]);
```

### Documentation
- [x] Create `doc/skeleton.md` - the slot/promise/skeleton/baked taxonomy and the
      taint algebra
- [x] Add skeletons to `README.md` with an example
- [X] Re-export the skeleton types from the crate root and add them to the
      `lib.rs` type map
- [X] Rustdoc with `# Examples` / `# Errors` on `into_skeleton`, `run`, `compose`

---

## Phase 9 - Fusion Rewrite Pass and FusedElementwise (v0.4)

**Goal:** Move fusion out of `TensorGraphNode::new` into a dedicated rewrite pass over
the DAG, and introduce a multi-input `FusedElementwise` op so trees like
`(x + 1) * (y - 2)` collapse into a single pass.

**Before starting:** this phase and `doc/performance.md`'s `FusedPipeline`
(cross-kind fusion) are one design with two names - an expression-tree interpreter
vs. a micro-op bytecode interpreter for the same multi-input fused kernel. Unify
them into a single design before building either; the bytecode framing is likely
the keeper (flat, cache-friendly to interpret, and closest to numexpr, the
reference worth reading first).

**Why here:** the current `try_fuse` only folds a new op into one of its parents,
operates only on linear chains over a single tensor input, and runs greedily at
construction time. That covered the early ops, but it does not generalize:

- Multi-tensor elementwise expressions cannot fuse at all (no representation for
  "kernel over N tensor inputs").
- Algebraic identities like `x * 0 → 0`, `x + 0 → x`, `transpose(transpose(x)) → x`
  are not applied - and autodiff (Phase 14) will generate huge graphs full of
  exactly those patterns.
- Adding a new fusion means editing one big `match` in `compute_fusion` instead of
  defining a rule independently.

### Implementation

**The rewrite framework:**
```rust
trait Rewrite<T: Copy> {
    fn try_apply(&self, node: &NodeKind<T>) -> Option<NodeKind<T>>;
}
```
A pass walks the DAG bottom-up, applies each rule, and iterates to fixpoint. The
pass runs at plan time (cached inside `PromiseSkeleton` so it pays only once).
`TensorGraphNode::new` becomes pure construction; the fusion work moves out.

**Rules to land:**
- `ScalarChainFusion` - what `fusion.rs` does today, expressed as one rule.
- `MatmulFusion` - `MatMul + ScalarOp`, `MatMul + Add`, etc.
- `AlgebraicIdentities` - `x*0`, `x+0`, `x*1`, `x-x`, `transpose(transpose(x))`,
  `view(view(x))`, broadcast-then-reduce-same-axis, etc.
- `AsContiguousDropContig` - drop `AsContiguous` when its input is already
  contiguous (currently in `compute_fusion`).
- `ElementwiseTreeBuilder` - see below.

**FusedElementwise op:**
```rust
enum ElemExpr<T: Copy> {
    Input(usize),                        // refers to inputs[i]
    Const(T),
    AxBy(T, T, Box<ElemExpr<T>>),
    Add(Box<ElemExpr<T>>, Box<ElemExpr<T>>),
    Sub(Box<ElemExpr<T>>, Box<ElemExpr<T>>),
    Mul(Box<ElemExpr<T>>, Box<ElemExpr<T>>),
    Div(Box<ElemExpr<T>>, Box<ElemExpr<T>>),
    Exp(Box<ElemExpr<T>>),
    Ln(Box<ElemExpr<T>>),
    // activations from Phase 6
}

OpKind::FusedElementwise { expr: ElemExpr<T>, n_inputs: usize }
```
The kernel is a per-chunk tree interpreter. The single-input chain stays on the
BLAS fast path (`FusedScalar` remains as a special case); only the multi-input
cases use the interpreter. The threshold for fusion: rewrite if it reduces total
memory traffic (saving one intermediate buffer write+read is almost always a win).

### Tests
```rust
// Two-input elementwise expression collapses to one kernel pass.
let x = arange(4).as_promise();
let y = arange(4).as_promise();
let result = ((&x + 1.0) * (&y - 2.0)).materialize();
// Verify the graph has exactly one FusedElementwise node before execution.

// Algebraic identity: x * 0 produces zeros without computing x.
let x = (arange(4) + 100.0).as_promise();  // expensive-looking subgraph
let zero = Tensor::from_scalar(0.0, &[4]);
let result = (&x * zero).materialize();
assert_eq!(result.data(), &[0.0; 4]);

// transpose(transpose(x)) folds away.
let t = arange_2d(3, 4);
let result = t.transpose().transpose().materialize();
// Plan should contain zero Transpose ops.
```

### Documentation
- Add `doc/fusion.md` describing the rewrite framework and the rule set
- Document each `Rewrite` impl with the pattern it matches and the rationale

---

## Phase 10 - Model Building Blocks (v0.5)

**Goal:** Implement `Linear`, `Softmax`, and `LayerNorm` by composing the primitives
from Phases 3–6.

**Why here:** by this point matmul, broadcasting, reductions, activations, and the
fusion rewriter are all in. Linear and Softmax become pure composition - and the
fusion pass folds the resulting graphs much more aggressively than it could before
Phase 9. None of these need gradients: they are forward-only building blocks, which
is exactly what the inference phases after this one consume.

**Also do here:** consolidate the macro explosion in `ops/impl_op.rs`. The
`ComputationDef` trait already exists; the four `impl Add/Sub/Mul/Div` macros over
`Tensor × Promise × CachedPromise × {ref, owned}` should collapse into one blanket
impl per operator. The macro layer becomes optional sugar instead of load-bearing.

### Tests
```rust
// Linear: [batch=2, in=3] @ [out=4, in=3]^T + [4] = [2, 4]
let x      = Tensor::from_scalar(1.0, &[2, 3]);
let weight = Tensor::from_scalar(1.0, &[4, 3]);
let bias   = Tensor::from_scalar(0.5, &[4]);
let out    = linear(x, weight, bias).materialize();
assert_eq!(out.shape(), &[2, 4]);
assert_approx_eq!(out.data(), &[3.5; 8]);

// Softmax output sums to 1 along last axis
let logits = Tensor::from_data(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
let probs  = softmax(&logits, 1).materialize();
let row0_sum: f64 = probs.data()[..3].iter().sum();
assert_approx_eq!(row0_sum, 1.0);
```

### Documentation
- Create `examples/linear_layer.rs` showing a single forward pass
- Document each function with shape conventions

---

## Phase 11 - Safetensors I/O (v0.5)

**Goal:** Load real model weights from `.safetensors` files, so a forward pass
hand-written from Phase 10 blocks runs an actual trained model. Saving comes along
for the ride: serializing a `Tensor` is a few lines once the parser exists, and the
round-trip tests want it anyway.

**Why here:** the gap between "interesting engine" and "runs a real model" is not
more ops - it's weights. [safetensors](https://github.com/huggingface/safetensors)
is a deliberately boring format (a JSON header plus raw buffers), so the loader
costs days, not the months an ONNX importer would. Combined with skeletons, this
is the demo that matters: compile a real model once, run it on new inputs with
zero planning overhead.

### Implementation
- Parse with the `safetensors` crate (or by hand - the format fits on a page).
- Map dtypes: `F32`/`F64` now; `F16`/`BF16` return a clear error until Phase 13.
- Load into owned `TensorData` buffers first; an mmap-backed zero-copy path can
  come later if loading ever shows up in a profile.
- Create `examples/mlp_inference.rs`: a small MLP trained in PyTorch, exported to
  safetensors, forward pass built from Phase 10 blocks.

### Tests
```rust
// Tensors written by the reference implementation load with correct shapes
let weights = load_safetensors("tests/data/mlp.safetensors")?;
assert_eq!(weights["fc1.weight"].shape(), &[16, 4]);

// Forward pass matches the PyTorch reference output within tolerance
let out = mlp_forward(&weights, &input).materialize();
assert_approx_eq!(out.data(), expected_from_pytorch);
```

### Documentation
- Document the dtype mapping and the error on unsupported dtypes
- Keep the PyTorch export script next to the test data, with a note on how to
  regenerate it

---

## Phase 12 - CUDA Backend (v0.6)

**Goal:** A `Cuda` backend implementing the `Backend` trait: each `OpKind` mapped to
a kernel, async execution on CUDA streams.

**Why moved up (it used to sit after autodiff):** Candela's ambitions are
inference-shaped - skeletons, predictable resources, real weights from Phase 11 -
and inference without a GPU stops being interesting at exactly the model sizes
people care about. By this point the `Backend` trait exists, the fusion rewriter
exists, and a real model runs on CPU, so every CUDA kernel has a reference
implementation to test against. Autodiff is deliberately *not* in yet: gradients
are just more graph, so landing the GPU first means the backward pass runs on it
from day one instead of being retrofitted.

CUDA is the largest phase in absolute terms but the smallest in architectural
surprise - the design points are already pinned down by the trait.

**Design question inherited from Phase 7:** `Backend::compute` is synchronous today.
Decide whether the CUDA path hides async behind it (record on a stream, synchronize
on read) or whether the trait grows an explicit async variant. Hiding it keeps the
CPU backends untouched; surfacing it makes overlap and multi-GPU explicit later.
Decide before writing the second kernel, not after the twentieth.

---

## Phase 13 - Quantization (fp16 / bf16) (v0.7)

**Goal:** Add `f16` and `bf16` as supported element types.

**Why after CUDA:** on CPU these dtypes are software-emulated by the `half` crate -
half the memory, but *slower* than `f32` in every other way. On tensor cores they
are faster *and* smaller, which is the whole point. Sequencing them after the GPU
backend means they land as a win instead of a benchmark regression.

### Implementation
- Add `half::f16` and `half::bf16` `Dtype` impls.
- CUDA path: tensor-core GEMM. Mixed-precision matmul (`f16` inputs, `f32`
  accumulator) is the common production shape - design the `MatMul` op to take
  separate input and accumulator dtypes.
- CPU path: software arithmetic via the `half` crate, for correctness testing and
  for loading f16 checkpoints (Phase 11's loader stops erroring on them here).
- MKL path: `cblas_h*gemm` where available; soft fallback otherwise.

---

## Phase 14 - Symbolic Autodiff (v0.8)

**Goal:** Implement automatic differentiation by traversing the existing computation
graph in reverse and building a new gradient graph.

**Why symbolic:** the graph already carries the op and inputs at every node. Building
the backward pass as a new graph means gradients get fusion (much more aggressive
thanks to Phase 9), execution planning, buffer reuse - and, after Phase 12, GPU
execution - for free. The same machinery applies to both passes.

**Why this late:** nothing before this phase needs a gradient. The inference story -
blocks, weights, CUDA, quantization - carries the project to "runs a real model
fast" without one, and because gradients are symbolic, every piece of infrastructure
that landed in the meantime applies to them retroactively. Backward passes generate
graphs full of `0`s, `1`s, broadcasts back, and transposed matmuls; landing autodiff
on top of an existing fusion pass means gradient code is fast from day one.

**Multi-output via Stitch.** A backward pass produces one gradient per parameter -
many outputs from a single graph - but the planner roots every plan at one node.
Stitch closes that gap: an N-input op that aliases its first input and drags the rest
into the same plan, so every gradient is scheduled, buffer-packed, and read out in one
`materialize` instead of one plan per gradient (which would recompute the shared
forward activations each time). Stitch is terminal-only: embedded inside a larger
graph it degrades to its single-output alias, because multi-output *composition*
through a `Baked` node is the expensive case and gradients never need it. The design
was settled well before this phase (it has no forcing consumer until now); it lands
here because this is that consumer. Multi-root skeletons (v0.3's external-buffers
note) surface the same machinery through the skeleton API: several roots at
`into_skeleton`, positional outputs, terminal-only with `compose` erroring rather
than degrading.

**Gradient rules you will need** (math, not implementation):
- `AxBy(a, b)`: `grad_input = a * grad_output`
- `Add`: `grad_lhs = grad_output`, `grad_rhs = grad_output`
- `Sub`: `grad_lhs = grad_output`, `grad_rhs = -grad_output`
- `Mul`: `grad_lhs = rhs * grad_output`, `grad_rhs = lhs * grad_output`
- `Matmul`: `grad_lhs = grad @ rhs^T`, `grad_rhs = lhs^T @ grad`
- `ReduceSum`: gradient is the broadcast of `grad_output` back to the input shape

### Tests
```rust
// Gradient of sum(x^2) w.r.t. x = 2x
let x = Parameter::new(Tensor::from_data(&[1.0, 2.0, 3.0], &[3]));
let loss = (x.as_promise() * x.as_promise()).sum(0, false);
let grads = loss.backward();
assert_approx_eq!(grads[&x].data(), &[2.0, 4.0, 6.0]);

// Chain rule: d/dx (3x + 1)^2 = 2(3x+1)*3 = 6(3x+1)
let x = Parameter::new(Tensor::from_scalar(2.0, &[1]));
let y = ((x.as_promise() * 3.0 + 1.0) * (x.as_promise() * 3.0 + 1.0)).sum(0, false);
let grads = y.backward();
assert_approx_eq!(grads[&x].data(), &[42.0]);

// Gradient flows through matmul
let w = Parameter::new(Tensor::from_scalar(1.0, &[3, 3]));
let x = Tensor::from_scalar(1.0, &[2, 3]);
let loss = linear(x, w, zeros(&[3])).sum(0, false).sum(0, false);
let grads = loss.backward();
assert_eq!(grads[&w].shape(), &[3, 3]);
```

### Documentation
- Add an `Autodiff` section to `README.md`
- Create `examples/gradient_descent.rs`: simple 1D regression for 10 steps
- Document the gradient rule for each `OpKind`

---

## Phase 15 - ONNX Importer (transformer subset) (v0.9)

**Goal:** Load a useful subset of ONNX models - enough that someone with a real model
can try Candela without transcribing by hand.

**Why last:** Phase 11 already covers "load my weights" for anyone willing to write
their own forward pass; ONNX removes that last step. The graph IR is already
DAG-shaped, so the translation from ONNX nodes to `OpKind` is mechanical. The harder
part is shape inference for ops Candela doesn't natively support (those become
`unimplemented!()` errors with the op name in the message).

**Scope:** transformer-shaped models - `Gemm`, `MatMul`, `LayerNorm`, `Softmax`,
`ReLU`, `GeLU`, `Add`, `Mul`, `Reshape`, `Transpose` and friends. Everything in that
list exists by Phase 11. Convolution is deliberately out: `Conv` is a phase-sized op
on its own, and it earns that phase when a model someone actually wants to run
demands it.

---

## Automated Testing

Tests should be written alongside each feature as it lands, not after the fact.

### Recommended crates

Add to `[dev-dependencies]` in `Cargo.toml`:

```toml
[dev-dependencies]
approx    = "0.5"   # assert_relative_eq!, assert_abs_diff_eq!
proptest  = "1"     # property-based testing
criterion = "0.5"   # benchmarks (goes under [[bench]])
```

### Directory structure

```
src/
  tensor/
    layout.rs        ← unit tests live here in #[cfg(test)] mod tests { }
    ops/
      fusion.rs      ← unit tests for every fusion rule
      impl_layout.rs ← unit tests for compute_layout per OpKind
tests/
  ops.rs             ← end-to-end op correctness (integration)
  graph.rs           ← materialization, shared nodes, caching
  layout.rs          ← slice/view/transpose/broadcast round-trips
  regression.rs      ← one test per bug that was found and fixed
benches/
  throughput.rs      ← criterion benchmarks
```

---

### Unit tests

Unit tests live inside the module they test using `#[cfg(test)] mod tests { }`.
They test one function or one code path in isolation.

**`mem_formats/layout.rs`**
- [ ] `from_shape` produces correct strides for 1D, 2D, 3D tensors
- [ ] `view` rejects incompatible total size
- [ ] `view` rejects non-contiguous input (`NonContiguousView`)
- [ ] `slice` produces correct shape, stride, and offset for a 2D subview
- [ ] `transpose` swaps shape and stride of the last two axes
- [ ] `transpose_axes` permutes correctly and rejects out-of-bound axes
- [ ] `is_contiguous` returns true for fresh `from_shape`, false after transpose
- [ ] `broadcast_to_shape` sets stride to 0 for expanded dimensions
- [ ] `broadcast_to_shape` sets `len` to the product of the *new* shape
- [ ] `broadcast_to_shape` returns `CannotBroadcast` for incompatible shapes

**`ops/fusion.rs`**
- [ ] Two consecutive `AxBy` ops fuse into one with correct `(a, b)` constants
- [ ] `AxBy` followed by `Exp` produces `FusedScalar([AxBy, Exp])`, not a single op
- [ ] `Exp` followed by `AxBy` produces `FusedScalar([Exp, AxBy])`
- [ ] View followed by AsContiguous fuses to just the View
- [ ] A node that is not scalar-compatible produces no fusion (`None`)

**`ops/impl_layout.rs`**
- [ ] `compute_layout` for `ScalarOp` returns the same layout as the input
- [ ] `compute_layout` for `Add` with equal shapes returns that shape
- [ ] `compute_layout` for `Add` with mismatched shapes returns `NotSameShape`
      with *both* shapes correctly reported (regression for the copy-paste bug)
- [ ] `compute_layout` for `Matmul` returns `[m, n]` for a `[m, k] @ [k, n]` input
- [ ] `compute_layout` for `Matmul` returns `NotSameBatch` for incompatible batch dims

---

### Integration tests (`tests/`)

Integration tests build graphs and materialize them end-to-end.
All expected values should be hand-computed and commented.

**`tests/ops.rs` - scalar ops**
```rust
// One test per OpKindScalar arm - catches copy-paste regressions
#[test] fn scalar_add() { /* ones(4) + 2.0 == [3;4] */ }
#[test] fn scalar_sub() { /* ones(4) - 2.0 == [-1;4] */ }
#[test] fn scalar_mul() { /* ones(4) * 3.0 == [3;4] */ }
#[test] fn scalar_div() { /* ones(4) / 4.0 == [0.25;4] */ }
#[test] fn scalar_exp() { /* from_scalar(0.0).exp() == [1.0;4] */ }
#[test] fn scalar_ln()  { /* from_scalar(1.0).ln()  == [0.0;4] */ }

// Fusion produces the same result as sequential application
#[test] fn fused_chain_correctness() {
    // (x * 2 + 3 - 1) applied with 20 ops matches single-pass
}

// Non-commutative ops are not reversed
#[test] fn sub_is_not_commutative() {
    let a = arange(4);      // [0,1,2,3]
    let b = ones(&[4]);
    let ab = (a - b).materialize(); // [-1,0,1,2]
    let ba = (b - a).materialize(); // [1,0,-1,-2]
    assert_ne!(ab.data(), ba.data());
}
#[test] fn div_is_not_commutative() { /* similar */ }
```

**`tests/ops.rs` - binary tensor ops**
```rust
#[test] fn tensor_add() { /* [1,2,3] + [4,5,6] == [5,7,9] */ }
#[test] fn tensor_sub() { /* [4,5,6] - [1,2,3] == [3,3,3] */ }
#[test] fn tensor_mul() { /* [1,2,3] * [4,5,6] == [4,10,18] */ }
#[test] fn tensor_div() { /* [4,6,8] / [2,3,4] == [2,2,2] */ }

// Shape mismatch panics (inline ops) and does not silently produce wrong output
#[test]
#[should_panic]
fn tensor_add_shape_mismatch_panics() {
    let _ = (ones(&[3]) + ones(&[4])).materialize();
}
```

**`tests/layout.rs` - shape operations**
```rust
#[test] fn view_zero_copy()       { /* same Arc pointer before and after */ }
#[test] fn transpose_zero_copy()  { /* same Arc pointer */ }
#[test] fn slice_zero_copy()      { /* same Arc pointer */ }
#[test] fn transpose_then_materialize_matches_manual() {
    // [[1,2],[3,4]].T == [[1,3],[2,4]]
}
#[test] fn slice_then_add()       { /* slice [2..4] of [0,1,2,3,4] + 1 == [3,4] */ }
#[test] fn as_contiguous_on_transposed_produces_correct_data() { }
```

**`tests/graph.rs` - graph execution**
```rust
// A node used by two branches is computed exactly once
#[test] fn shared_node_computed_once() {
    let t = arange(4).as_promise();
    let result = (&t * 2.0 - &t).materialize(); // == t
    assert_eq!(result.data(), &[0.0, 1.0, 2.0, 3.0]);
}

// CachedTensorPromise returns the same result on repeated materializations
#[test] fn cached_promise_stable() {
    let raw = arange(4);
    let cached = (raw.as_promise() + 1.0).cache();
    let r1 = (&cached * 2.0).materialize();
    let r2 = (&cached * 2.0).materialize();
    assert_eq!(r1.data(), r2.data());
}

// CachedTensorPromise is computed only once (result is shared, not recomputed)
#[test] fn cached_promise_computed_once() {
    // Build a graph where the cached node would produce different results
    // if re-evaluated (e.g., wrapping a counter). Verify it does not change.
}
```

---

### Regression tests (`tests/regression.rs`)

One test per bug that was found. Named clearly so it's obvious what broke.

```rust
// Bug: OpKindScalar::Sub was doing addition (copy-paste from Sum arm).
#[test] fn regression_scalar_sub_was_adding() {
    let t = Tensor::from_scalar(10.0, &[4]);
    assert_eq!((t - 3.0).materialize().data(), &[7.0; 4]);
}

// Bug: NotSameShape error reported inputs[0].shape() for both sides.
#[test] fn regression_not_same_shape_error_shows_both_shapes() {
    let a = ones(&[3, 4]);
    let b = ones(&[3, 5]);
    // The panic message must contain both "4" and "5".
    // Use std::panic::catch_unwind + check the string.
}

// Bug: unordered buffer reuse could pick rhs for output, reversing Sub/Div.
#[test] fn regression_sub_ordering_with_reusable_rhs() {
    let a = Tensor::from_scalar(10.0, &[4]);
    let b = (Tensor::from_scalar(3.0, &[4]).as_promise() + 0.0); // reusable
    assert_eq!((a - b).materialize().data(), &[7.0; 4]); // not [-7.0; 4]
}

// Bug: matmul inputs.pop() order was reversed (raw_a got rhs, raw_b got lhs).
// Add once matmul is implemented.
#[test] fn regression_matmul_input_order() {
    let a = Tensor::from_data(&[1.0,0.0, 0.0,1.0], &[2,2]); // identity
    let b = Tensor::from_data(&[1.0,2.0, 3.0,4.0], &[2,2]);
    // identity @ b == b, not b^T
    assert_eq!(a.matmul(&b).unwrap().materialize().data(), b.data());
}
```

---

### Property tests (`tests/property.rs`)

Property tests use `proptest` to generate random inputs and verify algebraic laws.
These catch edge cases that hand-written tests miss (large values, zero, negatives).

```rust
use proptest::prelude::*;

proptest! {
    // Commutativity of Add
    #[test]
    fn add_commutative(a in prop::collection::vec(-1e6f64..1e6, 1..100),
                       b in prop::collection::vec(-1e6f64..1e6, 1..100)) {
        prop_assume!(a.len() == b.len());
        let ta = Tensor::from_data(&a, &[a.len()]);
        let tb = Tensor::from_data(&b, &[b.len()]);
        let ab = (ta + tb).materialize();
        let ba = (tb + ta).materialize();
        assert_relative_eq!(ab.data(), ba.data(), max_relative = 1e-10);
    }

    // Commutativity of Mul
    #[test]
    fn mul_commutative(/* similar */) { }

    // Sub is anti-commutative: a - b == -(b - a)
    #[test]
    fn sub_anticommutative(a in vec, b in vec) {
        let ab  = (ta - tb).materialize();
        let neg_ba = (tb - ta) * -1.0).materialize();
        assert_relative_eq!(ab.data(), neg_ba.data(), max_relative = 1e-10);
    }

    // Additive identity: a + 0 == a
    #[test]
    fn add_zero_identity(a in vec) {
        let t    = Tensor::from_data(&a, &[a.len()]);
        let zero = Tensor::from_scalar(0.0, &[a.len()]);
        let result = (t + zero).materialize();
        assert_relative_eq!(result.data(), &a, max_relative = 1e-10);
    }

    // Multiplicative identity: a * 1 == a
    #[test]
    fn mul_one_identity(a in vec) { }

    // Scalar fusion produces the same result as sequential materialization
    #[test]
    fn fused_scalar_matches_sequential(
        a in -1e3f64..1e3,
        b in -1e3f64..1e3,
        c in -1e3f64..1e3,
        data in vec(-1e6f64..1e6, 1..1000),
    ) {
        let t = Tensor::from_data(&data, &[data.len()]);
        let fused      = (t.as_promise() * a + b - c).materialize();
        let sequential = data.iter().map(|&x| x * a + b - c).collect::<Vec<_>>();
        assert_relative_eq!(fused.data(), sequential.as_slice(), max_relative = 1e-10);
    }

    // View preserves all elements
    #[test]
    fn view_preserves_data(data in vec(any::<f64>(), 12..=12)) {
        let t = Tensor::from_data(&data, &[12]);
        let viewed = t.view(&[3, 4]).unwrap().materialize();
        assert_eq!(viewed.data(), &data);
    }
}
```

---

### Benchmarks (`benches/throughput.rs`)

Use `criterion`. Run with `cargo bench`.

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_scalar_fusion(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let mut group = c.benchmark_group("scalar_fusion");

    for &n in &sizes {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let t = arange(n);
            b.iter(|| {
                let mut p = t.as_promise();
                for i in 0..20 {
                    p = p + black_box(i as f64);
                }
                black_box(p.materialize())
            });
        });
    }
}

fn bench_matmul(c: &mut Criterion) {
    // 128x128, 512x512, 1024x1024
}

fn bench_non_contiguous_vs_contiguous(c: &mut Criterion) {
    // Compare (transposed + op).materialize() vs (contiguous + op).materialize()
    // to quantify the non-contiguous penalty.
}

criterion_group!(benches, bench_scalar_fusion, bench_matmul, bench_non_contiguous_vs_contiguous);
criterion_main!(benches);
```

---

### CI checklist

The GitHub Actions workflow should run all of the following on every push:

- [x] `cargo fmt --check` - no formatting drift
- [x] `cargo clippy -- -D warnings` - no lint regressions
- [x] `cargo test` - all unit and integration tests pass
- [ ] `cargo test --features debug_only_check` - validation paths also exercised
- [ ] `cargo bench --no-run` - benchmarks compile (do not run timing on CI)

---

## Documentation Backlog

Items that are not blocked on any phase but should be done incrementally.

- [x] Replace `simple_tensor` package name in `Cargo.toml` with `candela` to match
      `README.md` and the project name
- [x] ~~Add `#![doc = include_str!("../README.md")]` to `src/lib.rs`~~ Decided
      against: the README and the crate root serve different readers (see
      `doc/style.md` §6). `lib.rs` carries its own crate-root doc instead
- [ ] Add `# Safety` doc comments to every `pub unsafe fn` explaining the invariants
      the caller must uphold
- [ ] Add `# Panics` doc comments to every function that can panic, including the
      inline `+`, `-`, `*`, `/` operators
- [ ] Add doc comments (with `# Examples`) to the method macros in
      `src/tensor/ops/impl_op.rs`: `view`, `reshape`, `slice`, `transpose`,
      `transpose_axes`, `as_contiguous`, `exp`, `ln`, `log2`, and the scalar
      arithmetic operators. These are safe to place inside the macro body -
      rustdoc picks them up for each expanded type (`Tensor`, `TensorPromise`,
      `CachedTensorPromise`)
- [ ] Create `examples/` directory with at minimum:
      - `lazy_eval.rs` - basic graph construction and materialization
      - `fusion.rs` - scalar fusion collapsing a 20-op chain
      - `cached_promise.rs` - shared preprocessing across multiple flows
      - `matmul.rs` - matrix multiplication
      - `gradient_descent.rs` - simple regression (after Phase 14)
- [ ] Conformance tests against NumPy for every op: generate inputs in Python, export
      as JSON, import in Rust tests and compare outputs within floating-point tolerance
- [ ] Add `benches/` with at minimum:
      - Materialization throughput for a scalar-fused chain (large tensor)
      - Matmul throughput vs naive loop
      - Non-contiguous vs contiguous op throughput ratio
