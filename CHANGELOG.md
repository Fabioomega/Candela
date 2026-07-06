# Changelog

All notable changes to Candela are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows the pre-1.0 Cargo convention where a minor bump (`0.x → 0.(x+1)`) is a
release arc allowed to break the API.

## [0.2.0] - 2026-06-26

Skeletons: compile a graph once, run it many times, and know your costs before
you run.

### Added

- **Skeletons.** Pre-compile a graph's execution plan once and reuse it across
  runs with new data, skipping per-call planning.
  - `SkeletonSlot` - a typed hole in the graph: a `Layout` with no data behind
    it. Built from a `Layout` via `SkeletonSlot::new`, or borrowed from an
    existing tensor or promise with `.as_slot()`.
  - `SkeletonPromise` - what any op chain containing a slot becomes. The slot
    taint is enforced at the type level: an expression with a slot in its
    lineage has no `.materialize()`, so forgetting a slot is a compile error.
    The only exit is `into_skeleton`.
  - `Skeleton` - the compiled template. `run(&[&Tensor])` executes it; `compose`
    embeds its plan into a larger graph as a `BakedPromise`.
  - `BakedPromise` - a skeleton's plan spliced into an ordinary promise
    expression.
- **Dynamic-shape skeletons.** `DynamicSkeleton` (and `UnboundedDynamicSkeleton`)
  recompile per distinct input layout, keyed by the full tuple of slot layouts.
  Built on a public `SkeletonCache` with a pluggable `EvictionPolicy`
  (`LRUPolicy` and `UnboundedPolicy` ship) and a `BuildFunction` that fires only
  on a cache miss.
- **`Skeleton::memory_report()`.** Returns `MemoryMetrics` - peak bytes,
  allocation count, and buffer sizes - read straight off the compiled plan.
- **Public `skeleton` and `backend` modules** re-exporting the relevant types
  from the crate root, plus the `Composable` trait (the operand bound `compose`
  accepts).
- New `OpError` variants for skeleton binding: `IncorrectSlotAmount`,
  `NotSameSlot`, `NotSameLayoutAtSlot`.
- `doc/skeleton.md` (the slot/promise/skeleton/baked taxonomy and the taint
  algebra), a skeletons section in the README, `examples/dyn_skeleton.rs`, and
  test suites in `tests/skeleton.rs`, `tests/dynamic_skeleton.rs`, and
  `tests/memory.rs`.

### Changed

- **`tracing` is now an opt-in feature with an optional dependency.** It was a
  default feature; the default build no longer compiles or links `tracing` at
  all. The instrumentation sits on the compute kernels and planner at
  `trace` level - enable it with `--features tracing` when you need it.
- **Renamed `Tensor::clone_deep` to `Tensor::deep_clone`** for a consistent
  `verb_noun` method name. (breaking)
- **Reworked `Layout`'s constructors into a friendlier, more consistent API.**
  `Layout::new` now takes just a shape (`Layout::new(&[usize])`) for the common
  contiguous case; the old raw-field constructor is now `Layout::from_raw_parts`;
  `from_slice` is now `from_strided`; and `from_shape(shape, offset)` is replaced
  by `Layout::new(shape).with_offset(offset)`. (breaking)
- The executor was split into its own `src/tensor/executor.rs`, and the planner
  gained an owned-plan representation (`planner/owned.rs`) so a compiled plan can
  be stored and replayed - the foundation skeletons run on.

### Fixed

- In-place ops were never actually applied and silently forced an allocation;
  the planner and runtime now perform them in place as intended.
- `Layout` hashing, which the skeleton cache relies on to key entries by input
  layout.

### Removed

- The `approx` dev-dependency, replaced by in-house approximate-equality helpers
  in `tests/common`.

[0.2.0]: https://github.com/Fabioomega/candela/releases/tag/v0.2.0
