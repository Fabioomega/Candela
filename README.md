# Candela

A lazy, graph-based tensor engine written in Rust.

Candela is a learning project built to understand how tensor engines work from the inside: memory layouts, computation graphs, operator fusion, and BLAS integration. It currently supports CPU execution with `f64` tensors. GPU support is on the roadmap.

---

## Why "Candela"?

The name draws from two excellent Rust ML libraries — **[Candle](https://github.com/huggingface/candle)** and **[Burn](https://github.com/tracel-ai/burn)** — but tries to be its own thing. A *candela* is the SI unit of luminous intensity. The name felt fitting: this project is meant to illuminate — to teach me (an undergraduate student) how tensor systems work under the hood, and hopefully shed some light for anyone else curious enough to want to build something similar.

---

## Inspirations

These projects provided massive inspiration throughout the development of Candela:

- **[Candle](https://github.com/huggingface/candle)** — Hugging Face's minimalist ML framework in Rust. A proof that you don't need Python to do serious ML.
- **[Burn](https://github.com/tracel-ai/burn)** — A full-featured deep learning framework in Rust with a thoughtful design around backends and autodiff.
- **[TensorKraken](https://github.com/richardanaya/tensorkraken)** — A key source of inspiration for thinking about tensor graphs and lazy execution.

---

## Core Philosophy

Candela tries to embrace a Rust-native approach to memory: **you should own your allocations**.

Many tensor libraries silently allocate memory for intermediate results and caches. Candela tries to only allocate what you explicitly ask for, preserving the predictability that Rust gives you. The corollary is that **cache is a user choice** — tensors can be large, and the library won't hold on to them without being told to.

This isn't perfectly achieved everywhere yet, but the principle shapes every design decision.

---

## The Three Tensor Types

### `Tensor<T>`

A materialized tensor. It has data, a shape, and a stride. This is the result of a completed computation.

```rust
let t = Tensor::from_scalar(1.0, &[3, 3]); // 3x3 matrix filled with 1.0
let t = arange![12];                        // [0.0, 1.0, ..., 11.0], shape [12]
let t = zeros!(&[2, 3]);                    // 2x3 matrix of zeros
let t = ones!(&[4]);                        // 1D tensor of ones
```

### `TensorPromise<T>`

A description of a computation that hasn't run yet. Building a promise chain allocates no intermediate tensors — the graph is constructed, not evaluated. Calling `.materialize()` runs the whole thing.

```rust
let p = t.as_promise();        // wrap a Tensor into a Promise (free)
let p = p + 5.0;               // add to the graph
let p = p * 2.0;               // continue building
let result = p.materialize();  // execute the graph now
```

When a node appears in multiple branches of the same graph, the execution engine handles it correctly: the topological sort ensures each node is computed exactly once per `.materialize()` call, with intermediate results kept alive via reference counting and eagerly freed once no longer needed.

### `CachedTensorPromise<T>`

A promise that stores its result after evaluation and keeps it alive. This is useful when the same promise is going to be reused across **separate materializations** — for example, a preprocessed input that feeds into multiple independent execution flows called at different times.

```rust
let preprocessed = (raw + bias).cache();

// called at different times, preprocessed is computed only on the first call
let flow_a = (&preprocessed * weight_a).materialize();
let flow_b = (&preprocessed * weight_b).materialize();
```

You pay the memory cost of keeping that tensor alive, which is why this is opt-in.

---

## Lazy Evaluation and Operator Fusion

Operations build a computation graph (a DAG) that runs when you call `.materialize()`. This makes **operator fusion** possible: Candela can inspect the graph and collapse compatible operations before executing anything.

Currently, scalar operations are fused:

```rust
let t = arange![12];
let mut p = t.as_promise();
for i in 0..20 {
    p = p + i as f64;
}
let result = (p * 2.0).materialize();
// 21 scalar ops are fused into a single pass over the data
```

General tensor-level fusion is a future goal.

---

## Memory Layout

Candela separates logical shape from physical memory layout. Views, slices, and transposes are **zero-copy** — they produce a new layout descriptor pointing into the same underlying buffer. (To be fair, this is what every tensor library worth its salt does.)

```rust
let t = arange![0, 12]; // shape [1, 12]

let p = t.as_promise().view(&[3, 4])?;          // reshape, no copy
let p = t.as_promise().transpose();             // swap last two axes, no copy
let p = t.as_promise().slice(s![0..2, 1..3])?; // 2x2 subview, no copy
```

When an operation needs contiguous memory (e.g., for a BLAS call), Candela packs the data at that point using a chunked buffer to keep packing cache-friendly.

For a full explanation of the layout system — including the `adj_stride` field that makes non-contiguous iteration fast — see [doc/layout.md](doc/layout.md).

---

## Error Handling

Inline operators (`+`, `-`, `*`, `/`) **panic** on shape mismatches. This is intentional — if you're adding two tensors of incompatible shapes, that's almost certainly a programming error. Writing `(A + B).unwrap()` everywhere would be a poor tradeoff.

Method-based operations that can meaningfully fail return `Result`:

```rust
let result = a.matmul(&b)?;       // Result<TensorPromise<T>, OpError>
let reshaped = p.view(&[4, 3])?;  // Result<TensorPromise<T>, OpError>
```

The rule is: if the failure is something you should handle at runtime (e.g., user-provided shapes), it returns `Result`. If it's almost certainly a logic bug, it panics.

---

## Features

- Lazy evaluation via computation graph (DAG)
- Scalar operator fusion — long chains collapse into single-pass kernels
- Zero-copy views — reshape, transpose, and slice without touching data
- Memory reuse — reference-counted buffers are reused when safe
- Intel MKL vectorized kernels for `f64` element-wise ops
- Full stride/offset layout system for non-contiguous tensors
- Opt-in result caching via `CachedTensorPromise`
- Built-in `tracing` instrumentation (feature-gated)
- `arange!`, `srange!`, `zeros!`, `ones!` convenience macros

---

## Current Limitations

- **Data types:** only `f64` is backed by a CPU implementation. The generic framework supports any `NumberLike` type — other types just need their backends.
- **Matmul:** the graph and layout logic are complete. The `cblas_dgemm` call is stubbed and not yet fully wired.
- **Broadcasting:** `broadcast_to_shape()` exists in the layout system but isn't yet integrated into element-wise tensor operations.
- **GPU:** none yet.

---

## Internals

If you want to understand how things work under the hood:

- [doc/graph.md](doc/graph.md) — the computation graph: node types, sharing, and how ops fuse during construction
- [doc/planner.md](doc/planner.md) — the execution planner: how Candela decides what to compute, in what order, and which buffers to reuse
- [doc/layout.md](doc/layout.md) — the memory layout system: strides, zero-copy views, and the `adj_stride` iteration trick

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan — phases, rationale, tests, and what comes next.

---

## Building

Candela uses Intel MKL for CPU kernels. The `intel-mkl-src` crate handles linking, but MKL libraries need to be available on your system.

```bash
cargo build
cargo test --doc           # run the doctests
cargo run --example basic  # run the basic example
```

Examples live in `examples/`.

---

## License

MIT

---

## Author

Made by **Fabio** ([@Fabioomega](https://github.com/Fabioomega)).