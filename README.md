# Candela

Candela is a Rust tensor execution-engine project focused on **explicit memory behavior**, **lazy graph execution**, and **performance-oriented CPU compute paths**.

It is primarily a learning-driven systems project—but not a throwaway toy. The codebase explores the hard parts of tensor runtime design from first principles: shape/stride layout semantics, DAG materialization, buffer reuse, operation fusion, and controlled caching.

---

## Overview

Candela currently provides:

- **`Tensor<T>`** for materialized tensor data,
- **`TensorPromise<T>`** for deferred computation nodes,
- **`CachedTensorPromise<T>`** for explicit cache persistence,
- a CPU compute path currently specialized for **`f64`**,
- Intel MKL/VML-backed elementwise kernels,
- layout-aware transforms (view/slice/transpose) and iterator machinery for contiguous and non-contiguous tensors.

This repository is best understood as an execution-runtime core under active development.

---

## Why Candela Exists

Candela exists to answer a practical engineering question:

> How do you build a tensor runtime in Rust that is both ergonomic enough to use and explicit enough to preserve real control over allocations, copies, and execution?

The project’s position is that many tensor ecosystems over-optimize for convenience and hide too much memory behavior. Candela intentionally moves in the opposite direction where possible:

- keep allocations explicit,
- keep copies explicit,
- avoid retaining intermediates unless the user explicitly requests it,
- make caching a user-level choice.

The design notes in `project_decision.txt` reinforce this direction, including explicit discussion of eager intermediate discard during materialization and user-driven caching policy.

---

## Design Philosophy

Candela is aligned with core Rust instincts:

1. **Explicit copies over silent duplication**
2. **Explicit allocations over hidden ownership transitions**
3. **Explicit memory control over opaque runtime retention**
4. **User-directed caching over automatic graph hoarding**

This philosophy is reflected directly in the node model:

- **Edge nodes**: materialized tensor data,
- **Node nodes**: deferred operation nodes,
- **Cache nodes**: deferred nodes with persisted output (`OnceCell`) for reuse across materializations.

There are ergonomic compromises (e.g., overloaded operators for expression building), but they are deliberately constrained.

---

## Inspiration / Related Projects

Candela is inspired by Rust tensor and ML systems work, especially:

- [tensorkraken](https://github.com/kurtschelfthout/tensorken) *(as referenced by the project author’s notes)*
- [candle](https://github.com/huggingface/candle)
- [burn](https://github.com/tracel-ai/burn)

The project name and direction are strongly influenced by this ecosystem.

---

## Key Features (Current)

- **Lazy DAG execution** via `TensorPromise`.
- **Topological materialization** with dependency-aware execution.
- **Reference-count-guided intermediate release** to avoid retaining temporary tensors longer than necessary.
- **Explicit cache nodes** (`CachedTensorPromise`) for reusable subgraphs.
- **Scalar-op fusion** for compatible scalar chains.
- **Layout-first tensor semantics**:
  - shape / stride / adjusted-stride / offset / len,
  - view, slice, transpose, transpose-by-axes,
  - contiguous and non-contiguous iterator paths.
- **CPU elementwise kernels for `f64`** using Intel VML (`vdAdd`, `vdSub`, `vdMul`, `vdDiv`).
- **Macro conveniences** for construction and slicing (`arange!`, `srange!`, `zeros!`, `ones!`, `s!`).

---

## Architecture / Execution Model

### 1) Tensor Data Layer

- `Storage<T>` wraps `Arc<RwLock<Vec<T>>>`.
- `TensorData<T>` pairs storage with `Layout` and a `reusable` marker.
- `Layout` tracks shape/stride/adj_stride/offset/len and provides transformation methods.

This separation allows layout rewrites without mandatory data movement.

### 2) Graph Representation

Graph nodes are represented by `NodeKind`:

- `Edge(Arc<TensorGraphEdge<T>>)`
- `Node(Arc<TensorGraphNode<T>>)`
- `Cache(Arc<TensorGraphCacheNode<T>>)`

Each node receives a stable runtime id via an atomic counter.

### 3) Materialization Pipeline

When `.materialize()` is called on a promise:

1. the graph is traversed with DFS topological sort,
2. usage counts are collected,
3. nodes are computed in dependency order,
4. inputs are removed from cache as soon as their refcount reaches zero,
5. cached nodes reuse or populate their internal `OnceCell`.

The intent is to minimize live intermediate memory while preserving deterministic execution.

### 4) Compute Dispatch

`cpu_compute(op, output_layout, inputs)` dispatches by scalar type (`ComputeWrapperSpec`).

Current concrete implementation: **`f64`**.

Implemented operation groups include:

- scalar ops,
- fused scalar op chains,
- layout transforms (`View`, `Slice`, `Transpose`, `TransposeAxes`, `AsContiguous`),
- tensor-tensor elementwise ops (`Add`, `Sub`, `Mul`, `Div`),
- no-op forwarding.

`Matmul` layout inference exists, and matmul compute scaffolding is present, but full matmul execution is not complete.

---

## Memory Model and Allocation Philosophy

Candela’s memory behavior is intentionally visible:

- **Reusable outputs**: operations can mark outputs reusable.
- **Reuse-first strategy**: execution attempts to reuse uniquely owned buffers (`Arc::try_unwrap`) before allocating fresh vectors.
- **Contiguity-aware execution**: kernels branch between contiguous fast paths and generic strided iteration paths.
- **Eager discard of intermediates**: during graph execution, temporaries are dropped as soon as their dependency role ends.
- **User-directed persistence**: `.cache()` is explicit and opt-in.

This is a central identity of Candela: retain only what the user asks to retain.

---

## Repository Structure

```text
.
├── Cargo.toml
├── LICENSE
├── project_decision.txt
├── src
│   ├── main.rs
│   └── tensor
│       ├── mod.rs
│       ├── tensor.rs
│       ├── promise.rs
│       ├── graph.rs
│       ├── storage.rs
│       ├── traits.rs
│       ├── iter.rs
│       ├── internals.rs
│       ├── errors.rs
│       ├── convenience.rs
│       ├── macros.rs
│       ├── impl_generics.rs
│       ├── mkl_extension.rs
│       ├── mem_formats
│       │   ├── layout.rs
│       │   └── slice.rs
│       └── ops
│           ├── impl_op.rs
│           ├── impl_layout.rs
│           ├── impl_compute_op.rs
│           ├── fusion.rs
│           └── reusable.rs
```

---

## Build Instructions

### Prerequisites

- Rust toolchain (`rustc`, `cargo`)
- Toolchain/runtime support for dependencies in `Cargo.toml` (MKL-related crates are included)

### Build

```bash
cargo build
```

### Optional debug-check feature

```bash
cargo build --features debug_only_check
```

---

## Usage Example

Current `src/main.rs` demonstrates lazy scalar graph construction and materialization:

```rust
use crate::tensor::{Dimension, Tensor, arange};

fn main() {
    let t1 = arange![12];
    let mut p = t1.as_promise();

    for i in 0..20 {
        p = p + i as f64;
    }

    println!("{}", (p * 2.0).materialize());
}
```

Run:

```bash
cargo run
```

---

## Performance Considerations

Candela’s current performance strategy combines several systems-level choices:

- **Operation fusion (currently limited):** scalar chain fusion reduces repeated passes and temporary graph depth.
- **Buffer reuse:** contiguous reusable outputs are recycled where ownership allows.
- **Layout-aware iteration:** contiguous tensors use fast linear iterators; non-contiguous tensors use stride-aware iterators with precomputed `adj_stride` stepping.
- **DAG-level memory pressure reduction:** topological execution + reference counting releases intermediates aggressively.
- **Vendor kernels where beneficial:** Intel VML is used for core elementwise vector operations in the `f64` path.

No formal benchmark suite is currently included in the repository.

---

## Current Limitations

- Backend specialization is effectively **`f64`-first** today.
- Full matmul compute is still incomplete.
- Fusion logic is intentionally conservative and order-dependent.
- Broadcasting support is incomplete / evolving.
- There are currently no repository-integrated unit test or benchmark suites.
- Some files contain WIP paths, TODOs, or compiler warnings indicative of active development.

---

## TODO / Roadmap

The following are **future directions**, not currently implemented end-to-end features:

- [ ] Add CUDA backend support, including async execution support.
- [ ] Implement custom CPU kernels for non-contiguous tensor execution paths.
- [ ] Add model building blocks (e.g., `Linear`, `ReLU`, and related components).
- [ ] Expand dtype support (`f32`, `i32`, and others) with backend coverage.
- [ ] Add benchmark suites for CUDA kernels.
- [ ] Add benchmark suites for CPU kernels (contiguous and strided cases).
- [ ] Add Promise serialization for debugging, reproducibility, and graph inspection.
- [ ] Explore a **PromiseSkeleton**-style compile/pipeline representation for model reconstruction workflows.
- [ ] Complete and optimize matmul kernels (including batched scenarios).
- [ ] Improve diagnostics and consistency around panic-vs-`Result` behavior.

---

## Contributing

Contributions are welcome, especially in:

- compute kernel development,
- graph/fusion improvements,
- memory-model correctness and profiling,
- test and benchmark infrastructure,
- API ergonomics that preserve explicit memory control.

When contributing, prefer changes that keep allocation/copy behavior transparent and maintain the project’s explicit-control philosophy.

---

## License

MIT License. See [`LICENSE`](./LICENSE).
