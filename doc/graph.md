# The Computation Graph

When you build a chain of operations on a `TensorPromise`, Candela isn't running anything
yet — it's building a graph. This document explains what that graph actually looks like
under the hood.

---

## Nodes

Every node in the graph is one of three things, defined in `src/tensor/graph.rs`:

### `TensorGraphEdge<T>` — the leaves

An edge is how a plain `Tensor` enters the graph. When you call `.as_promise()`, Candela
wraps your tensor in an edge node. It holds a reference-counted copy of the data and
contributes it to the graph without any computation attached to it.

### `TensorGraphNode<T>` — the computations

This is where actual work lives. A node holds:
- an `OpKind` (what to do),
- a list of input nodes (where the data comes from),
- an output `Layout` (the shape and stride of the result).

One thing worth knowing: the constructor (`new`) runs **operator fusion** before storing
the op. If you chain scalar operations, they get collapsed into a single `FusedScalar`
node right here, before the node even exists. By the time you call `.materialize()`, the
graph is already as lean as Candela can make it.

### `TensorGraphCacheNode<T>` — the persistent ones

A cache node wraps a `TensorGraphNode` and adds a `OnceLock<TensorData<T>>`. The
computation inside runs at most once. After that, every call to `.compute()` returns the
stored result. This is what you get when you call `.cache()` on a promise — Candela
creates a `NoOp` cache node wrapping the original computation.

---

## Sharing and reference counting

All three node types are wrapped in `Arc`. This means a node can safely appear as an
input to multiple parents without copying anything — the graph is shared. The execution
planner uses node IDs to detect shared nodes and schedule them correctly (each shared
node is computed exactly once per `.materialize()` call).

---

## What comes next

Once a graph is built, calling `.materialize()` hands it off to the execution planner,
which works out the most memory-efficient order and buffer assignment before running
anything. That process is described in [planner.md](planner.md).
