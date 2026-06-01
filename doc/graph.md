# The Computation Graph

When you build a chain of operations on a `TensorPromise`, Candela isn't running anything
yet - it's building a graph. This document explains what that graph actually looks like
under the hood.

---

## Nodes

Every node in the graph is one of three things, defined in `src/tensor/graph.rs`:

### `TensorGraphEdge<T>` - the leaves

An edge is how a plain `Tensor` enters the graph. When you call `.as_promise()`, Candela
wraps your tensor in an edge node. It holds a reference-counted copy of the data and
contributes it to the graph without any computation attached to it.

The name is non-standard - in graph theory these would be called leaf nodes or source
nodes - but the intent is that a `TensorGraphEdge` marks the *boundary* where a concrete
`Tensor` enters the lazy evaluation world.

### `TensorGraphNode<T>` - the computations

This is where actual work lives. A node holds:
- an `OpKind` (what to do),
- a list of input nodes (where the data comes from),
- an output `Layout` (the shape and stride of the result).

One thing worth knowing: the constructor (`new`) runs **operator fusion** before storing
the op. If you chain scalar operations, they get collapsed into a single `FusedScalar`
node right here, before the node even exists. By the time you call `.materialize()`, the
graph is already as lean as Candela can make it.

To make this concrete: writing `(t + 1.0) * 2.0` creates a `ScalarOp` node for `+ 1.0`.
When `* 2.0` tries to construct a second node, fusion intercepts - both are scalar ops,
so they collapse. What gets stored is a single `FusedScalar` node computing `2x + 2` in
one pass over the data, identical in result to running the two ops in sequence. The
intermediate node was absorbed before it was ever stored.

```
// What you wrote:    Edge(t)  ←  ScalarOp(+1.0)  ←  ScalarOp(×2.0)
// What gets stored:  Edge(t)  ←  FusedScalar(2x + 2)
```

One thing fusion does *not* do: it skips `Edge` inputs entirely. Fusion works by
inspecting a new node's inputs looking for `Node` or `Cache` children to collapse -
`Edge` children are passed over. This means that if an `AsContiguous` node has a raw
`Tensor` as its only input, the usual contiguity check never fires and the node stays
in the graph. `as_contiguous` handles this specific case by short-circuiting: when the
source is an already-contiguous `Tensor`, it emits a `NoOp` node instead of `AsContiguous`
entirely, which the planner then treats as a reference pass-through. The gap still exists
for other ops that build nodes over `Edge` inputs, but none of the built-in ops currently
need that kind of fusion.

### `TensorGraphCacheNode<T>` - the persistent ones

A cache node wraps a `TensorGraphNode` and adds a `OnceLock<TensorData<T>>`. The
computation inside runs at most once. After that, every call to `.compute()` returns the
stored result.

Calling `.cache()` on a promise builds a two-node structure: an inner `AsContiguous`
node that packs the result into contiguous memory, followed by a `NoOp` cache node
wrapping it. The `AsContiguous` guarantees the stored result is always contiguous, so
ops that require contiguous memory (like a BLAS-backed matmul) can consume a filled
cache without re-packing. The `NoOp` is necessary to prevent operator fusion from
collapsing across the cache boundary - without it, a scalar op adjacent to `.cache()`
could fuse into the inner node and bypass the cache entirely.

---

## Sharing and reference counting

All three node types are wrapped in `Arc`. This means a node can safely appear as an
input to multiple parents without copying anything - the graph is shared. The execution
planner uses node IDs to detect shared nodes and schedule them correctly (each shared
node is computed exactly once per `.materialize()` call).

Candela wasn't always graph-based. Earlier iterations ran operations eagerly - `+`
returned a `Tensor<T>` immediately - and `Slice` and `View` were their own dedicated
types with separate implementations. That worked, but operator fusion requires inspecting
the whole operation chain *before* executing any of it, which means the graph had to
exist first.

Once a shared DAG is the target (a DAG - directed acyclic graph - is just a graph where
edges have direction and nothing loops back to a dependency), reference counting is the
natural fit in Rust. A node can appear as an input to multiple parents, and each parent
needs to keep it alive without knowing about the others. `Arc` does that. An alternative
- storing all nodes in a `Vec` and using indices as handles - avoids the atomic overhead
but requires a single owner for the entire graph, which gets awkward when building
branches independently. Hugging Face's [Candle](https://github.com/huggingface/candle)
uses a similar approach, which was enough validation to commit to it.

---

## What comes next

Once a graph is built, calling `.materialize()` hands it off to the execution planner,
which works out the most memory-efficient order and buffer assignment before running
anything. That process is described in [planner.md](planner.md).
