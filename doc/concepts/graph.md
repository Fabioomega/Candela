# The Computation Graph

Building a chain of operations on a `TensorPromise` runs nothing - it builds a graph.
This document covers what that graph is made of and how it is shaped before the planner
ever sees it.

---

## The nodes

Every node in the graph is one of three kinds:

- **`TensorGraphEdge<T>`** - the internal representation of a real, storage-backed
  tensor. It is where the graph usually starts.
- **`TensorGraphNode<T>`** - the operation nodes. Holds an operation, the operation's
  inputs, and the output layout.
- **`TensorGraphCacheNode<T>`** - the same as a `TensorGraphNode`, but caches the output
  in a `OnceLock` so it only needs to be run once.

---

## Fusion at construction

Operation fusion - unifying operations into more optimized, grouped versions to improve
performance - happens during graph creation. It runs eagerly, per node created, applying
a greedy fusion strategy that produces a single more optimized operation. One such example
is a scalar chain: `(t + 1.0) * 2.0` builds a scalar node for `+ 1.0`, and when `* 2.0`
goes to build a second one, fusion folds both into a single node computing `2t + 2` in one
pass.

```text
// input chain:   Edge(t)  ←  ScalarOp(+1.0)  ←  ScalarOp(×2.0)
// stored node:   Edge(t)  ←  FusedScalar(2t + 2)
```

---

## Sharing and reference counting

Every node is wrapped in `Arc`, so it can appear as an input to several parents without
copying anything - the graph is shared. As long as the child is alive the whole graph
above it remains alive too, edges included, to maintain the validity of that operation.

Each node carries an ID that is unique across the graph, which enables deduplication
during planning. Comparing two operations for equality is then a matter of comparing IDs;
proving it otherwise would require complex chain tracking, since the nodes would need the
exact same parents and the same operation - cumbersome to track.

---

## What comes next

Once a graph is built, calling `.materialize()` hands it off to the execution planner,
which works out the memory-efficient order and buffer assignment before running anything.
That process is described in [the execution planner](crate::docs::planner).
