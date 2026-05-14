# Execution Planner

When you call `.materialize()` on a promise, Candela doesn't just start executing nodes
in some arbitrary order. It first builds a *plan* — a fully ordered schedule that says
exactly what to compute, which buffer to write into, and what to free afterward. This
document explains how that plan is built.

If you haven't read [graph.md](graph.md) yet, it's worth a quick look first — the
planner works directly with the node types described there.

---

## Why not just execute the graph directly?

Earlier versions used a `reusable: bool` flag to decide whether a buffer could be
recycled. It worked until it didn't: for non-commutative ops like `Sub` and `Div`, the
flag could pick the *wrong* input buffer as the output, causing `a - b` to silently
compute `b - a`.

The root problem was that "is this buffer reusable?" is the wrong question. The right
question is "is this buffer *still being read* at this point?" That requires knowing
*when* each buffer is last consumed, not just whether it's eligible. The planner answers
that question before execution begins.

---

## Three passes

### 1. Topological sort

The planner starts by sorting the graph in dependency order — inputs before the ops that
consume them. This is done by `topological_sort` in `src/tensor/planner/sort.rs` using
an iterative post-order DFS (no recursion, so deep graphs don't blow the stack).

A few details worth knowing:
- Shared nodes are deduplicated via a `HashSet` of node IDs. If the same node feeds into
  two different parents, it appears in the sorted output exactly once.
- If a `CachedTensorPromise` is already filled, the planner doesn't traverse its subtree
  — it treats it as a leaf.
- The root node (the one you called `.materialize()` on) is *not* yielded by the
  iterator. It's planned separately at the end.

### 2. Lifetime analysis

After sorting, the planner makes one pass to find out when each intermediate result is
last read. Specifically, for each node it records the *index* of the last operation that
uses its output as an input. Once execution reaches that index, the buffer is free.

Nodes whose output is never read by anything else in the graph — the root node, for
instance — get `end = None`.

### 3. Buffer assignment

This is the core of the planner. For each node in order, it picks one of five strategies:

**In-place reuse.** Scalar ops and binary tensor ops can overwrite one of their own
inputs. If an input's buffer is already free at this point and the sizes match, the
planner chooses this — no extra allocation, no cache lookup.

**Reference pass-through.** Layout-only ops (`View`, `Slice`, `Transpose`) don't produce
new data at all. They alias an existing buffer. The planner just extends the original
buffer's lifetime to cover all aliases and moves on.

**Redirect deduplication.** Some ops produce a transformed version of their input that
downstream consumers should use instead of the original — `AsContiguous` is the main
example, packing a non-contiguous tensor into a fresh contiguous buffer. When a second
`AsContiguous` node appears for the same input, the planner skips planning a new buffer
entirely: it extends the first node's slot lifetime to cover both sets of consumers and
records `duplicate_id → canonical_id` in a redirect table. No plan step is emitted for
the duplicate — the executor resolves its ID via the redirect table at runtime.

**Buffer reuse.** If a previously allocated buffer has been freed and has the right size,
the planner reclaims it. This is a linear scan over the current slot list.

**Fresh allocation.** If none of the above apply, a new `Vec<T>` is scheduled for
allocation at execution time.

After assigning all outputs, the planner populates the `dealloc_after` lists — each
buffer gets appended to the plan step at its `end` index, so the executor drops it
immediately after its last use.

---

## Execution

The executor in `TensorGraphNode::compute` (in `src/tensor/graph.rs`) works through the
plan one step at a time, maintaining a live-buffer cache keyed by node ID. After each
step it removes any IDs listed in `dealloc_after`, dropping the buffer as soon as it's
no longer needed.

The plan also carries a redirect table alongside the step list. When resolving inputs for
any step, the executor checks this table first: if an input's node ID has a redirect
entry, the canonical ID is used for the cache lookup instead. This is how deduplicated
`AsContiguous` nodes are served — they emit no step and hold no cache entry of their own,
but their consumers find the packed buffer transparently through the redirect table.

---

## Assumptions

A few things the planner takes for granted:

- **Shapes don't change.** All output shapes are computed when the promise is
  constructed. The plan bakes them in.
- **No cycles.** A cycle in the graph would cause the topological sort to loop forever.
- **Single-threaded execution.** The plan is regenerated on every `.materialize()` call.
  Cached nodes use `OnceLock` for thread safety, but execution itself is sequential.

---

## Performance

Planning time scales with graph size. For the common case — a linear chain of ops —
it's effectively linear and the overhead is negligible. However, if you're building very
large graphs with many nodes that all have distinct output sizes (preventing buffer
reuse), you may notice planning overhead growing faster than expected. The internal slot
search is O(n) per node in the worst case, making planning O(n²) overall for those
graphs.
