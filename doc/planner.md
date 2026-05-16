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
consume them. (A topological sort is any ordering of a DAG's nodes where every
dependency appears before the thing that depends on it. For a computation graph, that
just means: no op runs before its inputs are ready.) This is done by `topological_sort` in `src/tensor/planner/sort.rs` using
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

**Reference pass-through.** Layout-only ops (`View`, `Slice`, `Transpose`) produce no
new data at all. They alias an existing buffer. The planner extends the original buffer's
lifetime to cover all aliases and emits a plan step with no real computation.

**Redirect deduplication.** Some ops produce a canonical transformed version of their
input — `AsContiguous` being the main example. If a duplicate of that op appears later
for the same input, no buffer is planned for it; the planner resolves its ID, so that
the executor does not have to worry . See the next section for the full story on
when this fires and what to watch out for.

**Buffer reuse.** If a previously allocated buffer has been freed and has the right size,
the planner reclaims it. This is a linear scan over the current slot list.

**Fresh allocation.** If none of the above apply, a new `Vec<T>` is scheduled for
allocation at execution time.

After assigning all outputs, the planner populates the `dealloc_after` lists — each
buffer gets appended to the plan step at its `end` index, so the executor drops it
immediately after its last use.

---

## Redirect deduplication

When an op packs a non-contiguous tensor into a fresh contiguous buffer — matmul's BLAS
path is the main consumer (BLAS is the standard library of optimized linear algebra
routines; it requires contiguous row-major input) — it would be wasteful to do that work twice if two branches of
the same graph both need a packed version of the same input. The redirect mechanism
handles this: the first `AsContiguous` over a given input is registered as *canonical* and
planned normally. Any later `AsContiguous` over the *same* input node (same `Arc`, same
ID) within the same planning pass emits no plan step and no buffer — it's recorded in a
redirect table instead, and the executor transparently serves the canonical buffer to all
its consumers.

A few things to know before you rely on this.

**Deduplication requires a shared node, not shared data.** Two `AsContiguous` ops
collapse only if they wrap exactly the same `Arc<TensorGraphNode>` — the same pointer,
the same ID. If two parts of the graph each build their own path to "the same data" (say,
by calling `.as_promise()` on the same tensor twice), they produce separate nodes with
separate IDs and each gets its own buffer. There is no deduplication across separately
constructed graph paths.

**DFS order picks the canonical node, and it matters across passes.** The redirect table
is rebuilt on every `.materialize()` call. Whichever `AsContiguous` the DFS visits first
becomes canonical for that pass. If the canonical node is ephemeral — not backed by a
`CachedTensorPromise` — it re-allocates and re-computes on every subsequent call even
after a cached version of the same data is warm. To guarantee the cached buffer is always
canonical, pass the `CachedTensorPromise` directly as the input to downstream ops:

```
// Suboptimal: both AsContiguous ops target raw_promise.id.
// They deduplicate within a pass, but if the ephemeral one wins the DFS
// it re-allocates on every call even after the cache is warm.
let cached = raw_promise.clone().cache();
let result = (op_a(&cached) + op_b(&raw_promise)).materialize();

// Better: all downstream ops take the cached promise directly.
// Once warm, AsContiguous(filled_cache) is immediately recognized as
// redundant — no allocation, no plan step, regardless of DFS order.
let cached = raw_promise.cache();
let result = (op_a(&cached) + op_b(&cached)).materialize();
```

**Filled caches as direct inputs skip the ordering question entirely.** When
`AsContiguous`'s direct input is an already-filled `CachedTensorPromise`, the planner
doesn't wait for another `AsContiguous` to appear first. Filled caches always store a
contiguous result (see [graph.md](graph.md)), so packing is pointless regardless of what
else is in the graph or what order the DFS visits things. The cache is immediately
recorded as the redirect target — no plan step, no allocation.

**Same-ID edge case.** The sort deduplicates nodes by ID, and `NodeKind::Cache` is keyed
on its *inner* node's ID. If a `NodeKind::Cache` and a `NodeKind::Node` share the same
inner ID — possible when constructing graphs manually with cloned nodes, but not
reachable through the public `.cache()` API — only the first one the DFS encounters is
planned; the other is silently dropped. The regular node tends to win because it appears
as an ancestor inside the cache's own subtree and is therefore encountered first during
depth-first traversal. Avoid constructing graphs where this can happen.

---

## Execution

The executor in `TensorGraphNode::compute` (in `src/tensor/graph.rs`) works through the
plan one step at a time, maintaining a live-buffer cache keyed by node ID. After each
step it removes any IDs listed in `dealloc_after`, dropping the buffer as soon as it's
no longer needed.

Redirect resolution is entirely a plan-time concern. Each step's `resolved_inputs`
already holds the canonical IDs — any node whose input would have been redirected sees
the canonical ID baked in. The executor never consults a redirect table; it reads
`resolved_inputs` and looks up those IDs in the live-buffer cache directly. Deduplicated
`AsContiguous` nodes emit no plan step at all; their consumers receive the canonical ID
through `resolved_inputs` without any special handling at execution time.

---

## Why redirect resolution moved to plan time

The redirect mechanism went through one design iteration worth understanding, because
the problem it solved is subtle.

The original design carried the redirect map alongside the plan at execution time. The
executor would resolve every input through the map before each cache lookup: if a node
ID had a redirect entry, the canonical ID was used instead. Simple to reason about
locally — but it introduced a timing hazard.

Consider a graph where a `Transpose` node (`node_1`) feeds two consumers: an
`AsContiguous` node (`node_2`, which registers the redirect `node_1 → node_2`) and a
plain scalar op (`node_3`). The topological sort is a depth-first traversal with a LIFO
stack. If `root.inputs = [node_2, node_3]`, `node_3`'s subtree is popped and explored
first, yielding the execution order:

```
node_1, node_3, node_2
```

During planning, `node_2` is processed last and inserts the redirect `node_1 → node_2`.
At execution time, `node_3` runs before `node_2`. When it resolves its `node_1` input
through the global table, the redirect is already there — pointing to `node_2`'s buffer,
which hasn't been computed yet. Panic.

The root problem: the redirect table was global state, applied to every input lookup
regardless of when the lookup happened relative to the redirect being registered.

The fix moves resolution to plan time. `build_resolved_inputs` is called when each plan
step is emitted, consulting `id_redirect` as it exists at that exact moment. The redirect
for `node_1 → node_2` is inserted into the map *after* `node_2`'s step is emitted (via a
`pending_redirect_from` that fires at the end of `plan_node`). So `node_3`, planned
before `node_2`, sees a redirect-free map and stores `node_1`'s ID directly in its
`resolved_inputs`. Steps planned after the redirect is registered pick up the canonical
ID. Execution order no longer matters — each step carries exactly what it needs.

This design also keeps the door open for parallel execution. A runtime redirect table
would be shared mutable state across threads; with redirects baked into each step's
`resolved_inputs`, steps are self-contained and can be dispatched independently. When
GPU execution arrives, the plan steps can be distributed without shared bookkeeping.

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
