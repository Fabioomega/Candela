# Execution Planner

`.materialize()` does not execute the graph directly - it first builds a *plan*, then runs
it. This document covers how that plan is built; the planner works on the node types from
[the computation graph](crate::docs::graph). For how the design got here - the dead ends and
the bugs that shaped it - see [the design history](crate::docs::planner_history).

---

## Why not just execute the graph directly?

Earlier versions used a `reusable: bool` flag to decide whether a buffer could be recycled.
It worked until it didn't: for non-commutative ops like `Sub` and `Div`, the flag could pick
the *wrong* input buffer as the output, causing `a - b` to silently compute `b - a`.

The root problem was that "is this buffer reusable?" is the wrong question. The right
question is "is this buffer *still being read* at this point?" - which requires knowing
*when* each buffer is last consumed, not just whether it is eligible. The planner answers
that before execution begins.

---

## The plan at a glance

Before the details, here is the whole pipeline at a high level. Planning a graph is
five steps:

1. **Order the nodes.** A topological sort places every node after the inputs it
   depends on, so nothing is ever scheduled before the data it reads.
2. **Classify aliasing.** Some nodes produce a buffer of their own; others only
   re-point at a buffer another node already owns. This step decides which is which.
3. **Assign each output a buffer.** For every node that does produce one, pick how:
   overwrite a dead input in place, reclaim an earlier buffer that has since been
   freed, reference an existing one, or allocate a fresh buffer.
4. **Record lifetimes.** Note the last step that reads each buffer, so it can be
   freed the moment nothing needs it again.
5. **Execute.** Walk the ordered steps, running each op into its assigned buffer and
   dropping buffers as their lifetimes end. Whatever holds the root's result is the
   answer.

---

## Example

Take `(a + b) * 2.0`. The graph has four nodes: the two input tensors `a` and `b`, an
`Add` that combines them, and a scalar `× 2.0` over the sum. Planning walks them in
order:

- `a` and `b` are inputs - buffers the planner never allocates and never frees, they
  already exist.
- `Add` has no dead buffer to overwrite (its inputs are the caller's tensors, which
  have to survive), so it allocates one fresh buffer for the sum.
- `× 2.0` reads `Add`'s buffer, and `Add` has no other consumer, so that buffer is
  dead the instant the scalar has read it. Rather than allocate, the scalar writes its
  result back into that same buffer, in place.
- The scalar is the root, so its buffer is the result and is kept.

Written out as a schedule:

```text
  a ─┐
     Add ── ×2.0     (root)
  b ─┘

  step 0   Add    → buf0 (fresh)       a, b are inputs, kept alive
  step 1   ×2.0   → buf0 (in place)    buf0 is dead after Add, so reuse it
                                       buf0 holds the final result
```

The whole expression runs in a single allocation, reused for the final step - no
intermediate is ever held past the moment it is consumed. The rest of this document is
how the planner arrives at decisions like that one.

---

## How a plan is built

The planner runs in two passes over the graph, followed by a small fixup.

### Pass 1 - the pre-planner

A single walk in topological order (`pre_plan` in `plan.rs`) does four things at once, for
every node:

1. **Orders it.** The walk is driven by `topological_sort` (`sort.rs`), an iterative
   post-order DFS - inputs before the ops that consume them, no recursion so deep graphs
   don't overflow the stack. Shared nodes are deduplicated by ID, so a node feeding two
   parents appears once. A *filled* `CachedTensorPromise` is treated as a leaf: its subtree
   isn't traversed. The root node is not yielded - it is handled at the end of the pass.

2. **Classifies its aliasing.** Each node is sorted into an `AliasKind` - `Alias`,
   `Takeover`, or `NoAlias` - which decides whether it produces a buffer at all. This is the
   heart of the planner and gets its own section below.

3. **Snapshots its resolved inputs.** For each input, the planner records *the node that
   actually produces that input's buffer*, as the alias map stands at this point in the
   walk. The snapshot is frozen onto the node and never recomputed. This is what makes
   resolution independent of execution order - again, more below.

4. **Records its lifetime.** It notes the index of the last step that reads the node's
   output. Once execution passes that index, the buffer is free. A node whose output is
   never read again - the root, for instance - gets `end = None`.

### Pass 2 - buffer assignment

Now the planner walks the staged nodes in order and, for each, picks how the executor should
produce its output buffer. This is `classify` in `runtime.rs`, returning one of five
`ExecKind` strategies:

**In-place reuse** (`InPlace`). Scalar ops and binary tensor ops can overwrite one of their
own inputs. If an input's buffer is already free at this point and the sizes match, the
planner writes the result straight into it - no allocation, no copy.

**Reference, slot-backed** (`ReferenceSlot`). Layout-only ops - `View`, `Slice`,
`Transpose`, `TransposeAxes`, `Broadcast`, and `NoOp` - re-point an existing slot-backed
buffer at a new layout; only the layout descriptor changes, no elements are copied. The
buffer stays shared - other nodes may still read it through their own layouts - so the
planner extends its lifetime to cover the reference.

**Reference, eternal** (`ReferenceEternal`). The same, but the aliased input is an edge or a
cache buffer - things the planner never frees - so there is no slot lifetime to extend. This
also covers reference *chains*: when a layout-only op aliases another layout-only op that
itself bottomed out at an edge or cache (e.g. matmul's `edge → View → Broadcast`), the inner
node owns no slot, so the outer op is eternal too.

**Buffer reuse** (`UseSlot`). If a previously allocated buffer has been freed and is the
right size, the planner reclaims it. This is a linear scan over the live slot list.

**Fresh allocation** (`Allocate`). If none of the above apply, a new `Vec<T>` is scheduled
for allocation at execution time.

`AsContiguous` straddles two of these: if its input is already contiguous it is a reference
(no packing needed); otherwise it reuses a freed buffer or allocates one to pack into.

### Fixup - deallocation lists

After every output is assigned, the planner walks the slots and appends each buffer's ID to
the `dealloc_after` list of the plan step at its `end` index. The executor drops a buffer the
instant the step that last needed it completes.

---

## Aliasing: alias vs takeover

This is the core of the planner, and where two earlier designs went wrong (see
[the design history](crate::docs::planner_history)).

Some nodes don't produce a buffer; they *are* another node's buffer. `NoOp` is the obvious
case: it is the identity, so its result is exactly its input's buffer. `AsContiguous` is
subtler - it packs a non-contiguous input into a fresh contiguous buffer, but if two branches
of the graph both pack the *same* input, the second packing is wasted work. They should share
one buffer.

The planner handles both with an **alias map**: a `node id -> owning node` table (`AliasMap`
in `alias.rs`). `resolve(input)` turns an input into the node that actually produces its
buffer; absence from the map means a node produces its own. A node enters the map one of two
ways.

### Alias - "I am someone else's buffer"

An `Alias` node contributes no computation and emits no plan step. It records
`node.id -> target` and disappears. A `NoOp` aliases its input. A *second* `AsContiguous` over
an input that's already been packed aliases the first one's result. An `AsContiguous` over a
cache node aliases the cache directly (caches store contiguous results - see
[the computation graph](crate::docs::graph)). Aliases point *backward*, to a node already
visited earlier in the sort, so resolving one always lands on something real.

### Takeover - "I am now the canonical version of my input"

A `Takeover` node *is* computed - it is the first `AsContiguous` over its input, the thing
that does the packing - but it also **claims** its input. The claim means: from here on,
anyone who wanted that input should use *me* instead, because I'm the contiguous version of
it. The same machinery backs cache nodes claiming their inner computation.

Claiming is more than inserting one entry. If something already aliased the claimed node,
that alias is now stale - it should point at the claimer too. So `takeover` rewrites *every*
entry pointing at the claimed node to point at the new owner, then points the claimed node
itself at the new owner. The map stays **single-hop**: after a takeover nothing points at the
old node, so `resolve` is always one lookup, never a chain.

### The `Tag` - telling "already done" from "needs claiming"

How does the planner know whether a second `AsContiguous` should *alias* the first (dedup) or
*take over* (because the existing alias isn't actually contiguous)? Each alias entry carries a
`Tag` describing what its target guarantees: `Anything` (same data, no layout promise - a
`NoOp`), `AsContiguous` (a contiguous packing), or `AsContiguousCache` (a contiguous packing
that also survives across materializations). A later `AsContiguous` deduplicates only when the
existing alias already promises contiguity; over a plain `NoOp` alias (`Anything`) it takes
over instead.

### Why resolution is a per-node snapshot

Here is the subtle part. A `Takeover` reaches *backward* to claim its input - but consumers
of that input are scattered across the graph, some sorted *before* the takeover and some
*after*. They can't be treated the same:

- A consumer planned **before** the takeover must read the *original* input. The claimer
  hasn't been computed yet at that point in execution; pointing the consumer at it would be a
  read of a buffer that doesn't exist.
- A consumer planned **after** the takeover should read the claimer - that's the whole point
  of deduplication, and by then the claimer's buffer is live.

The planner gets this right for free by **resolving each node's inputs at the moment that node
is reached in the sort, and freezing the result.** The alias map is built incrementally in the
same walk, so a node sees only the claims registered *before* its own position:

- Consumers sorted before the takeover froze their resolution while the map still pointed at
  the original input. They read the original. Correct.
- Consumers sorted after see the rewritten entry and resolve to the claimer. Correct.
- The takeover node resolves its *own* inputs **before** registering its claim, so it never
  resolves to itself - it reads the input it is about to pack, exactly as it should.

The backward rewrite (`takeover` fixing up prior entries) only affects consumers planned
*after* the claim; earlier consumers already froze their inputs and never consult the map
again. So "frozen at my position in the sort" hands every consumer the one correct answer with
no special cases and no ordering hazard.

A worked example. `transposed` (a `Transpose`) feeds two consumers:
`contiguous = transposed.as_contiguous()` and `shifted = &transposed + 1.0`. The sort, driven
by a LIFO stack, can yield `shifted` *before* `contiguous`:

```text
transposed, shifted, contiguous
```

When `shifted` is reached, no claim on `transposed` exists yet, so its frozen input is
`transposed` - and at execution it reads the transpose, which is what it wants. When
`contiguous` is reached it takes over `transposed`; a *later* consumer would resolve to
`contiguous`. Nobody reads a buffer before it exists. (This exact graph used to panic -
[the design history](crate::docs::planner_history) tells that story.)

### What deduplication can and can't do

- **It needs a shared node, not shared data.** Two `AsContiguous` ops collapse only if they
  wrap the same `Arc<TensorGraphNode>` - the same ID. If two parts of the graph each build
  their own path to "the same data" (e.g. by calling `.as_promise()` on the same tensor
  twice), they produce separate nodes with separate IDs and each gets its own buffer.

- **DFS order picks the canonical node, and the map is rebuilt every `.materialize()`.**
  Whichever `AsContiguous` the sort reaches first becomes the `Takeover`; if that node is
  ephemeral (not backed by a `CachedTensorPromise`) it re-packs on every call. An
  `AsContiguous` over a cache input is recognized as an `Alias` immediately, regardless of DFS
  order, so a cached buffer stays canonical.

---

## The root node

The node `.materialize()` was called on is special: nothing consumes its output, and the sort
doesn't yield it. The planner classifies it at the end of the pre-planner.

Usually it is ordinary computation and becomes the last plan step; the result lands in the
live-buffer cache under the root's own ID. But if the root is a *pure alias* - say
`x.as_contiguous()` where `x` is already contiguous - it produces no step at all, and the
result is its target's buffer. `Plan::root_id` records which ID that is; the planner forces
that target's lifetime to `None` so it survives to the end instead of being reclaimed, and the
executor returns whatever sits under `root_id`.

---

## Execution

The executor (`run_plan` in `executor.rs`) works through the plan one step at a time, keyed by
node ID, dropping each buffer as its `dealloc_after` entry comes due.

Alias resolution is entirely a plan-time concern: each step's `resolved_inputs` already holds
the canonical IDs, so a node whose input was aliased or taken over sees the resolved ID baked
in. The executor never consults the alias map - aliased nodes emit no step at all, and their
consumers received the right ID at plan time.

That also keeps the door open for parallel execution: there is no shared mutable resolution
table to coordinate, so every step is self-contained and can be dispatched independently when
GPU execution arrives.

---

## Assumptions

A few things the planner takes for granted:

- **Shapes don't change.** All output shapes are computed when the promise is constructed. The
  plan bakes them in.
- **No cycles.** A cycle in the graph would cause the topological sort to loop forever.
- **Single-threaded execution.** The plan is regenerated on every `.materialize()` call. Cached
  nodes use `OnceLock` for thread safety, but execution itself is sequential.

---

## Performance

Planning is `O(n²)` in the worst case: the buffer-reuse search (`UseSlot`) is a linear scan
over the live slot list, run once per node.
