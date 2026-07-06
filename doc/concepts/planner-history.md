# Planner - Design History

This is an archival document. It records how the planner's buffer-reuse and
deduplication machinery evolved, and *why* each design was replaced. None of the
intermediate designs described here exist in the code anymore - if you want to know how
the planner works today, read [the planner docs](crate::docs::planner). Read this one if you're curious
why it works the way it does, or if you're tempted to "simplify" it back toward something
that already failed.

The story has four chapters. The first three each answered the previous one's flaw; the
fourth restructured the third's *correct* design rather than fixing a bug.

---

## 1. `reusable: bool`

The first design tagged each node with whether its buffer could be recycled. When an op
needed an output buffer, it grabbed a reusable input's buffer and wrote into it.

It broke on non-commutative ops. `a - b` and `a / b` care which operand is which, but the
flag only said "this buffer is free to reuse," not "this buffer is still being read." For
`Sub` and `Div` the reuse could pick the *wrong* input as the output and silently compute
`b - a`.

The lesson - "is this buffer reusable?" is the wrong question; "is it still being read at
this point?" is the right one - is the foundation of every later design and is told in
full in [the planner docs](crate::docs::planner). Answering it
required tracking *when* each buffer is last consumed, which means analysing lifetimes
before execution. That gave us a planner. It also gave us a new problem: deduplicating
`AsContiguous`.

---

## 2. The redirect table at execution time

`AsContiguous` packs a non-contiguous tensor into a contiguous buffer. When two branches
need a packed copy of the same input, doing it twice is wasteful - they should share one
buffer. The fix was a *redirect table*: the first `AsContiguous` over an input is
canonical, and later ones are recorded as `input → canonical` redirects.

The first implementation carried that table to execution time. Before each cache lookup,
the executor resolved every input ID through the table: if an ID had a redirect, the
canonical ID was used instead. Simple to reason about locally - and it introduced a
timing hazard.

Consider a `Transpose` (`node_1`) feeding two consumers: an `AsContiguous` (`node_2`,
which registers `node_1 → node_2`) and a plain scalar op (`node_3`). The topological sort
is a depth-first traversal with a LIFO stack; if `root.inputs = [node_2, node_3]`, it
yields:

```text
node_1, node_3, node_2
```

At execution, `node_3` runs *before* `node_2`. When it resolves its `node_1` input
through the global table, the redirect `node_1 → node_2` is already there - pointing at a
buffer that hasn't been computed yet. Panic.

The root problem: the redirect table was **global state applied to every lookup
regardless of when the lookup happened** relative to the redirect being registered.

---

## 3. Plan-time resolution

The next design - the one this rewrite grew directly out of - moved resolution to plan
time. The redirect table was built *on the fly* as nodes were planned in topological
order, and each plan step emitted its own `resolved_inputs` into the plan, consulting the
table as it stood at that moment. The canonical entry for `node_1 → node_2` was registered
*after* `node_2` emitted its own step, so `node_2` read its pre-redirect input and never
itself; `node_3`, planned before `node_2`, saw a redirect-free table and stored
`node_1`'s ID directly; steps planned after picked up the canonical ID. The affected
buffers' lifetimes were extended as the table grew. Execution order stopped mattering, and
the timing hazard was gone.

This was correct - and it is, almost exactly, what the planner does today. What pushed the
rewrite wasn't a resolution bug; it was structural. Aliasing wasn't a concept here, it was
*emergent* from three separate predicates run side by side in the planning pass -
`is_a_redirect`, `is_a_reference`, and `find_buffer_inplace` - each consulting different
state, each able to disagree. When `NoOp` was promoted to a redirect, the reference and
in-place checks didn't know to resolve *through* the redirect table, so a `View` over a
`NoOp` tried to reference a node that had been redirected away and owned no buffer. Crash.
The predicates shared no single notion of "what does this input actually resolve to" - and
that is the gap the rewrite closed.

---

## 4. Alias / takeover with per-node resolution snapshots (current)

The current design is chapter 3's resolution idea ported onto a dedicated pre-planner that
owns *all* aliasing, leaving a buffer-assignment pass that only allocates. The full
mechanism is in [the planner docs](crate::docs::planner); the parts that answer
"why restructure a correct design" are:

- **Resolution is a per-node snapshot taken in topological order.** Each node resolves its
  inputs at its own position in the sort and freezes the result - the same
  register-the-claim-after-the-node-emits rule chapter 3 used, now a property of *when* a
  node is visited in the pre-planner rather than a deferral threaded through the planning
  loop. A consumer before a takeover reads the original input, one after reads the
  claimer, and a takeover never resolves to itself.
- **One resolution path, not three predicates that can disagree.** `AliasKind` (`Alias`,
  `Takeover`, `NoAlias`) is the single classifier, and references, in-place reuse, and the
  executor all go through one `resolve`. Chapter 3's `NoOp`-breaks-`View` crash cannot
  recur: there is no second predicate left to disagree with the first about what an input
  resolves to. Adding an aliasing op is one classifier arm, nothing more.
- **The buffer-assignment pass never touches the alias map.** Aliasing is frozen by the
  time allocation runs, so a new alias kind cannot perturb allocation decisions.

The bug that pins this down is the same `transposed` / `as_contiguous` / `shifted` graph
that panicked under chapter 2. Under the current design it produces the correct result
regardless of sort order; there's a regression test guarding it.

