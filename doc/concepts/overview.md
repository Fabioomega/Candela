# The Pipeline

Nothing in Candela computes until you ask. As you build an expression on a
[`Tensor`](crate::Tensor), each operation records itself into a graph - allocating
nothing, running nothing. The numbers appear only when you call `.materialize()`,
which plans that graph and then executes it.

## Building - the graph

Every operation you write links back to the operations it came from, and together
they form a graph that reference counting keeps alive. The leaves are the tensors
you fed in; the inner nodes are the operations. Each op returns a
[`TensorPromise`](crate::TensorPromise) rather than a tensor, so nothing has run
yet - the graph is only a record of what you asked for.

Some operations merge as the graph is built. When a sequence of operations can be
carried out in a single pass over the data, they are fused into one node, so the
stored graph does fewer, larger steps than the one you wrote. A chain of scalar
operations folding into a single pass is the simplest case, but fusion is not
limited to scalars. The [computation graph](crate::docs::graph) covers which
operations fuse and how.

## Planning - the schedule

`.materialize()` does not walk the graph and start computing. It first hands the
graph to the *planner*, which produces a fully ordered schedule: what to compute,
in what order, which buffer each result writes into, which buffers to free after
each step, and which buffer holds the final answer.

This is where Candela earns its memory behaviour. The planner tracks when each
buffer is last read, so it can reclaim a buffer the instant it is dead, compute a
shared subexpression exactly once, and never allocate an intermediate it does not
need. A long chain runs in roughly the peak of what is live at once, not the sum of
every intermediate it passes through. How the schedule is built is the subject of
[the execution planner](crate::docs::planner).

## Execution - running the plan

The executor walks the finished plan one step at a time, holding live results in a
small cache and dropping each the instant the plan says it is no longer needed. The
one thing the executor does not do itself is the arithmetic: it hands each op to the
*backend*, which owns the compute functions (kernels) that produce the numbers.

---

## The two systems underneath

**Layout - what a tensor is.** A tensor is a flat buffer paired with a `Layout`. The
buffer holds the numbers in the order they sit in memory; the `Layout` - a shape,
strides, and an offset - describes the order you actually index them in. Separating
the two is what lets reshape, slice, transpose, and broadcast return instantly: they
rewrite the `Layout`, changing how the same buffer is read, and copy nothing. A copy
happens only when an operation genuinely needs its elements physically in order, and
when it does the graph makes that copy an explicit node. The details are in
[memory layout](crate::docs::layout).

**Backend - where it runs.** A tensor is `Tensor<T, B>`. The dtype `T` (`f32`,
`f64`) says what the numbers are; the backend `B` says where and how they compute.
Candela supports more than one backend - the default is pure Rust, and one of the
others is Intel MKL on the CPU. [Backends and dtypes](crate::docs::backends) covers
the split and how to add your own.

## Reusing a plan - skeletons

Planning is cheap next to computing, but if you run the *same* expression on new
data thousands of times, re-deriving the same plan on every call is wasted work. A
[`Skeleton`](crate::skeleton::Skeleton) is the graph with its plan already frozen:
you build it once over placeholder slots, then feed it inputs and it only executes.
A plain skeleton is fixed to one input layout; a
[`DynamicSkeleton`](crate::skeleton::DynamicSkeleton) lifts that restriction by
caching one skeleton per input shape it encounters and building a new one on a miss.
Both are described in [skeletons](crate::docs::skeleton).
