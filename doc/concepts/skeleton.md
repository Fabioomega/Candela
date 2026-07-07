# Skeletons

A skeleton is a compiled computation. Like a function, it is built once and run many times
on new inputs; unlike an ordinary promise chain, it removes the planning. Every
`.materialize()` re-plans its graph from scratch - a skeleton plans once, freezes the
result, and every later run only executes it. That saved planning is the whole reason a
skeleton exists.

```rust
use candela::skeleton::SkeletonSlot;
use candela::{Layout, Tensor};

// Build the plan once, over a slot standing in for the input...
let slot = SkeletonSlot::new(Layout::new(&[4]));
let skeleton = (&slot * 2.0 + 1.0).into_skeleton(std::slice::from_ref(&slot))?;

// ...then run it repeatedly, with no planning in between.
let a = skeleton.run(&[&Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4])])?;
let b = skeleton.run(&[&Tensor::from_scalar(5.0, &[4])])?;
assert_eq!(a.data(), &[1.0, 3.0, 5.0, 7.0]);
assert_eq!(b.data(), &[11.0; 4]);
# Ok::<(), candela::OpError>(())
```

---

## Baking

[`into_skeleton`](crate::skeleton::SkeletonPromise::into_skeleton) runs the graph through
the ordinary planner - the same pass `.materialize()` uses, described in
[the execution planner](crate::docs::planner) - and then converts the resulting borrowed
plan into an owned form held by the [`Skeleton`](crate::skeleton::Skeleton). Planning
happens exactly once, here. Alongside the plan the skeleton stores its *declared slots*:
each slot's graph-unique ID and the `Layout` it was declared with.

A [`SkeletonSlot`](crate::skeleton::SkeletonSlot) is a layout-only node with no data behind
it. In the plan it appears as an external input - a hole the plan reads from but never
computes. The order the slots are declared in is the order later inputs are matched to
those holes.

The public interface refuses to materialize any graph with a slot in its lineage - such a
graph is not meant to exist, and would panic if one were forced through at runtime, since a
slot has no data to compute.

---

## Running

[`run`](crate::skeleton::Skeleton::run) does no planning. It checks the inputs against the
declared slots - their count and their exact `Layout` (shape, stride, and offset) - then
feeds each input's buffer into the plan under its slot's ID and executes. The check
guarantees the inputs are compatible with the frozen plan, which expects every layout to be
known at plan time; a mismatch is rejected rather than silently repacked.

---

## Composing

[`compose`](crate::skeleton::Skeleton::compose) binds inputs to a skeleton but, instead of
executing, produces a [`BakedPromise`](crate::skeleton::BakedPromise): the frozen plan
wrapped as a single opaque node that can sit inside a larger graph. To the outer planner it
is one unit - the node's inputs are computed, then the inner plan runs - and the inner plan
is sealed, so outer fusion never reaches into it.

That seal costs memory reuse: a composed skeleton reuses buffers worse than materializing
the equivalent raw chain would, because the planner cannot reclaim buffers across the
boundary. It is a convenience for extending a chain that comes out of a skeleton, not the
efficient path. Unlike `run`, the inputs to `compose` may be any non-slot operand: a
`Tensor`, a `TensorPromise`, or another `BakedPromise`.

---

## Knowing the cost up front

Because the plan is fixed, a skeleton knows every allocation it will make before it runs.
[`memory_report`](crate::skeleton::Skeleton::memory_report) walks the stored plan and
returns a [`MemoryMetrics`](crate::skeleton::MemoryMetrics) - peak memory, number of
allocations, individual buffer sizes, and the output size. The figures are Candela's own
accounting and do not model reuse by the system allocator, so they describe what the plan
asks for rather than what the operating system ultimately does.

Each run currently allocates its buffers afresh. Since the plan already enumerates every
allocation, reusing them across runs through a per-skeleton buffer pool is a planned
addition.

---

## Dynamic skeletons

A [`Skeleton`](crate::skeleton::Skeleton) is fixed to one set of input layouts. A
[`DynamicSkeleton`](crate::skeleton::DynamicSkeleton) lifts that limit by holding a cache of
skeletons keyed by input layout - a hashmap wrapper with a custom eviction policy, building
a new skeleton through a supplied function whenever an unseen shape arrives. It is built
entirely on the public API, so it doubles as a worked example of extending skeletons and as
a base for custom caching strategies.
