# Skeletons

Every `.materialize()` calls pays a planning cost - see [planner.md](planner.md) for more information.
So when you need to run complex models with hundred of chained ops and do backpropagation - not implemented yet! - on top of that, you get a humongous graph
and Candela has to plan for it. That, by itself, is not a problem as the planning time is going to be a very small fraction of the overall compute cost.

But you do the same computation, for hundreds, thousands of times. So, while planning is far from a computation bottleneck, it increases computational costs for a plan
that is, mostly, the same in those repeated computations. For this kind of cases Candela has `Skeleton`. It solves this exact situation, you bake a graph with some holes
and the output is a runtime function that already has the most optimized plan Candela can make that you can reuse as long the `Layout` of your inputs remains consistent.

```rust
use candela::{Layout, SkeletonSlot, Tensor};

// A slot is a typed hole: a layout with no data behind it.
let slot = SkeletonSlot::new(Layout::from_shape(&[4], 0));

// Ops over a slot build a graph exactly like ops over a tensor would -
// but the result is a SkeletonPromise, and it has no `.materialize()`.
let skeleton = (&slot * 2.0 + 1.0)
    .into_skeleton(std::slice::from_ref(&slot))?;

// Planning already happened. These calls only execute.
let a = skeleton.run(&[Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0], &[4])])?;
let b = skeleton.run(&[Tensor::from_scalar(5.0, &[4])])?;
assert_eq!(a.data(), &[1.0, 3.0, 5.0, 7.0]);
assert_eq!(b.data(), &[11.0; 4]);
```

This example demonstrates the pattern to create and run a `Skeleton`: (1) define slots, (2) define your computation and (3) bake your skeleton (`.into_skeleton`).
How this works will be explained in the next section, but, as I already said, you can imagine this is the equivalent of creating a runtime function that can reuse
computation resources if necessary.

# To Bake or not Bake

There are 3 types that you need to be aware: `SkeletonSlot`, `SkeletonPromise` and `Skeleton`. Each one is described in the section below:

## SkeletonSlot

A slot is just a hole that must be filled from outside the graph. They exist as markers and represent your intent to create a `Skeleton` in the future.
They can be operated as if they are real tensors but produce a `SkeletonPromise` instead of a `TensorPromise` and cannot be materialized.

```rust
use candela::{Layout, SkeletonSlot, Tensor};

// Real tensor
let tensor = Tensor::from_scalar(2.0, &[8]);

// A slot created from a tensor
let slot = tensor.as_slot();

// A slot created from thin air
let i_am_different = SkeletonSlot::new(Layout::from_shape(&[8], 0));
```

## SkeletonPromise

They are just sugar around a `TensorPromise` but they serve as gatekeepers that stop you from trying to do evil things, like materializing computations chains with holes.
Beyond being gatekeepers they store the computation chain up to that point, so, when you do `into_skeleton` from a `SkeletonPromise` it reads the operation chain and produces
a `Skeleton`. So, when you see a `SkeletonPromise` you know that is a future `Skeleton` being constructed and that any operation you do there is being recorded
so it can be replayed by the `Skeleton`.

```rust
use candela::{Layout, SkeletonSlot, Tensor};

let tensor = Tensor::from_scalar(2.0, &[8]);

// The slot
let slot = tensor.as_slot();

// Creating a slot from another
let slot2 = slot.deep_clone();

// The promise created from the slot
let promise = slot * 2.0 + 1.0;

// Creating a skeleton
let skeleton = promise.log2().into_skeleton(&[slot, slot2])?;
```