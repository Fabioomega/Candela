# Candela

A lazy, graph-based tensor engine written in Rust.

Candela is a learning project — a JAX-like tensor engine, inspired by Candle — meant to shed light on the almost magical and complex world of tensor engines. It features `f32` and `f64` CPU support through a modular backend, with operator fusion and optional BLAS support.

---

## Installation

```bash
cargo add candela-tensor
```

---

# Get started

Here's a quick example of computing a matrix multiplication with Candela. You first define the operation chain — called a promise chain — and when everything is set up you call `.materialize()`, and the tensor becomes real.

```rust
use candela::{Tensor, srange};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Tensor<f32> = srange![2 * 3, &[2, 3]];
    println!("a: {a}");

    let b: Tensor<f32> = srange![3 * 2, &[3, 2]];
    println!("b: {b}");

    let c = a.matmul(&b)?.materialize();

    println!("c: {c}");
    Ok(())
}
```

You can find more at [examples](examples/), but the basic idea is: you define a plan for how things should be computed, and then you execute it! No allocation or computation happens before you give the go-ahead — that's why Candela is _lazy_. The neat part is that you get shape errors *before* anything runs, so you never waste compute on a graph that was never going to work.

---

## Core Philosophy

When Candela started, I wanted it to be different — something that would let me explore new ideas. While searching for inspiration (see the Inspirations section), I came to realize that the Rust ecosystem was lacking a truly "rusty" tensor engine.

Everyone just tries to be [PyTorch](https://github.com/pytorch/pytorch): a nice, easy-to-use API with great performance. The price for that is that *you never truly know what's happening under the hood*. So I wanted another approach — one where the end user understands the actual costs they're paying, and maybe even makes those costs predictable (embedded programming, I'm looking at you!).

My thought process was: **you should own your allocations** — that's the Rust way! And if you're crazy enough, or desperate enough, to try a Rust tensor library, I imagine that's not a big deal for you. So I made caching a **user choice**: nothing in the library will silently allocate memory and hold onto it without asking you first. Of course, once you let Candela loose it does the best it can to be performant — and that involves memory reuse and a lot of planner magic to make your graph cheaper to compute.

In the future, Candela will be able to tell you how many resources a computation needs before it even runs. All the information is already there!

---

## The Three Tensor Types

Candela is structured around three tensor types: `Tensor<T>`, `TensorPromise<T>`, and `CachedTensorPromise<T>`. Together they draw a hard boundary between what's actually real and what's still a promise — a future computation waiting to happen. Here's a brief look at each:

### `Tensor<T>`

A materialized tensor. It holds data and is, more or less, an abstraction over a `Vec`. When you have one, you know the data is ready and there.

```rust
let t = Tensor::from_scalar(1.0, &[3, 3]); // 3x3 matrix filled with 1.0
```

### `TensorPromise<T>`

A `TensorPromise<T>`, or just a promise, is a computation chain. It stores what should be computed and how. When you're ready for the plan to become real, you call `.materialize()`, and it produces a `Tensor<T>` holding your output.

```rust
let t = Tensor::from_scalar(1.0, &[3, 3]);
let mut p = t + 5.0;           // ops return a TensorPromise automatically
p *= 2.0;                      // continue building the graph
let result = p.materialize();  // execute the graph now
```

To make sure everything runs properly, we apply a topological sort to the computation graph and reuse buffers wherever possible. Intermediate computations are eagerly freed once they're no longer needed and can't be reused. So in the end you get exactly what you started with, plus the final result — nothing more, nothing less.

### `CachedTensorPromise<T>`

This one is closer to the everyday `Tensor` you see in other libraries. It combines a promise's ability to hold a computation chain with a `Tensor<T>`'s ability to hold data, which lets you reuse intermediate computations across separate `.materialize()` calls.

One example: you want to inspect a result in the middle of a chain, but you don't want to recompute everything twice.

```rust
let a = Tensor::from_scalar(1.0, &[3, 3]);

let b = (a + 2.0).cache();          // b will cache its result once it's computed
println!("{}", b.snapshot());       // peek at the intermediate value — this computes it and fills the cache

// When this runs, we reuse the value of b instead of computing it again
let c = (b * 10.0).materialize();
println!("{}", c);
```

You pay the memory cost of keeping the cache alive, which is why this is opt-in.

---

## Features

- Lazy evaluation via computation graph (DAG)
- Scalar operator fusion - long chains collapse into single-pass kernels
- Zero-copy views (see [the layout docs](https://docs.rs/candela-tensor/latest/candela/docs/layout/))
- Memory reuse across long chains (see [the planner docs](https://docs.rs/candela-tensor/latest/candela/docs/planner/))
- Pluggable CPU backends - pure-Rust by default, Intel MKL behind `--features mkl`
- `f32` and `f64` element types via the `Backend`/`Dtype` split
- Matrix multiplication, including batched and batch-broadcast cases
- Axis reductions - `sum`, `mean`, `max` (with `keepdim`)
- Activation ops - `relu`, `tanh` (more landing incrementally)
- Full stride/offset layout system for non-contiguous tensors
- Opt-in result caching via `CachedTensorPromise`
- Plan-once / run-many skeletons - pre-planned graphs over slot placeholders
  (see [the skeleton docs](https://docs.rs/candela-tensor/latest/candela/docs/skeleton/))
- Built-in `tracing` instrumentation (feature-gated)
- `arange!`, `srange!`, `zeros!`, `ones!` convenience macros

---

## Error Handling

Here Candela has a nice feature — though a divisive one. Inline operators (`+`, `-`, `*`, `/`) **panic** on shape mismatches. This is intentional: the alternative would force you to litter every expression with `.unwrap()`, and a shape mismatch here is almost certainly a programming error anyway. Anything else that can fail returns a `Result` instead.

On top of that, you get the error at graph-construction time rather than during materialization. That sidesteps a problem some tensor libraries have, where the computation is rewritten so heavily behind the scenes that it's hard to tell **where** something actually went wrong.

```rust
let result = a.matmul(&b)?;       // Result<TensorPromise<T>, OpError>
let reshaped = p.view(&[4, 3])?;  // Result<TensorPromise<T>, OpError>
```

---

## Current Limitations

- **Data types:** `f32` and `f64` are the only CPU-supported data types for now. The generic framework is written against a `NumberLike` trait, so other element types just need a backend implementation.
- **GPU:** No GPU support yet, but the whole execution flow is designed to make it possible down the line.

---

## Internals

If you want to understand how things work under the hood, the design is documented on docs.rs:

- [overview](https://docs.rs/candela-tensor/latest/candela/docs/overview/) - the whole pipeline, from expression to computed tensor
- [the computation graph](https://docs.rs/candela-tensor/latest/candela/docs/graph/) - node types, sharing, and how ops fuse during construction
- [the execution planner](https://docs.rs/candela-tensor/latest/candela/docs/planner/) - how Candela decides what to compute, in what order, and which buffers to reuse
- [memory layout](https://docs.rs/candela-tensor/latest/candela/docs/layout/) - strides, zero-copy views, and the `adj_stride` iteration trick
- [backends](https://docs.rs/candela-tensor/latest/candela/docs/backends/) - the backend/dtype split and the `mkl` feature flag
- [skeletons](https://docs.rs/candela-tensor/latest/candela/docs/skeleton/) - compile a graph once, run it many times, and how the frozen plan is reused

---

## Roadmap

See the — lovely — [ROADMAP.md](ROADMAP.md) (written with a lot of AI assistance) for the full plan: phases, rationale, tests, and what comes next.

---

## Building

The default build is pure Rust, no system dependencies, builds on any target Rust supports:

```bash
cargo build
cargo test --doc            # run the doctests
cargo run --example fusion  # run an example (see examples/)
```

To use the Intel MKL backend instead, enable the `mkl` feature. This links against MKL, so the libraries need to be available on your system (the `intel-mkl-src` crate handles the linking):

```bash
cargo build --features mkl
```

---

## Why "Candela"?

Like the SI unit [candela](https://en.wikipedia.org/wiki/Candela), a measure of "luminous intensity", my hope is that this library sheds some light on a very opaque and complicated topic: *tensor engines*. Especially for me — I'm an undergraduate taking on a huge project, 6+ months in the making, that I started with only an AI and math background. Which turned out to be nowhere near enough.

I chose the name because two amazing projects follow the same theme: **[Candle](https://github.com/huggingface/candle)** and **[Burn](https://github.com/tracel-ai/burn)**. At this point I think every AI project in the community is contractually obligated to have a fire- or light-related pun somewhere.

But beyond broadening my own understanding, I hope to share my ideas — and a deeper dive into how tensor engines are built, and how anyone crazy enough could build one too. **[Tensorken](https://github.com/kurtschelfthout/tensorken)** does an excellent job of showing how they work; I just wanted to go in a different direction.

---

## Inspirations

These projects provided massive inspiration throughout the development of Candela:

- **[Candle](https://github.com/huggingface/candle)** - Hugging Face's minimalist ML framework in Rust. A proof that you don't need Python to do serious ML.
- **[Burn](https://github.com/tracel-ai/burn)** - A full-featured deep learning framework in Rust with a thoughtful design around backends and autodiff.
- **[Tensorken](https://github.com/kurtschelfthout/tensorken)** - A key source of inspiration for thinking about tensor graphs and lazy execution.

You may call Candela "JAX-like", and you'd be right — the lazy, fused, functional flavor lines up almost perfectly. But I can't honestly list [JAX](https://github.com/jax-ml/jax) as an inspiration: I only stumbled onto it about a month ago, well after these ideas had already taken shape; I'm a PyTorch guy after all. I'd call it convergent evolution rather than borrowing.

---

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

---

## Author

Made by **Fabio** ([@Fabioomega](https://github.com/Fabioomega)).
