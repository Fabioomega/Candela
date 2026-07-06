# Backends

For most of Candela's life there was no such thing as a "backend." There was *the*
implementation: Intel MKL, linked unconditionally, specialized for `f64`. It was fast and
it worked - on an Intel machine, for one element type. Everywhere else it was a wall. The
crate wouldn't build on ARM (Apple Silicon, a Raspberry Pi, Graviton), and adding `f32`
meant copy-pasting the whole compute layer with the types swapped.

The backend split exists to break those two couplings at once: *where* a computation runs
should be independent of *what* it runs on, and neither should be welded into the rest of
the library.

---

## The two halves

A tensor is `Tensor<T, B>`. The two type parameters answer two separate questions:

- **`T: Dtype`** - the element type. *What* the numbers are. `f32` and `f64` today.
- **`B: Backend`** - the compute device. *Where* and *how* an op executes.

The pair `(T, B)` selects a concrete kernel set. Keeping them orthogonal means a new
dtype doesn't touch the backends and a new backend doesn't touch the dtypes - the
combinatorial explosion that the old single-implementation design invited never forms.

Both traits live in the [`backend`](crate::backend) module.

---

## Why a `ComputeFor<B>` layer instead of one big trait?

The obvious design is to put `compute` directly on `Backend` and be done. Candela has an
extra hop - `Backend::compute` delegates to `ComputeFor<B>::compute`, a trait parameterized
on the dtype:

```rust,ignore
pub trait ComputeFor<B: Backend>: Dtype {
    fn compute(/* ... */) -> TensorData<Self>;
    fn compute_inplace(/* ... */) -> TensorData<Self>;
}
```

The reason is trait coherence. Each `(dtype, backend)` pair wants to specialize
independently - MKL has a `vdExp` for `f64` and a `vsExp` for `f32` that are genuinely
different symbols, not one generic function. Expressing that as `impl ComputeFor<CpuMkl>
for f64` and `impl ComputeFor<CpuMkl> for f32` lets each pair stand alone without the
generic-method-vs-specialization conflicts you hit trying to do it all on one trait. The
`Backend` methods stay generic over `T: ComputeFor<Self>` and just forward.

---

## The backends that ship

Both are CPU. Both are zero-sized policy types - all the state lives in the `TensorData`
buffers that flow through `compute`.

### `CpuPure` - the default

Pure Rust. [`matrixmultiply`](https://crates.io/crates/matrixmultiply) for `MatMul`,
straightforward loops over the layout for everything else. No system dependencies, no
linker surprises, builds on any target Rust supports. This is what you get unless you ask
for otherwise.

### `CpuMkl` - opt-in, behind `--features mkl`

Routes element-wise math through MKL/VML and matmul through CBLAS. Faster on Intel
hardware. The cost is a hard link-time dependency on the MKL libraries, which is exactly
why it's gated off by default - the pure build skips compiling and linking MKL entirely.

```bash
cargo build                 # CpuPure
cargo build --features mkl  # CpuMkl
```

### `DefaultBackend`

The backend the constructors infer when you don't name one. It tracks the feature flag:

```rust,ignore
#[cfg(feature = "mkl")]
pub type DefaultBackend = cpu_mkl::CpuMkl;
#[cfg(not(feature = "mkl"))]
pub type DefaultBackend = cpu_pure::CpuPure;
```

So the same source picks up MKL just by flipping the flag - you almost never write `B`
out:

```rust
# use candela::Tensor;
let t = Tensor::from_scalar(1.0_f64, &[4]);          // Tensor<f64, DefaultBackend>
let t: Tensor<f32> = Tensor::from_scalar(1.0, &[4]); // f32, default backend
```

---

## What it takes to add a backend

Implement `Backend` for a new zero-sized type, then implement `ComputeFor<YourBackend>`
for each dtype you want it to handle. `Backend` itself is small: two methods (`compute`,
`compute_inplace`) and a couple of associated consts that describe what the backend's
matmul can ingest without help from the op layer.

Those consts are the part worth understanding, because they move work across the
backend/op-layer boundary:

- **`SUPPORTS_NON_CONTIGUOUS_MATMUL`** - `true` if the backend's matmul accepts any input
  whose last two axes are contiguous. When `false`, the op layer inserts an `AsContiguous`
  before the matmul.
- **`SUPPORTS_2D_TRANSPOSED_MATMUL`** - `true` if a rank-2 transposed view (`row_stride ==
  1`, `col_stride > 1`) can go straight to GEMM with a trans-flag and no packing copy.
  Defaults to `SUPPORTS_NON_CONTIGUOUS_MATMUL`.

Setting these honestly is how a backend opts into doing less copying: claim a capability
and the op layer trusts you with the strided input; leave it `false` and the op layer
packs the data contiguous first. There's one hard requirement that isn't optional, though
- a backend's `MatMul` **must** accept a stride-0 batch axis, because the op layer relies
on re-reading the same matrix per batch iteration to implement batch broadcasting without
materializing the broadcast.

`compute_inplace` carries a contract from the planner: the buffer at `inputs[output_idx]`
is guaranteed to be dead - no live node still reads it - so writing the result over it is
sound. The planner earns that guarantee through the liveness analysis described in
[the planner docs](crate::docs::planner); the backend just trusts it.
