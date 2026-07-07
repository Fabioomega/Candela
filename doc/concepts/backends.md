# Backends

For most of Candela's life there was no "backend" - there was one implementation: Intel
MKL, linked unconditionally and specialized for `f64`. That scaled badly once new dtypes
such as `f32` arrived and needed different kernel implementations. Inspired by Burn, the
backend became a [trait](crate::backend::Backend), to ease porting across hardware and
kernel implementations.

---

## The `ComputeFor<B>` layer

Each backend may specialize a dtype differently - some dtypes are vectorizable, others are
not, and that changes from backend to backend. [`ComputeFor`](crate::backend::ComputeFor)
expresses this `(backend, dtype)` space in a somewhat scalable way, one implementation per
pair. It also doubles as a construction gate: only a dtype that implements `ComputeFor` for
a given backend can be constructed against it.

---

## The backends that ship

Both are CPU, and both are zero-sized policy types - all the state lives in the `TensorData`
buffers that flow through `compute`.

- **`CpuPure`** - the default. Pure Rust: [`matrixmultiply`](https://crates.io/crates/matrixmultiply)
  for `MatMul`, straightforward loops over the layout for everything else. No system
  dependencies, builds on any target Rust supports.
- **`CpuMkl`** - opt-in, behind `--features mkl`. Routes element-wise math through MKL/VML
  and matmul through CBLAS, faster on Intel hardware. Requires the MKL libraries on the
  system.

`DefaultBackend` is the type alias the constructors infer when none is named, selected by
the `mkl` feature flag: `CpuMkl` when enabled, `CpuPure` otherwise.
