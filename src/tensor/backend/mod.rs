mod common;
mod common_kernels;
#[cfg(feature = "mkl")]
mod cpu_mkl;
mod cpu_pure;

use std::fmt::Debug;

use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::storage::TensorData;
use crate::tensor::traits::Numeric;

/// Scalar element type the tensor framework supports.
pub trait Dtype: Copy + Numeric {}
impl Dtype for f32 {}
impl Dtype for f64 {}

/// Per-`(T, B)` compute dispatch. Splitting kernels across this trait lets each
/// dtype specialize independently without trait-coherence conflicts; the
/// [`Backend`] methods delegate here.
pub trait ComputeFor<B: Backend>: Dtype {
    fn compute(
        op: &OpKind<Self>,
        output_buffer: Vec<Self>,
        output_layout: &Layout,
        inputs: &[TensorData<Self>],
    ) -> TensorData<Self>;

    fn compute_inplace(
        op: &OpKind<Self>,
        output_layout: &Layout,
        inputs: Vec<TensorData<Self>>,
        output_idx: usize,
    ) -> TensorData<Self>;
}

/// Compute strategy used by [`Tensor`](crate::tensor::Tensor) and the planner
/// to execute graph nodes. A `Backend` impl is a zero-sized policy type; all
/// state lives in the `TensorData` buffers passed through `compute`.
///
/// # Required behaviour
///
/// Implementations must accept stride-0 batch axes in
/// `OpKind::MatMul`. A batched
/// matmul whose leading axis has stride 0 is computed by re-reading the same
/// matrix per batch iteration; the op layer relies on this to skip
/// materializing batch broadcasts.
pub trait Backend: Sized + Debug {
    /// `true` if `OpKind::MatMul`
    /// accepts a 2D input whose strides describe a transposed view
    /// (`row_stride == 1`, `col_stride > 1`) - i.e. the underlying GEMM is
    /// invoked with a trans-flag and no copy is required. The fast path is
    /// deliberately scoped to rank 2; higher-rank tensors are always
    /// contiguified by the op layer before reaching the kernel.
    ///
    /// When `false`, the op layer inserts an `AsContiguous` on any matmul
    /// input that is not already contiguous.
    const SUPPORTS_2D_TRANSPOSED_MATMUL: bool = Self::SUPPORTS_NON_CONTIGUOUS_MATMUL;
    /// `true` if `OpKind::MatMul`
    /// accepts any memory configuration as long the last 2 axis are contiguous.
    ///
    /// When `false`, the op layer inserts an `AsContiguous` on any matmul
    /// input that is not already contiguous.
    const SUPPORTS_NON_CONTIGUOUS_MATMUL: bool;

    /// Run `op` over `inputs` into a fresh allocation. `output_buffer` is the
    /// destination `Vec<T>`; the returned `TensorData` wraps it with
    /// `output_layout`.
    fn compute<T>(
        op: &OpKind<T>,
        output_buffer: Vec<T>,
        output_layout: &Layout,
        inputs: &[TensorData<T>],
    ) -> TensorData<T>
    where
        T: Dtype + ComputeFor<Self>;

    /// Run `op` reusing `inputs[output_idx]`'s buffer as the destination. The
    /// planner guarantees that buffer is no longer referenced by any live
    /// node at this point, so the in-place write is sound.
    fn compute_inplace<T>(
        op: &OpKind<T>,
        output_layout: &Layout,
        inputs: Vec<TensorData<T>>,
        output_idx: usize,
    ) -> TensorData<T>
    where
        T: Dtype + ComputeFor<Self>;
}

/// Backend selected when no explicit type parameter is supplied at the
/// [`Tensor`](crate::tensor::Tensor) construction site. Defaults to the
/// pure-Rust backend; enabling the `mkl` feature switches it to the Intel MKL
/// backend.
///
/// # Examples
///
/// ```
/// use candela::backend::DefaultBackend;
/// use candela::Tensor;
///
/// // `Tensor<T>` is shorthand for `Tensor<T, DefaultBackend>` - the same type.
/// let a: Tensor<f64> = Tensor::from_scalar(1.0, &[3]);
/// let b: Tensor<f64, DefaultBackend> = Tensor::from_scalar(1.0, &[3]);
/// assert_eq!(a.data(), b.data());
/// ```
#[cfg(feature = "mkl")]
pub type DefaultBackend = cpu_mkl::CpuMkl;
#[cfg(not(feature = "mkl"))]
pub type DefaultBackend = cpu_pure::CpuPure;

pub mod implementation {
    #[cfg(feature = "mkl")]
    pub use crate::tensor::backend::cpu_mkl::CpuMkl;
    pub use crate::tensor::backend::cpu_pure::CpuPure;
}
