#[cfg(test)]
mod backend_tests;
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
        output_buffer: &mut [Self],
        output_layout: &Layout,
        inputs: &[TensorData<Self>],
    );

    fn compute_inplace(
        op: &OpKind<Self>,
        output_buffer: &mut [Self],
        output_layout: &Layout,
        inputs: &[TensorData<Self>],
        output_idx: usize,
    );
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
    /// accepts a rank-2 input whose strides describe a transposed view
    /// (`row_stride == 1`, `col_stride > 1`) - i.e. the underlying GEMM is
    /// invoked with a trans-flag and no copy is required. This path exists
    /// because some BLAS accept a 2D-transposed operand as a special case,
    /// which is significantly faster than cloning the tensor into a contiguous
    /// buffer.
    ///
    /// Only consulted when
    /// [`SUPPORTS_NON_CONTIGUOUS_MATMUL`](Self::SUPPORTS_NON_CONTIGUOUS_MATMUL)
    /// is `false`. Every other non-contiguous input - padded, broadcast, or
    /// transposed above rank 2 - is contiguified by the op layer before
    /// reaching the kernel.
    ///
    /// # Note
    ///
    /// A backend accepting arbitrary strides accepts transposed ones as a
    /// special case, which is what the default value says. Setting this to
    /// `false` alongside a `true`
    /// [`SUPPORTS_NON_CONTIGUOUS_MATMUL`](Self::SUPPORTS_NON_CONTIGUOUS_MATMUL)
    /// has no effect.
    const SUPPORTS_2D_TRANSPOSED_MATMUL: bool = Self::SUPPORTS_NON_CONTIGUOUS_MATMUL;
    /// `true` if `OpKind::MatMul`
    /// accepts inputs with arbitrary strides - any rank, any view, broadcast
    /// (stride-0) axes included - so the op layer never inserts an
    /// `AsContiguous` ahead of a matmul.
    ///
    /// When `false`, the op layer falls back to
    /// [`SUPPORTS_2D_TRANSPOSED_MATMUL`](Self::SUPPORTS_2D_TRANSPOSED_MATMUL)
    /// and inserts an `AsContiguous` on any input that fast path misses.
    const SUPPORTS_NON_CONTIGUOUS_MATMUL: bool;

    /// Run `op` over `inputs` storing at `output_buffer`.
    fn compute<T>(
        op: &OpKind<T>,
        output_buffer: &mut [T],
        output_layout: &Layout,
        inputs: &[TensorData<T>],
    ) where
        T: Dtype + ComputeFor<Self>;

    /// Run `op` over `inputs` storing at `output_buffer` considering
    /// the output_buffer as it were at `inputs[output_idx]`.
    ///
    /// # Note
    ///
    /// The output_layout is the same as layout of the output_buffer.
    /// This is true because only contiguous operations may be done inplace.
    fn compute_inplace<T>(
        op: &OpKind<T>,
        output_buffer: &mut [T],
        output_layout: &Layout,
        inputs: &[TensorData<T>],
        output_idx: usize,
    ) where
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
