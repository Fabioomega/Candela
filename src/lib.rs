pub mod tensor;

pub use tensor::arange;
pub use tensor::errors;
pub use tensor::{CachedTensorPromise, Dimension, Layout, SliceRange, Tensor, TensorPromise};

use std::ops::Neg;

use crate::tensor::ops::CanMatMul;
use crate::tensor::ops::FloatLike;
use crate::tensor::ops::TensorElement;
use crate::tensor::FromIndex;

/// Sealed marker for floating-point tensor element types: `f32` and `f64`.
///
/// Bundles the bounds required for full tensor operation support: computation
/// dispatch (`TensorElement`, `ComputeWrapperSpec`), floating-point ops
/// (`FloatLike`, `Neg`), matrix multiplication (`CanMatMul`), index-based
/// construction (`FromIndex`), and a lossless `Into<f64>` conversion used by
/// comparison utilities such as `assert_approx_eq`.
///
/// `ZERO` and `ONE` expose the additive and multiplicative identities as typed
/// values. Both are `0.0` and `1.0` respectively for `f32` and `f64`.
///
/// The trait is sealed: the `pub(crate)` supertraits (`ComputeWrapperSpec`,
/// `CanMatMul`, `FromIndex`) cannot be named outside this crate, so no external
/// implementation is possible.
///
/// # Examples
///
/// ```
/// use candela::{FloatLikeTensorElement, Tensor};
///
/// fn scaled_identity<T: FloatLikeTensorElement>(n: usize, factor: T) -> Tensor<T> {
///     (Tensor::from_scalar(T::ONE, &[n]) * factor).materialize()
/// }
///
/// let f64_result: Tensor<f64> = scaled_identity(4, 3.0);
/// let f32_result: Tensor<f32> = scaled_identity(4, 3.0);
/// assert_eq!(f64_result.data(), &vec![3.0f64; 4]);
/// assert_eq!(f32_result.data(), &vec![3.0f32; 4]);
/// ```
pub trait FloatLikeTensorElement:
    TensorElement + FloatLike + Into<f64> + Neg<Output = Self> + CanMatMul + FromIndex
{
    /// The additive identity — `0.0` for both `f32` and `f64`.
    const ZERO: Self;
    /// The multiplicative identity — `1.0` for both `f32` and `f64`.
    const ONE: Self;
}

impl FloatLikeTensorElement for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}
impl FloatLikeTensorElement for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}
