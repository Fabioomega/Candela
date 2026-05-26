#![allow(private_bounds)]
pub mod tensor;

pub use tensor::arange;
pub use tensor::errors;
pub use tensor::{CachedTensorPromise, Dimension, Layout, SliceRange, Tensor, TensorPromise};

use std::ops::Neg;

use crate::tensor::FromIndex;
use crate::tensor::backend::ComputeFor;
use crate::tensor::backend::DefaultBackend;
use crate::tensor::definitions::NumberLike;
use crate::tensor::ops::CanMatMul;
use crate::tensor::ops::FloatLike;
use crate::tensor::traits::Numeric;

const PACKING_BUFFER_SIZE: usize = 2048;

/// Sealed marker for floating-point tensor element types: `f32` and `f64`.
///
/// Bundles the bounds required for full tensor operation support: arithmetic
/// (`NumberLike`), floating-point ops (`FloatLike`, `Neg`), matrix
/// multiplication (`CanMatMul`), index-based construction (`FromIndex`), and a
/// lossless `Into<f64>` conversion used by comparison utilities such as
/// `assert_approx_eq`. The inverse, [`from_f64`](Self::from_f64), lets generic
/// code construct typed values from float literals (precision is lost when `T`
/// is `f32` and the source does not fit).
///
/// The trait is sealed: the `pub(crate)` supertraits (`CanMatMul`, `FromIndex`)
/// cannot be named outside this crate, so no external implementation is
/// possible.
///
/// # Examples
///
/// ```
/// use candela::{FloatLikeTensorElement, Tensor};
///
/// fn scaled_identity<T: FloatLikeTensorElement>(n: usize, factor: T) -> Tensor<T> {
///     (Tensor::from_scalar(T::from_f64(1.0), &[n]) * factor).materialize()
/// }
///
/// let f64_result: Tensor<f64> = scaled_identity(4, 3.0);
/// let f32_result: Tensor<f32> = scaled_identity(4, 3.0);
/// assert_eq!(f64_result.data(), &vec![3.0f64; 4]);
/// assert_eq!(f32_result.data(), &vec![3.0f32; 4]);
/// ```
pub trait FloatLikeTensorElement:
    NumberLike
    + Numeric
    + FloatLike
    + Into<f64>
    + Neg<Output = Self>
    + CanMatMul
    + FromIndex
    + ComputeFor<DefaultBackend>
{
    /// Construct a typed value from an `f64` literal. Lossless for `f64`; for
    /// `f32` the value is narrowed via `as f32`.
    fn from_f64(v: f64) -> Self;
}

impl FloatLikeTensorElement for f64 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
}
impl FloatLikeTensorElement for f32 {
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}
