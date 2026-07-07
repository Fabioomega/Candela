#![allow(private_bounds)]
//! A lazy, graph-based tensor engine for the CPU, with `f32`/`f64` element
//! types and a pluggable backend (pure-Rust by default, Intel MKL behind the
//! `mkl` feature).
//!
//! Operations on a [`Tensor`] don't compute anything - they build a computation
//! graph and return a [`TensorPromise`]. Calling `.materialize()` plans the
//! whole graph (ordering, buffer reuse, scalar-op fusion) and runs it in one
//! pass, handing back a finished [`Tensor`].
//!
//! ```
//! use candela::{arange, Tensor};
//!
//! // Building the expression allocates nothing; only `.materialize()` runs it.
//! let x: Tensor<f64> = arange!(4);          // [0, 1, 2, 3]
//! let y = (x * 2.0 + 1.0).materialize();    // 2x + 1, fused into one pass
//! assert_eq!(y.data(), &[1.0, 3.0, 5.0, 7.0]);
//! ```
//!
//! # The types
//!
//! - [`Tensor`] - a materialized buffer with a shape and stride.
//! - [`TensorPromise`] - an unevaluated computation graph; `.materialize()` runs it.
//! - [`CachedTensorPromise`] - a promise that keeps its result alive for reuse
//!   across separate materializations.
//! - [`Skeleton`](skeleton::Skeleton) - a graph compiled once over
//!   [`SkeletonSlot`](skeleton::SkeletonSlot) placeholders and run many times against
//!   new data, skipping per-call planning. See the [`skeleton`] module for the
//!   dynamic-shape and caching variants.
//!
//! # Concepts
//!
//! The [`docs`] module shows in detail how the processing pipeline actually works,
//! from expression to computed tensor. Start with the [overview](docs::overview) for
//! the general behaviour; it links each subsystem from there.

mod tensor;

pub use tensor::arange;
pub use tensor::errors::OpError;
pub use tensor::traits::Composable;
pub use tensor::{CachedTensorPromise, Dimension, Layout, SliceRange, Tensor, TensorPromise};

pub mod backend {
    pub use crate::tensor::backend::implementation::*;
    pub use crate::tensor::backend::{Backend, ComputeFor, DefaultBackend};
}

pub mod skeleton {
    pub use crate::tensor::skeleton::{
        BakedPromise, BuildFunction, DynamicSkeleton, EvictionPolicy, LRUPolicy, MemoryMetrics,
        Skeleton, SkeletonCache, SkeletonPromise, SkeletonSlot, UnboundedDynamicSkeleton,
        UnboundedPolicy,
    };
}

/// Design documentation.
///
/// Start with [`overview`](crate::docs::overview) for the whole pipeline, then
/// dive into whichever subsystem you need.
#[cfg(doc)]
pub mod docs {
    #[doc = include_str!("../doc/concepts/backends.md")]
    pub mod backends {}
    #[doc = include_str!("../doc/concepts/graph.md")]
    pub mod graph {}
    #[doc = include_str!("../doc/concepts/layout.md")]
    pub mod layout {}
    #[doc = include_str!("../doc/concepts/planner.md")]
    pub mod planner {}
    #[doc = include_str!("../doc/concepts/planner-history.md")]
    pub mod planner_history {}
    #[doc = include_str!("../doc/concepts/skeleton.md")]
    pub mod skeleton {}
    #[doc = include_str!("../doc/concepts/overview.md")]
    pub mod overview {}
}

use std::ops::Neg;

use crate::backend::{ComputeFor, DefaultBackend};
use crate::tensor::FromIndex;
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
