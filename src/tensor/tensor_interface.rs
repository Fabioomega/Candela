#![allow(private_bounds)]
use crate::OpError;
use crate::tensor::backend::{Backend, ComputeFor, DefaultBackend};
use crate::tensor::graph::{NodeKind, TensorGraphEdge};
use crate::tensor::iter::{InformedIter, Iter, StepInfo};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::promise::TensorPromise;
use crate::tensor::shape::IntoShape;
use crate::tensor::skeleton::SkeletonSlot;
use crate::tensor::skeleton::{Clean, Tainting};
use crate::tensor::storage::TensorData;
use crate::tensor::traits::{Composable, Dimension, Numeric, Operand};
use std::ops::Index;
use std::sync::Arc;

/// Allocated tensor data exposed through the public API.
///
/// Internally, a `Tensor<T>` is an `Arc<TensorGraphEdge<T>>`: a
/// reference-counted leaf node that carries the concrete `TensorData<T>`
/// (data buffer + [`Layout`]) and the unique ID that the execution planner uses
/// to track this value in the graph.
///
/// [`Layout`]: crate::tensor::mem_formats::layout::Layout
///
/// # Examples
///
/// ```
/// use candela::Tensor;
///
/// // Fill every element with the same value.
/// let t = Tensor::from_scalar(1.0_f64, (3, 3));
/// assert_eq!(t.data().len(), 9);
///
/// // From an existing vec - total elements must equal the product of `shape`.
/// let t = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], 3);
/// assert_eq!(t.data(), &vec![1.0, 2.0, 3.0]);
///
/// // From any iterator.
/// let t = Tensor::from_iter([1.0_f64, 2.0, 3.0, 4.0], 4);
/// assert_eq!(t.data().len(), 4);
///
/// // Ops build a graph and run when you materialize.
/// let t = Tensor::from_scalar(3.0_f64, 4);
/// let result = (t * 2.0 + 1.0).materialize();
/// assert_eq!(result.data(), &vec![7.0; 4]);
/// ```
///
/// # The `_in` variants
///
/// Every constructor comes in two forms. The bare name (e.g.
/// [`from_scalar`](Self::from_scalar)) builds on [`DefaultBackend`]; the
/// `_in`-suffixed form (e.g. [`from_scalar_in`](Self::from_scalar_in)) is generic
/// over the backend `B`, taken from the tensor's type annotation. The two are
/// otherwise identical.
///
/// ```
/// use candela::Tensor;
/// use candela::backend::CpuPure;
///
/// let a: Tensor<f64> = Tensor::from_scalar(1.0, 3);             // DefaultBackend
/// let b: Tensor<f64, CpuPure> = Tensor::from_scalar_in(1.0, 3); // explicit B
/// assert_eq!(a.data(), b.data());
/// ```
pub struct Tensor<T, B: Backend = DefaultBackend> {
    pub(crate) graph: Arc<TensorGraphEdge<T, B>>,
}

impl<T: ComputeFor<B>, B: Backend> Tensor<T, B> {
    /// Like [`Tensor::from_scalar`], but on an explicit backend `B`. See
    /// [`Tensor`] for the `_in` convention.
    #[inline]
    pub fn from_scalar_in(scalar: T, shape: impl IntoShape) -> Self {
        let (rank, shape) = shape.into_shape();
        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(TensorData::from_scalar(
                scalar,
                &shape[..rank],
            ))),
        }
    }

    /// Like [`Tensor::from_vec`], but on an explicit backend `B`. See [`Tensor`]
    /// for the `_in` convention.
    ///
    /// # Panics
    ///
    /// Panics if `vector` length does not equal the product of `shape`.
    #[inline]
    pub fn from_vec_in(vector: Vec<T>, shape: impl IntoShape) -> Self {
        let (rank, shape) = shape.into_shape();
        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(TensorData::from_vec(
                vector,
                &shape[..rank],
                0,
            ))),
        }
    }

    /// Like [`Tensor::from_vec_with_layout`], but on an explicit backend `B`. See [`Tensor`]
    /// for the `_in` convention.
    ///
    /// # Panics
    ///
    /// Panics if `layout` can address memory outside of `vector` - the last
    /// addressable index (`offset + (shape - 1) * stride` summed over all axes)
    /// must be within the buffer - or if any stride is negative.
    #[inline]
    pub fn from_vec_with_layout_in(vector: Vec<T>, layout: Layout) -> Self {
        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(
                TensorData::from_vec_with_layout(vector, layout),
            )),
        }
    }

    /// Like [`Tensor::from_slice`], but on an explicit backend `B`. See [`Tensor`]
    /// for the `_in` convention.
    ///
    /// # Panics
    ///
    /// Panics if `data` length does not equal the product of `shape`.
    #[inline]
    pub fn from_slice_in(data: &[T], shape: impl IntoShape) -> Self {
        Self::from_vec_in(data.to_vec(), shape)
    }

    /// Like [`Tensor::from_iter`], but on an explicit backend `B`. See [`Tensor`]
    /// for the `_in` convention.
    ///
    /// # Panics
    ///
    /// Panics if `iter` yields fewer than the product of `shape` elements.
    ///
    /// # Note
    ///
    /// Surplus elements beyond the product of `shape` are ignored.
    #[inline]
    pub fn from_iter_in<I>(iter: I, shape: impl IntoShape) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let (rank, shape) = shape.into_shape();
        let size: usize = shape[..rank].iter().product();

        let vector: Vec<T> = std::vec::Vec::from_iter(iter.into_iter().take(size));
        Self::from_vec_in(vector, &shape[..rank])
    }

    /// Like [`Tensor::eye`], but on an explicit backend `B`. See [`Tensor`] for
    /// the `_in` convention.
    #[inline]
    pub fn eye_in(n: usize, m: usize) -> Self {
        let mut data: Vec<T> = vec![T::ZERO; n * m];

        let mut i: usize = 0;
        while i < data.len() {
            data[i] = T::ONE;
            i += m + 1;
        }

        Self::from_vec_in(data, (n, m))
    }
}

impl<T: ComputeFor<DefaultBackend>> Tensor<T> {
    /// Create a tensor with every element set to `scalar`.
    ///
    /// Uses [`DefaultBackend`]; see [`from_scalar_in`](Self::from_scalar_in) to
    /// pick a backend explicitly.
    #[inline]
    pub fn from_scalar(scalar: T, shape: impl IntoShape) -> Self {
        Self::from_scalar_in(scalar, shape)
    }

    /// Create a tensor from `vector` interpreted with `shape`.
    ///
    /// Uses [`DefaultBackend`]; see [`from_vec_in`](Self::from_vec_in) to pick a
    /// backend explicitly.
    ///
    /// # Panics
    ///
    /// Panics if `vector` length does not equal the product of `shape`.
    #[inline]
    pub fn from_vec(vector: Vec<T>, shape: impl IntoShape) -> Self {
        Self::from_vec_in(vector, shape)
    }

    /// Creates a tensor from `vector` interpreted with `layout`.
    ///
    /// The buffer may be larger than the layout requires: gapped (strided)
    /// layouts leave elements unread, and stride-0 (broadcast) layouts read
    /// the same elements repeatedly.
    ///
    /// Uses [`DefaultBackend`]; see
    /// [`from_vec_with_layout_in`](Self::from_vec_with_layout_in) to pick a
    /// backend explicitly.
    ///
    /// # Panics
    ///
    /// Panics if `layout` can address memory outside of `vector` - the last
    /// addressable index (`offset + (shape - 1) * stride` summed over all axes)
    /// must be within the buffer - or if any stride is negative.
    #[inline]
    pub fn from_vec_with_layout(vector: Vec<T>, layout: Layout) -> Self {
        Self::from_vec_with_layout_in(vector, layout)
    }

    /// Create a tensor by copying `data` into a buffer with the given `shape`.
    ///
    /// Uses [`DefaultBackend`]; see [`from_slice_in`](Self::from_slice_in) to
    /// pick a backend explicitly.
    ///
    /// # Panics
    ///
    /// Panics if `data` length does not equal the product of `shape`.
    #[inline]
    pub fn from_slice(data: &[T], shape: impl IntoShape) -> Self {
        Self::from_slice_in(data, shape)
    }

    /// Create a tensor by collecting `iter` into a buffer with the given `shape`.
    ///
    /// Uses [`DefaultBackend`]; see [`from_iter_in`](Self::from_iter_in) to pick
    /// a backend explicitly.
    ///
    /// # Panics
    ///
    /// Panics if `iter` yields fewer than the product of `shape` elements.
    ///
    /// # Note
    ///
    /// Surplus elements beyond the product of `shape` are ignored.
    #[inline]
    pub fn from_iter<I>(iter: I, shape: impl IntoShape) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::from_iter_in(iter, shape)
    }

    /// Create an `n`×`m` matrix with ones on the main diagonal and zeros elsewhere.
    ///
    /// With `n == m` this is the identity matrix.
    ///
    /// Uses [`DefaultBackend`]; see [`eye_in`](Self::eye_in) to pick a backend
    /// explicitly.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Tensor;
    /// let i: Tensor<f64> = Tensor::eye(2, 2);
    /// assert_eq!(i.data(), &[1.0, 0.0, 0.0, 1.0]);
    /// ```
    #[inline]
    pub fn eye(n: usize, m: usize) -> Self {
        Self::eye_in(n, m)
    }
}

#[cfg(feature = "rand")]
impl<T: ComputeFor<B>, B: Backend> Tensor<T, B> {
    /// Like [`Tensor::rand`], but on an explicit backend `B` and driven by a
    /// caller-supplied `rng`. See [`Tensor`] for the `_in` convention.
    #[inline]
    pub fn rand_with_in<R: rand::Rng>(shape: impl IntoShape, rng: &mut R) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        use rand::distr::Distribution;

        Self::from_iter_in(rand::distr::StandardUniform.sample_iter(rng), shape)
    }

    /// Like [`Tensor::rand`], but on an explicit backend `B`. See [`Tensor`] for
    /// the `_in` convention.
    #[inline]
    pub fn rand_in(shape: impl IntoShape) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        Self::rand_with_in(shape, &mut rand::rng())
    }

    /// Like [`Tensor::randn`], but on an explicit backend `B` and driven by a
    /// caller-supplied `rng`. See [`Tensor`] for the `_in` convention.
    #[inline]
    pub fn randn_with_in<R: rand::Rng>(shape: impl IntoShape, rng: &mut R) -> Self
    where
        rand_distr::StandardNormal: rand::distr::Distribution<T>,
    {
        use rand::distr::Distribution;

        Self::from_iter_in(rand_distr::StandardNormal.sample_iter(rng), shape)
    }

    /// Like [`Tensor::randn`], but on an explicit backend `B`. See [`Tensor`] for
    /// the `_in` convention.
    #[inline]
    pub fn randn_in(shape: impl IntoShape) -> Self
    where
        rand_distr::StandardNormal: rand::distr::Distribution<T>,
    {
        Self::randn_with_in(shape, &mut rand::rng())
    }

    /// Like [`Tensor::sample`], but on an explicit backend `B` and driven by a
    /// caller-supplied `rng`. See [`Tensor`] for the `_in` convention.
    #[inline]
    pub fn sample_with_in<D, R>(shape: impl IntoShape, dist: D, rng: &mut R) -> Self
    where
        D: rand::distr::Distribution<T>,
        R: rand::Rng,
    {
        Self::from_iter_in(dist.sample_iter(rng), shape)
    }

    /// Like [`Tensor::sample`], but on an explicit backend `B`. See [`Tensor`] for
    /// the `_in` convention.
    #[inline]
    pub fn sample_in<D>(shape: impl IntoShape, dist: D) -> Self
    where
        D: rand::distr::Distribution<T>,
    {
        Self::sample_with_in(shape, dist, &mut rand::rng())
    }
}

impl<T: Clone, B: Backend> Tensor<T, B> {
    #[inline]
    pub(crate) fn from_data(data: TensorData<T>) -> Self {
        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(data)),
        }
    }

    /// Returns a reference to the underlying data buffer.
    ///
    /// Slicing, transposition, etc will change the layout of the tensor
    /// so this is not guaranteed to be what you expect the tensor to
    /// logically contain.
    ///
    /// Use [`.iter()`][Self::iter], to iterate over the whole tensor following
    /// logical order or [`.index()`][Self::index] to access a single element by index.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Tensor;
    /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    /// assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn data(&self) -> &[T] {
        self.graph.get().data()
    }

    /// Iterate over the tensor's elements in logical (row-major) order.
    ///
    /// Unlike [`.data()`][Self::data], this follows the tensor's layout, so a
    /// sliced or transposed tensor yields its elements in the order its shape
    /// implies.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Tensor;
    /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0], 3);
    /// let collected: Vec<f64> = t.iter().copied().collect();
    /// assert_eq!(collected, vec![1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        self.graph.get().iter()
    }

    /// Iterate over the backing buffer using `layout` instead of this tensor's own
    /// layout. Useful for traversals more exotic than the safe interface exposes.
    ///
    /// # Safety
    ///
    /// `layout` must be a valid transformation of this tensor's current layout -
    /// every index it addresses must fall within the backing buffer. A layout
    /// derived from this tensor's layout (a view, slice, transpose, or broadcast
    /// of it) upholds this; an unrelated layout may read out of bounds and is
    /// undefined behaviour.
    #[inline]
    pub unsafe fn iter_as_layout<'a>(&'a self, layout: &'a Layout) -> Iter<'a, T> {
        unsafe { self.graph.get().iter_as_layout(layout) }
    }

    /// Walk the tensor depth-first, yielding a [`StepInfo`] for each element and
    /// for each dimension boundary crossed along the way.
    ///
    /// Unlike [`.iter()`][Self::iter], which yields a flat stream of elements,
    /// these events carry enough structure to reconstruct the tensor's nesting.
    /// The walk follows the logical layout, so a sliced or transposed tensor is
    /// visited in the order its shape implies. It rebuilds that order one index
    /// at a time and is not intended for hot paths. The
    /// [`Display`](std::fmt::Display) implementation is built on it.
    ///
    /// # Examples
    ///
    /// Regroup a flat buffer back into its rows - something [`.iter()`][Self::iter]
    /// alone can't do, because it never signals where one row ends and the next
    /// begins:
    ///
    /// ```
    /// use candela::{StepInfo, Tensor};
    ///
    /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    ///
    /// let mut rows: Vec<Vec<f64>> = Vec::new();
    /// for step in t.informed_iter() {
    ///     match step {
    ///         // The innermost dimension (axis 1) opening means a new row starts.
    ///         StepInfo::EnterDimension(1) => rows.push(Vec::new()),
    ///         StepInfo::Value(v) => rows.last_mut().unwrap().push(v),
    ///         _ => {}
    ///     }
    /// }
    /// assert_eq!(rows, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    /// ```
    #[inline]
    pub fn informed_iter(&self) -> InformedIter<'_, T> {
        self.graph.get().informed_iter()
    }

    /// Makes a deep copy of this tensor.
    #[inline]
    pub fn deep_clone(&self) -> Self {
        let data = self.graph.get();

        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(data.deep_clone())),
        }
    }

    /// Make a shallow copy of this tensor with a new graph identity.
    ///
    /// The underlying buffer is shared with the original, but the new tensor carries a fresh
    /// graph ID. The planner treats it as an unrelated input - no connection is maintained to
    /// any live promises that reference the original. Use [`Tensor::clone`] to preserve that
    /// connection, or [`Tensor::deep_clone`] for a fully independent buffer.
    ///
    /// [`Tensor::clone`]: Tensor::clone
    /// [`Tensor::deep_clone`]: Tensor::deep_clone
    #[inline]
    pub fn clone_detached(&self) -> Self {
        let data = self.graph.get();

        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(data.clone())),
        }
    }
}

#[cfg(feature = "rand")]
impl<T: ComputeFor<DefaultBackend>> Tensor<T> {
    /// Fill a tensor of the given `shape` with samples from the uniform
    /// distribution over `[0, 1)`.
    ///
    /// Uses [`DefaultBackend`] and [`rand::rng()`]; see [`rand_in`](Self::rand_in)
    /// to pick a backend and [`rand_with`](Self::rand_with) to supply the RNG. For
    /// any other distribution, see [`sample`](Self::sample).
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Tensor;
    /// let t: Tensor<f64> = Tensor::rand((2, 3));
    /// assert_eq!(t.data().len(), 6);
    /// assert!(t.data().iter().all(|&x| (0.0..1.0).contains(&x)));
    /// ```
    #[inline]
    pub fn rand(shape: impl IntoShape) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        Self::rand_in(shape)
    }

    /// Like [`rand`](Self::rand), but driven by a caller-supplied `rng`. Seeding
    /// the RNG makes the result reproducible.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Tensor;
    /// use rand::SeedableRng;
    /// use rand::rngs::StdRng;
    ///
    /// let a: Tensor<f64> = Tensor::rand_with((2, 2), &mut StdRng::seed_from_u64(42));
    /// let b: Tensor<f64> = Tensor::rand_with((2, 2), &mut StdRng::seed_from_u64(42));
    /// assert_eq!(a.data(), b.data()); // same seed, same tensor
    /// ```
    #[inline]
    pub fn rand_with<R: rand::Rng>(shape: impl IntoShape, rng: &mut R) -> Self
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        Self::rand_with_in(shape, rng)
    }

    /// Fill a tensor of the given `shape` with samples from the standard normal
    /// distribution (mean `0`, standard deviation `1`).
    ///
    /// Uses [`DefaultBackend`] and [`rand::rng()`]; see
    /// [`randn_in`](Self::randn_in) to pick a backend and
    /// [`randn_with`](Self::randn_with) to supply the RNG. For any other
    /// distribution, see [`sample`](Self::sample).
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Tensor;
    /// let t: Tensor<f32> = Tensor::randn(1000);
    /// assert_eq!(t.data().len(), 1000);
    /// ```
    #[inline]
    pub fn randn(shape: impl IntoShape) -> Self
    where
        rand_distr::StandardNormal: rand::distr::Distribution<T>,
    {
        Self::randn_in(shape)
    }

    /// Like [`randn`](Self::randn), but driven by a caller-supplied `rng`. Seeding
    /// the RNG makes the result reproducible.
    #[inline]
    pub fn randn_with<R: rand::Rng>(shape: impl IntoShape, rng: &mut R) -> Self
    where
        rand_distr::StandardNormal: rand::distr::Distribution<T>,
    {
        Self::randn_with_in(shape, rng)
    }

    /// Fill a tensor of the given `shape` with samples from `dist`, drawn from any
    /// [`Distribution`].
    ///
    /// [`rand`](Self::rand) and [`randn`](Self::randn) are the uniform and
    /// standard-normal specializations of this; reach for `sample` for anything
    /// else.
    ///
    /// Uses [`DefaultBackend`] and [`rand::rng()`]; see
    /// [`sample_in`](Self::sample_in) to pick a backend and
    /// [`sample_with`](Self::sample_with) to supply the RNG.
    ///
    /// [`Distribution`]: rand::distr::Distribution
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Tensor;
    /// use rand::distr::Uniform;
    ///
    /// let dist = Uniform::new(-1.0, 1.0).unwrap();
    /// let t: Tensor<f64> = Tensor::sample(4, dist);
    /// assert!(t.data().iter().all(|&x| (-1.0..1.0).contains(&x)));
    /// ```
    #[inline]
    pub fn sample<D>(shape: impl IntoShape, dist: D) -> Self
    where
        D: rand::distr::Distribution<T>,
    {
        Self::sample_in(shape, dist)
    }

    /// Like [`sample`](Self::sample), but driven by a caller-supplied `rng`.
    /// Seeding the RNG makes the result reproducible.
    #[inline]
    pub fn sample_with<D, R>(shape: impl IntoShape, dist: D, rng: &mut R) -> Self
    where
        D: rand::distr::Distribution<T>,
        R: rand::Rng,
    {
        Self::sample_with_in(shape, dist, rng)
    }
}

impl<T: Numeric, B: Backend> Tensor<T, B> {
    /// Wrap this tensor as a [`TensorPromise`] without applying any transformation.
    ///
    /// The primary use case is initializing a mutable accumulator that will have ops
    /// applied to it in a loop - as it needs a [`TensorPromise<T>`] on both sides
    /// of the assignment:
    ///
    /// ```
    /// use candela::arange;
    /// let t = arange!(4);         // [0.0, 1.0, 2.0, 3.0]
    /// let mut p = t.to_promise();
    /// for i in 0..5_u32 {
    ///     p += i as f64;
    /// }
    /// // each element gains 0+1+2+3+4 = 10
    /// assert_eq!(p.materialize().data(), &[10.0, 11.0, 12.0, 13.0]);
    /// ```
    ///
    /// [`TensorPromise<T>`]: crate::tensor::promise::TensorPromise
    #[inline]
    pub fn to_promise(&self) -> TensorPromise<T, B> {
        unsafe {
            TensorPromise::new(
                super::ops::def_op::OpKind::NoOp,
                [NodeKind::Edge(self.graph.clone())].into(),
            )
            .unwrap_unchecked()
        }
    }

    /// Return a reference to the element at `index`, following the tensor's layout.
    ///
    /// # Errors
    ///
    /// Returns [`OpError::NotEnoughAxes`] if `index` doesn't have one entry per
    /// axis, or [`OpError::IndexOutOfBounds`] if an index is past the end of its axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Tensor;
    /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    /// assert_eq!(*t.get(&[1, 2])?, 6.0);
    /// assert!(t.get(&[2, 0]).is_err()); // row 2 is past the end
    /// # Ok::<(), candela::OpError>(())
    /// ```
    // TODO: Add support for negative indexing
    pub fn get(&self, index: &[usize]) -> Result<&T, OpError> {
        self.graph.get().get(index)
    }

    /// Return a reference to the tensor's first element.
    ///
    /// Most useful for reading a one-element result, such as a full reduction like
    /// [`.sum()`][Self::sum].
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Tensor;
    /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], (2, 2));
    /// let total = t.sum().materialize();
    /// assert_eq!(*total.item(), 10.0);
    /// ```
    pub fn item(&self) -> &T {
        self.graph.get().item()
    }

    /// Creates a [`SkeletonSlot`] shaped like this tensor.
    ///
    /// The slot is an input placeholder for a [`Skeleton`]. It has the tensor's
    /// [`Layout`] but holds no data.
    ///
    /// [`Skeleton`]: crate::skeleton::Skeleton
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Tensor;
    ///
    /// let a = Tensor::from_scalar(0.3, 4);
    /// let slot = a.to_slot();                       // placeholder shaped like a
    /// let skeleton = (&slot * 2.0 + 1.0).into_skeleton(&[slot])?;
    /// assert!(skeleton.run(&[&a]).is_ok());
    /// # Ok::<(), candela::OpError>(())
    /// ```
    pub fn to_slot(&self) -> SkeletonSlot<T, B> {
        SkeletonSlot::new(self.layout().clone())
    }
}

impl<T, B: Backend> Dimension for Tensor<T, B> {
    #[inline]
    fn layout(&self) -> &super::mem_formats::layout::Layout {
        self.graph.layout()
    }
}

impl<T, B: Backend> Operand<T, B> for Tensor<T, B> {
    fn to_node(&self) -> NodeKind<T, B> {
        NodeKind::Edge(self.graph.clone())
    }
}

impl<T, B: Backend> Tainting for Tensor<T, B> {
    type Mark = Clean;
}

impl<T, B: Backend> Composable<T, B> for Tensor<T, B> {}

impl<T, B: Backend> Clone for Tensor<T, B> {
    /// Shallow copy sharing the same underlying buffer and graph identity.
    ///
    /// Equivalent to bumping an `Arc` reference count. The copy is connected to all promises
    /// that reference the original - the planner sees them as the same input node. For a copy
    /// the graph treats as unrelated, use [`clone_detached`]. For an independent buffer, use
    /// [`deep_clone`].
    ///
    /// [`clone_detached`]: Tensor::clone_detached
    /// [`deep_clone`]: Tensor::deep_clone
    #[inline]
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
        }
    }
}

#[allow(private_bounds)]
impl<T: std::fmt::Display + Copy, B: Backend> std::fmt::Debug for Tensor<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tensor {:?}", self.layout())?;
        std::fmt::Display::fmt(self, f)
    }
}

impl<T, B> Index<&[usize]> for Tensor<T, B>
where
    T: Copy,
    B: Backend,
{
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        &self.graph.get()[index]
    }
}

#[allow(private_bounds)]
impl<T: std::fmt::Display + Copy, B: Backend> std::fmt::Display for Tensor<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut indent = 0;
        let mut in_seq = false;

        debug_assert!(!self.shape().is_empty(), "Tensor rank must be >= 1");
        let last = self.shape().len() - 1;

        for step in self.informed_iter() {
            match step {
                StepInfo::EnterDimension(dim) => {
                    write!(f, "{:indent$}[", "", indent = indent)?;
                    indent += 2;

                    if dim != last {
                        writeln!(f)?;
                    }
                }
                StepInfo::ExitDimension(dim) => {
                    indent -= 2;
                    in_seq = false;

                    if dim != last {
                        write!(f, "{:indent$}", "", indent = indent)?;
                    }

                    writeln!(f, "]")?;
                }
                StepInfo::Value(v) => {
                    if in_seq {
                        write!(f, ", ")?;
                    }

                    write!(f, "{:>4}", v)?;

                    in_seq = true;
                }
                _ => {}
            }
        }

        Ok(())
    }
}
