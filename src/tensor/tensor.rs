#![allow(private_bounds)]
use crate::errors::OpError;
use crate::tensor::backend::{Backend, ComputeFor, DefaultBackend};
use crate::tensor::graph::{NodeKind, TensorGraphEdge};
use crate::tensor::iter::{InformedSliceIter, SliceIter, StepInfo};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::promise::TensorPromise;
use crate::tensor::storage::TensorData;
use crate::tensor::traits::{Dimension, Numeric};
use std::ops::Index;
use std::sync::Arc;

/// Allocated tensor data exposed through the public API.
///
/// Internally, a `Tensor<T>` is an `Arc<`[`TensorGraphEdge<T>`]`>`: a
/// reference-counted leaf node that carries the concrete [`TensorData<T>`]
/// (data buffer + [`Layout`]) and the unique ID that the execution planner uses
/// to track this value in the graph. `Clone` is therefore a cheap Arc bump — the
/// underlying storage is shared with the original. Call [`clone_deep`] when you
/// need a fully independent buffer, or [`clone_detached`] when you want a shallow
/// copy that the graph will treat as an unrelated tensor.
///
/// [`TensorGraphEdge<T>`]: crate::tensor::graph::TensorGraphEdge
/// [`TensorData<T>`]: crate::tensor::storage::TensorData
/// [`Layout`]: crate::tensor::mem_formats::layout::Layout
/// [`clone_deep`]: Tensor::clone_deep
/// [`clone_detached`]: Tensor::clone_detached
///
/// # Examples
///
/// ```
/// use candela::Tensor;
///
/// // Fill every element with the same value.
/// let t = Tensor::from_scalar(1.0_f64, &[3, 3]);
/// assert_eq!(t.data().len(), 9);
///
/// // From an existing vec — total elements must equal the product of `shape`.
/// let t = Tensor::from_vec(vec![1.0_f64, 2.0, 3.0], &[3]);
/// assert_eq!(t.data(), &vec![1.0, 2.0, 3.0]);
///
/// // From any iterator.
/// let t = Tensor::from_iter([1.0_f64, 2.0, 3.0, 4.0], &[4]);
/// assert_eq!(t.data().len(), 4);
///
/// // Ops build a graph and run when you materialize.
/// let t = Tensor::from_scalar(3.0_f64, &[4]);
/// let result = (t * 2.0 + 1.0).materialize();
/// assert_eq!(result.data(), &vec![7.0; 4]);
/// ```
pub struct Tensor<T, B: Backend = DefaultBackend> {
    pub(crate) graph: Arc<TensorGraphEdge<T, B>>,
}

impl<T: ComputeFor<DefaultBackend>> Tensor<T> {
    /// Create a tensor with every element set to `scalar`.
    #[inline]
    pub fn from_scalar(scalar: T, shape: &[usize]) -> Self {
        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(TensorData::from_scalar(
                scalar, shape,
            ))),
        }
    }

    /// Create a tensor from `vector` interpreted with `shape`.
    ///
    /// Panics if `vector.len()` does not equal the product of `shape`.
    #[inline]
    pub fn from_vec(vector: Vec<T>, shape: &[usize]) -> Self {
        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(TensorData::from_vec(
                vector, shape, 0,
            ))),
        }
    }

    /// Create a tensor by copying `data` into a buffer with the given `shape`.
    ///
    /// Panics if `data.len()` does not equal the product of `shape`.
    #[inline]
    pub fn from_slice(data: &[T], shape: &[usize]) -> Self {
        Self::from_vec(data.to_vec(), shape)
    }

    /// Create a tensor by collecting `iter` into a buffer with the given `shape`.
    #[inline]
    pub fn from_iter<I>(iter: I, shape: &[usize]) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let vector: Vec<T> = std::vec::Vec::from_iter(iter);
        Self::from_vec(vector, shape)
    }

    #[inline]
    pub fn eye(n: usize, m: usize) -> Self {
        let mut data: Vec<T> = vec![T::ZERO; n * m];

        let mut acc: usize = 0;
        for _ in 0..n {
            data[acc] = T::ONE;
            acc += m + 1;
        }

        Self::from_vec(data, &[n, m])
    }
}

impl<T: Copy, B: Backend> Tensor<T, B> {
    #[inline]
    pub fn from_data(data: TensorData<T>) -> Self {
        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(data)),
        }
    }

    /// Return a reference to the underlying data buffer.
    /// Slices, transposition, etc will change the layout of the tensor
    /// so this is not guaranteed to be what you expect
    /// the tensor to logically contain.
    #[inline]
    pub fn data(&self) -> &Vec<T> {
        self.graph.get().data()
    }

    #[inline]
    pub fn iter(&self) -> SliceIter<'_, T> {
        self.graph.get().iter()
    }

    #[inline]
    pub unsafe fn iter_as_layout<'a>(&'a self, layout: &'a Layout) -> SliceIter<'a, T> {
        unsafe { self.graph.get().iter_as_layout(layout) }
    }

    #[inline]
    pub fn informed_iter(&self) -> InformedSliceIter<'_, T> {
        self.graph.get().informed_iter()
    }

    /// Makes a deep copy of this tensor.
    #[inline]
    pub fn clone_deep(&self) -> Self {
        let data = self.graph.get();

        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(data.clone_deep())),
        }
    }

    /// Make a shallow copy of this tensor with a new graph identity.
    ///
    /// The underlying buffer is shared with the original, but the new tensor carries a fresh
    /// graph ID. The planner treats it as an unrelated input — no connection is maintained to
    /// any live promises that reference the original. Use [`Tensor::clone`] to preserve that
    /// connection, or [`Tensor::clone_deep`] for a fully independent buffer.
    ///
    /// [`Tensor::clone`]: Tensor::clone
    /// [`Tensor::clone_deep`]: Tensor::clone_deep
    #[inline]
    pub fn clone_detached(&self) -> Self {
        let data = self.graph.get();

        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(data.clone())),
        }
    }
}

impl<T: Numeric, B: Backend> Tensor<T, B> {
    /// Wrap this tensor as a [`TensorPromise`] without applying any transformation.
    ///
    /// Creates a `NoOp` [`TensorGraphNode`] with the tensor's edge as its sole input.
    /// The primary use case is initializing a mutable accumulator that will have ops
    /// applied to it in a loop — as it needs a [`TensorPromise<T>`] on both sides
    /// of the assignment:
    ///
    /// ```
    /// use candela::arange;
    /// let t = arange!(4);         // [0.0, 1.0, 2.0, 3.0]
    /// let mut p = t.as_promise();
    /// for i in 0..5_u32 {
    ///     p = p + i as f64;
    /// }
    /// // each element gains 0+1+2+3+4 = 10
    /// assert_eq!(p.materialize().data(), &[10.0, 11.0, 12.0, 13.0]);
    /// ```
    ///
    /// [`TensorPromise<T>`]: crate::tensor::promise::TensorPromise
    /// [`TensorGraphNode`]: crate::tensor::graph::TensorGraphNode
    #[inline]
    pub fn as_promise(&self) -> TensorPromise<T, B> {
        unsafe {
            TensorPromise::new(
                super::ops::def_op::OpKind::NoOp,
                [NodeKind::Edge(self.graph.clone())].into(),
            )
            .unwrap_unchecked()
        }
    }

    pub fn get(&self, index: &[usize]) -> Result<&T, OpError> {
        self.graph.get().get(index)
    }

    pub fn item(&self) -> &T {
        self.graph.get().item()
    }
}

impl<T, B: Backend> Dimension for Tensor<T, B> {
    #[inline]
    fn layout(&self) -> &super::mem_formats::layout::Layout {
        self.graph.layout()
    }
}

impl<T, B: Backend> Clone for Tensor<T, B> {
    /// Shallow copy sharing the same underlying buffer and graph identity.
    ///
    /// Equivalent to bumping an `Arc` reference count. The copy is connected to all promises
    /// that reference the original — the planner sees them as the same input node. For a copy
    /// the graph treats as unrelated, use [`clone_detached`]. For an independent buffer, use
    /// [`clone_deep`].
    ///
    /// [`clone_detached`]: Tensor::clone_detached
    /// [`clone_deep`]: Tensor::clone_deep
    #[inline]
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
        }
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
                        write!(f, "\n")?;
                    }
                }
                StepInfo::ExitDimension(dim) => {
                    indent -= 2;
                    in_seq = false;

                    if dim != last {
                        write!(f, "{:indent$}", "", indent = indent)?;
                    }

                    write!(f, "]\n")?;
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
