//! Promise types for lazy tensor evaluation.
//!
//! A [`TensorPromise`] describes a computation without running it. Building a
//! chain of ops constructs a graph; nothing is allocated or computed until you
//! call `.materialize()`. [`CachedTensorPromise`] is the same idea, but it keeps
//! the result around so subsequent uses don't need to recompute.

#![allow(private_bounds)]
use std::marker::PhantomData;
use std::sync::Arc;

use crate::tensor::errors::OpError;
use crate::tensor::graph::{NodeKind, TensorGraphCacheNode, TensorGraphNode};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::ops::{ComputeWrapperSpec, TensorElement};
use crate::tensor::tensor::Tensor;
use crate::tensor::traits::{Dimension, Promising};

/// A lazy computation that runs when you call [`.materialize()`].
///
/// Building a `TensorPromise` chain allocates no intermediate tensors — the
/// graph is constructed, not evaluated. Compatible scalar operations are fused
/// during construction so the work is already minimised before execution begins.
///
/// [`.materialize()`]: TensorPromise::materialize
///
/// # Examples
///
/// ```
/// use candela::Tensor;
///
/// let t = Tensor::from_scalar(3.0_f64, &[4]);
/// let result = (t.as_promise() * 2.0 + 1.0).materialize();
/// assert_eq!(result.data(), &vec![7.0; 4]);
/// ```
pub type TensorPromise<T> = RawTensorPromise<TensorGraphNode<T>>;

/// A lazy computation whose result is kept alive after the first evaluation.
///
/// Once [`.materialize()`] has been called (directly or through a derived
/// promise), the result is cached. Every subsequent call returns the stored
/// value without re-running the graph.
///
/// Use this when the same promise feeds into multiple independent downstream
/// graphs that materialise at different times. You pay the memory cost of
/// keeping the tensor alive, which is why caching is opt-in.
///
/// [`.materialize()`]: CachedTensorPromise::materialize
///
/// # Examples
///
/// ```
/// use candela::Tensor;
///
/// let t = Tensor::from_scalar(1.0_f64, &[4]);
/// let cached = (t.as_promise() + 2.0).cache();
///
/// // Two separate materializations — the inner graph runs only once.
/// let r1 = (&cached * 2.0).materialize();
/// let r2 = (&cached + 10.0).materialize();
/// assert_eq!(r1.data(), &vec![6.0; 4]);
/// assert_eq!(r2.data(), &vec![13.0; 4]);
/// ```
pub type CachedTensorPromise<T> = RawTensorPromise<TensorGraphCacheNode<T>>;

/// The underlying generic promise struct, parameterised over the graph node type.
///
/// You won't normally use this type directly — work with the [`TensorPromise`]
/// and [`CachedTensorPromise`] aliases instead.
pub struct RawTensorPromise<P> {
    pub(crate) graph: Arc<P>,
}

impl<T: TensorElement> TensorPromise<T> {
    pub fn new(op: OpKind<T>, inputs: Box<[NodeKind<T>]>) -> Result<Self, OpError> {
        let node = TensorGraphNode::new(op, inputs);

        match node {
            Ok(node) => Ok(Self {
                graph: Arc::new(node),
            }),
            Err(err) => Err(err),
        }
    }

    pub fn with_layout(op: OpKind<T>, inputs: Box<[NodeKind<T>]>, layout: Layout) -> Self {
        Self {
            graph: Arc::new(TensorGraphNode::with_layout(op, inputs, layout)),
        }
    }

    /// Convert this promise into a [`CachedTensorPromise`] that stores its
    /// result after the first evaluation.
    ///
    /// Internally this wraps `self` in a `NoOp` cache node, so the original
    /// graph is unchanged — caching is layered on top, not baked in.
    /// The `NoOp` is used to stop the fusion layer from skipping the cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Tensor;
    ///
    /// let t = Tensor::from_scalar(5.0_f64, &[3]);
    /// let cached = t.as_promise().cache();
    ///
    /// // Safe to use multiple times; result is computed only once.
    /// let _ = (&cached * 2.0).materialize();
    /// let _ = (&cached + 1.0).materialize();
    /// ```
    pub fn cache(self) -> CachedTensorPromise<T> {
        unsafe {
            CachedTensorPromise::new(OpKind::NoOp, [NodeKind::Node(self.graph)].into())
                .unwrap_unchecked()
        }
    }
}

impl<T: TensorElement> CachedTensorPromise<T> {
    pub fn new(op: OpKind<T>, inputs: Box<[NodeKind<T>]>) -> Result<Self, OpError> {
        let node = TensorGraphCacheNode::new(op, inputs);

        match node {
            Ok(node) => Ok(Self {
                graph: Arc::new(node),
            }),
            Err(err) => Err(err),
        }
    }

    pub fn with_layout(op: OpKind<T>, inputs: Box<[NodeKind<T>]>, layout: Layout) -> Self {
        Self {
            graph: Arc::new(TensorGraphCacheNode::with_layout(op, inputs, layout)),
        }
    }

    pub fn from_node(node: TensorGraphCacheNode<T>) -> Self {
        Self {
            graph: Arc::new(node),
        }
    }
}

impl<P: Promising<Output: TensorElement>> RawTensorPromise<P> {
    /// Execute the computation graph and return the result as a [`Tensor`].
    ///
    /// This is where the work actually happens. The planner analyses the graph,
    /// assigns buffers, and then the executor runs each op in dependency order,
    /// freeing intermediate results as soon as they're no longer needed.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Tensor;
    ///
    /// let t = Tensor::from_scalar(4.0_f64, &[3]);
    /// let result = (t.as_promise() - 1.0).materialize();
    /// assert_eq!(result.data(), &vec![3.0; 3]);
    /// ```
    pub fn materialize(self) -> Tensor<P::Output> {
        let data = self.graph.compute();

        Tensor::from_data(data)
    }
}

impl<T: ComputeWrapperSpec> CachedTensorPromise<T> {
    /// Return the cached result if it has already been computed, or `None` if
    /// [`.materialize()`] has not been called yet.
    ///
    /// [`.materialize()`]: CachedTensorPromise::materialize
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Tensor;
    ///
    /// let t = Tensor::from_scalar(1.0_f64, &[2]);
    /// let cached = t.as_promise().cache();
    ///
    /// assert!(cached.get_cache().is_none());
    /// let _ = (&cached + 0.0).materialize();
    /// assert!(cached.get_cache().is_some());
    /// ```
    pub fn get_cache(&self) -> Option<Tensor<T>> {
        if let Some(tensor) = self.graph.get_cache() {
            Some(Tensor::from_data(tensor.clone()))
        } else {
            None
        }
    }
}

impl<P: Promising> Dimension for RawTensorPromise<P> {
    #[inline]
    fn layout(&self) -> &Layout {
        self.graph.layout()
    }
}

impl<P: Promising> Clone for RawTensorPromise<P> {
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
        }
    }
}
