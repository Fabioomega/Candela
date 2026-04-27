use crate::impl_display;
use crate::tensor::errors::OpError;
use crate::tensor::graph::{NodeKind, TensorGraphEdge};
use crate::tensor::iter::{ContiguousIter, InformedSliceIter, SliceIter};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::promise::TensorPromise;
use crate::tensor::storage::TensorData;
use crate::tensor::traits::{Dimension, Promising};
use std::sync::Arc;

pub struct Tensor<T: Copy> {
    pub(crate) graph: Arc<TensorGraphEdge<T>>,
}

impl<T: Copy> Tensor<T> {
    #[inline]
    pub fn from_scalar(scalar: T, shape: &[usize]) -> Self {
        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(TensorData::from_scalar(
                scalar, shape,
            ))),
        }
    }

    #[inline]
    pub fn from_vec(vector: Vec<T>, shape: &[usize]) -> Self {
        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(TensorData::from_vec(
                vector, shape, 0,
            ))),
        }
    }

    #[inline]
    pub fn from_iter<I>(iter: I, shape: &[usize]) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let vector: Vec<T> = std::vec::Vec::from_iter(iter);
        Self::from_vec(vector, shape)
    }

    #[inline]
    pub fn from_data(data: TensorData<T>) -> Self {
        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(data)),
        }
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

    #[inline]
    /// Makes a deep copy of this tensor.
    pub fn clone_deep(&self) -> Self {
        let data = self.graph.get();

        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(data.clone_deep())),
        }
    }

    #[inline]
    /// Make a shallow copy of this tensor.
    /// That means that the underlying memory is, or may be, shared with other objects.
    /// The shallow copy does not maintain connection with promises depending on this tensor.
    /// It will be treated as a completely different tensor during topological sorting
    /// and materialization.
    /// If you want to create a shallow copy while maintaining connection with existing promises
    /// use .clone() instead.
    pub fn clone_detached(&self) -> Self {
        let data = self.graph.get();

        Self {
            graph: Arc::new(TensorGraphEdge::from_tensor_data(data.clone())),
        }
    }
}

impl<T: NumberLike> Tensor<T> {
    #[inline]
    pub fn as_promise(&self) -> TensorPromise<T> {
        unsafe {
            TensorPromise::new(
                super::ops::def_op::OpKind::NoOp,
                [NodeKind::Edge(self.graph.clone())].into(),
            )
            .unwrap_unchecked()
        }
    }
}

impl<T: Copy> Dimension for Tensor<T> {
    #[inline]
    fn layout(&self) -> &super::mem_formats::layout::Layout {
        self.graph.layout()
    }
}

/// Make a shallow copy of this tensor.
/// That means that the underlying memory is, or may be, shared with other objects.
/// The shallow copy still maintain connection with all the promises depending on this tensor
/// If you want to create a shallow copy without any connection with existing promises
/// use clone_detached() instead, or clone_deep for a deep clone.
impl<T: Copy> Clone for Tensor<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
        }
    }
}

impl_display!(Tensor<T>);
