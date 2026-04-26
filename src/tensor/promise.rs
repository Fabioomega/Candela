use std::marker::PhantomData;
use std::sync::Arc;

use crate::tensor::definitions::NumberLike;
use crate::tensor::graph::{NodeKind, TensorGraphCacheNode, TensorGraphNode};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::tensor::Tensor;
use crate::tensor::traits::{Dimension, Promising};

pub type TensorPromise<T> = RawTensorPromise<TensorGraphNode<T>>;
pub type CachedTensorPromise<T> = RawTensorPromise<TensorGraphCacheNode<T>>;

pub struct RawTensorPromise<P> {
    pub(crate) graph: Arc<P>,
}

impl<T: NumberLike> TensorPromise<T> {
    pub fn new(op: OpKind<T>, inputs: Box<[NodeKind<T>]>) -> Self {
        Self {
            graph: Arc::new(TensorGraphNode::new(op, inputs)),
        }
    }

    pub fn with_layout(op: OpKind<T>, inputs: Box<[NodeKind<T>]>, layout: Layout) -> Self {
        Self {
            graph: Arc::new(TensorGraphNode::with_layout(op, inputs, layout)),
        }
    }

    pub fn cache(self) -> CachedTensorPromise<T> {
        CachedTensorPromise::new(OpKind::NoOp, [NodeKind::Node(self.graph)].into())
    }
}

impl<T: NumberLike> CachedTensorPromise<T> {
    pub fn new(op: OpKind<T>, inputs: Box<[NodeKind<T>]>) -> Self {
        Self {
            graph: Arc::new(TensorGraphCacheNode::new(op, inputs)),
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

impl<P: Promising<Output: NumberLike>> RawTensorPromise<P> {
    pub fn materialize(self) -> Tensor<P::Output> {
        let data = self.graph.compute();

        Tensor::from_data(data)
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
