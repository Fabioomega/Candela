use crate::tensor::backend::Backend;
use crate::tensor::graph::NodeKind;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::storage::TensorData;

pub trait Dimension {
    fn layout(&self) -> &Layout;

    fn shape(&self) -> &'_ [usize] {
        self.layout().shape()
    }

    fn stride(&self) -> &'_ [i32] {
        self.layout().stride()
    }

    fn adj_stride(&self) -> &'_ [i32] {
        self.layout().adj_stride()
    }

    fn len(&self) -> usize {
        self.layout().len()
    }

    fn is_empty(&self) -> bool {
        self.layout().is_empty()
    }

    fn offset(&self) -> usize {
        self.layout().offset()
    }

    fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    fn is_contiguous_at_axis(&self, axis: usize) -> bool {
        self.layout().is_contiguous_at_axis(axis)
    }

    fn is_transposed(&self) -> bool {
        self.layout().is_transposed()
    }

    fn is_transposed_at_axis(&self, axis: usize) -> bool {
        self.layout().is_transposed_at_axis(axis)
    }
}

pub trait Promising {
    type Output;

    fn compute(&self) -> TensorData<Self::Output>;
}

/// Represents any graph type that can take part in an op: it produces a graph node and
/// carries a layout. Implemented by every operand kind - `Tensor`,
/// `TensorPromise`, `CachedTensorPromise`, `BakedPromise`, `SkeletonSlot`,
/// `SkeletonPromise` - plus internal intermediates.
pub(crate) trait Operand<T, B: Backend>: Dimension {
    fn to_node(&self) -> NodeKind<T, B>;
}

/// A "materializable" subset of [`Operand`]: values that are legal as
/// concrete inputs when binding a skeleton via `compose`. `SkeletonSlot` and
/// `SkeletonPromise` are [`Operand`]s but not `Composable`. This
/// exclusion guarantees that no unbound slot appears in the materialization
/// path.
pub trait Composable<T, B: Backend>: Operand<T, B> {}

pub trait StreamingIterator {
    type Item<'a>
    where
        Self: 'a;

    fn next_stream<'a>(&'a mut self) -> Option<Self::Item<'a>>;

    #[allow(unused)]
    fn zip<Other>(self, other: Other) -> StreamingZip<Self, Other>
    where
        Self: Sized,
        Other: StreamingIterator,
    {
        StreamingZip {
            left: self,
            right: other,
        }
    }
}

#[allow(unused)]
pub struct StreamingZip<A: StreamingIterator, B: StreamingIterator> {
    left: A,
    right: B,
}

impl<A, B> StreamingIterator for StreamingZip<A, B>
where
    A: StreamingIterator,
    B: StreamingIterator,
{
    type Item<'a>
        = (A::Item<'a>, B::Item<'a>)
    where
        Self: 'a;

    fn next_stream<'a>(&'a mut self) -> Option<Self::Item<'a>> {
        let l = self.left.next_stream()?;
        let r = self.right.next_stream()?;
        Some((l, r))
    }
}

pub(crate) trait Numeric: crate::tensor::definitions::NumberLike {
    const MUL_NEUTRAL: Self;
    const SUM_NEUTRAL: Self;
    const ONE: Self;
    const ZERO: Self;
    const MIN: Self;
}

impl Numeric for f64 {
    const MUL_NEUTRAL: Self = 1.0;
    const SUM_NEUTRAL: Self = 0.0;
    const ONE: Self = 1.0;
    const ZERO: Self = 0.0;
    const MIN: Self = f64::NEG_INFINITY;
}

impl Numeric for f32 {
    const MUL_NEUTRAL: Self = 1.0;
    const SUM_NEUTRAL: Self = 0.0;
    const ONE: Self = 1.0;
    const ZERO: Self = 0.0;
    const MIN: Self = f32::NEG_INFINITY;
}

pub(crate) trait FromIndex {
    fn from_index(i: usize) -> Self;
}

impl FromIndex for f64 {
    #[inline]
    fn from_index(i: usize) -> Self {
        i as f64
    }
}

impl FromIndex for f32 {
    #[inline]
    fn from_index(i: usize) -> Self {
        i as f32
    }
}
