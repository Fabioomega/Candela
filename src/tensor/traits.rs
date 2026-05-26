use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::storage::TensorData;

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
    type Output: Copy;

    fn compute(&self) -> TensorData<Self::Output>;

    fn layout(&self) -> &Layout;
}

pub trait StreamingIterator {
    type Item<'a>
    where
        Self: 'a;

    fn next_stream<'a>(&'a mut self) -> Option<Self::Item<'a>>;

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
}

impl Numeric for f64 {
    const MUL_NEUTRAL: Self = 1.0;
    const SUM_NEUTRAL: Self = 0.0;
    const ONE: Self = 1.0;
    const ZERO: Self = 0.0;
}

impl Numeric for f32 {
    const MUL_NEUTRAL: Self = 1.0;
    const SUM_NEUTRAL: Self = 0.0;
    const ONE: Self = 1.0;
    const ZERO: Self = 0.0;
}
