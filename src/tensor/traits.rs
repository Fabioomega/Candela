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

    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>>;
}
