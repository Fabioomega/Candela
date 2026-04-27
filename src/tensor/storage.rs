use std::sync::Arc;

use crate::tensor::iter::{
    ChunkedSliceIter, ContiguousIter, CopiedContiguousIter, CopiedSliceIter, InformedSliceIter,
    SliceIter,
};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::traits::Dimension;
use crate::{debug_assert_positive, impl_display};

pub enum IterImpl<C, N> {
    Contiguous(C),
    NotContiguous(N),
}

//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct Storage<T: Copy> {
    pub(crate) buffer: Arc<Vec<T>>,
}

impl<T: Copy> Storage<T> {
    #[inline]
    pub fn from_scalar(scalar: T, len: usize) -> Self {
        Self {
            buffer: Arc::new(vec![scalar; len]),
        }
    }

    #[inline]
    pub fn from_arc(buffer: Arc<Vec<T>>) -> Self {
        Self { buffer }
    }

    #[inline]
    pub fn from_vec(vector: Vec<T>) -> Self {
        Self {
            buffer: Arc::new(vector),
        }
    }

    #[inline]
    pub fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let vector = std::vec::Vec::from_iter(iter);
        Self::from_vec(vector)
    }

    #[inline]
    pub fn clone_reference(&self) -> Self {
        Storage::from_arc(self.buffer.clone())
    }
}

impl<T: Copy> Clone for Storage<T> {
    fn clone(&self) -> Self {
        let buffer = self.buffer.to_vec();
        Storage::from_vec(buffer)
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct TensorData<T: Copy> {
    pub(crate) storage: Storage<T>,
    layout: Layout,
    pub(crate) reusable: bool,
}

impl<T: Copy> TensorData<T> {
    #[inline]
    pub fn new(storage: Storage<T>, layout: Layout) -> Self {
        Self {
            storage,
            layout,
            reusable: false,
        }
    }

    #[inline]
    pub fn from_scalar(scalar: T, shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();

        debug_assert_positive!(len);

        Self {
            storage: Storage::from_scalar(scalar, len as usize),
            layout: Layout::from_shape(shape, 0),
            reusable: false,
        }
    }

    #[inline]
    pub fn from_arc(buffer: Arc<Vec<T>>, shape: &[usize]) -> Self {
        Self {
            storage: Storage::from_arc(buffer),
            layout: Layout::from_shape(shape, 0),
            reusable: false,
        }
    }

    #[inline]
    pub fn from_vec(vector: Vec<T>, shape: &[usize], offset: usize) -> Self {
        debug_assert!(vector.len() <= (shape.iter().product()));

        Self {
            storage: Storage::from_vec(vector),
            layout: Layout::from_shape(shape, offset),
            reusable: false,
        }
    }

    #[inline]
    pub fn from_iter<I>(iter: I, shape: &[usize]) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let vector = std::vec::Vec::from_iter(iter);
        Self::from_vec(vector, shape, 0)
    }

    #[inline]
    pub fn as_layout(&self, layout: Layout) -> Self {
        Self {
            storage: self.storage.clone_reference(),
            layout,
            reusable: self.reusable,
        }
    }

    #[inline]
    pub fn iter(&self) -> SliceIter<'_, T> {
        SliceIter::new(&self.storage.buffer, self.len(), self.layout())
    }

    #[inline]
    pub unsafe fn iter_as_layout<'a>(&'a self, layout: &'a Layout) -> SliceIter<'a, T> {
        SliceIter::new(&self.storage.buffer, layout.len(), layout)
    }

    #[inline]
    pub fn fast_iter(&self) -> IterImpl<ContiguousIter<'_, T>, SliceIter<'_, T>> {
        let buffer = &self.storage.buffer;

        if self.is_contiguous() {
            IterImpl::Contiguous(ContiguousIter::new(buffer, self.offset(), self.len()))
        } else {
            IterImpl::NotContiguous(SliceIter::new(buffer, self.len(), self.layout()))
        }
    }

    #[inline]
    pub fn copied_iter(&self) -> CopiedSliceIter<'_, T> {
        CopiedSliceIter::new(&self.storage.buffer, self.len(), self.layout())
    }

    #[inline]
    pub fn copied_fast_iter(
        &self,
    ) -> IterImpl<CopiedContiguousIter<'_, T>, CopiedSliceIter<'_, T>> {
        let buffer = &self.storage.buffer;

        if self.is_contiguous() {
            IterImpl::Contiguous(CopiedContiguousIter::new(buffer, self.offset(), self.len()))
        } else {
            IterImpl::NotContiguous(CopiedSliceIter::new(buffer, self.len(), self.layout()))
        }
    }

    #[inline]
    pub fn informed_iter(&self) -> InformedSliceIter<'_, T> {
        InformedSliceIter::new(&self.storage.buffer, &self.layout)
    }

    #[inline]
    pub fn clone_deep(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            layout: self.layout.clone(),
            reusable: self.reusable,
        }
    }

    #[inline]
    pub fn as_contiguous(&self) -> Self {
        if !self.is_contiguous() {
            Self::from_iter(self.copied_iter(), self.shape())
        } else {
            self.clone_deep()
        }
    }

    #[inline]
    pub fn mark_as_reusable(mut self) -> Self {
        self.reusable = true;

        self
    }

    #[inline]
    pub fn mark_as_not_reusable(mut self) -> Self {
        self.reusable = false;

        self
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T: Copy + Default> TensorData<T> {
    #[inline]
    pub fn packed_iter(&self) -> crate::tensor::definitions::ChunkedIter<'_, T> {
        ChunkedSliceIter::new(self.copied_iter())
    }
}

impl<T: Copy> Clone for TensorData<T> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone_reference(),
            layout: self.layout.clone(),
            reusable: self.reusable,
        }
    }
}

impl<T: Copy> Dimension for TensorData<T> {
    #[inline]
    fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl_display!(TensorData<T>);
