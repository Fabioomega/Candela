use std::fmt::Display;
use std::iter::zip;
use std::sync::Arc;

use crate::tensor::iter::{
    ChunkedSliceIter, ContiguousIter, CopiedContiguousIter, CopiedSliceIter, InformedSliceIter,
    MutSliceIter, SliceIter,
};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::traits::Dimension;
use crate::{SliceRange, Tensor, debug_assert_positive, impl_display};

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
    pub fn data(&self) -> &Vec<T> {
        &self.buffer
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.buffer.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> Option<*mut T> {
        if let Some(buffer) = Arc::get_mut(&mut self.buffer) {
            Some(buffer.as_mut_ptr())
        } else {
            None
        }
    }

    #[inline]
    pub fn clone_deep(&self) -> Self {
        let buffer = self.buffer.to_vec();
        Storage::from_vec(buffer)
    }
}

impl<T: Copy> Clone for Storage<T> {
    fn clone(&self) -> Self {
        Storage::from_arc(self.buffer.clone())
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct TensorData<T: Copy> {
    pub(crate) storage: Storage<T>,
    layout: Layout,
}

impl<T: Copy> TensorData<T> {
    #[inline]
    pub fn new(storage: Storage<T>, layout: Layout) -> Self {
        Self { storage, layout }
    }

    #[inline]
    pub fn from_scalar(scalar: T, shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();

        debug_assert_positive!(len);

        Self {
            storage: Storage::from_scalar(scalar, len as usize),
            layout: Layout::from_shape(shape, 0),
        }
    }

    #[inline]
    pub fn from_arc(buffer: Arc<Vec<T>>, shape: &[usize]) -> Self {
        Self {
            storage: Storage::from_arc(buffer),
            layout: Layout::from_shape(shape, 0),
        }
    }

    #[inline]
    pub fn from_vec(vector: Vec<T>, shape: &[usize], offset: usize) -> Self {
        debug_assert!(vector.len() <= (shape.iter().product()));

        Self {
            storage: Storage::from_vec(vector),
            layout: Layout::from_shape(shape, offset),
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
            storage: self.storage.clone(),
            layout,
        }
    }

    #[inline]
    pub fn into_layout(mut self, layout: Layout) -> Self {
        self.layout = layout;

        self
    }

    #[inline]
    pub fn data(&self) -> &Vec<T> {
        self.storage.data()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.storage.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> Option<*mut T> {
        self.storage.as_mut_ptr()
    }

    #[inline]
    pub fn iter(&self) -> SliceIter<'_, T> {
        SliceIter::new(&self.storage.buffer, self.len(), self.layout())
    }

    #[inline]
    pub fn iter_mut(&mut self) -> Option<MutSliceIter<'_, T>> {
        if let Some(data) = Arc::get_mut(&mut self.storage.buffer) {
            Some(MutSliceIter::new(data, self.layout.len, &self.layout))
        } else {
            None
        }
    }

    #[inline]
    pub unsafe fn iter_as_layout<'a>(&'a self, layout: &'a Layout) -> SliceIter<'a, T> {
        debug_assert!(self.layout().len() == layout.len());
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
            storage: self.storage.clone_deep(),
            layout: self.layout.clone(),
        }
    }

    #[inline]
    // This is an internal method and should be, mostly used for
    //  tests and the like. DO NOT USE IT IN ANYTHING USER FACING!
    pub fn slice(&self, range: &[SliceRange]) -> Self {
        let lay = self.layout.slice(range).unwrap();

        self.as_layout(lay)
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
            storage: self.storage.clone(),
            layout: self.layout.clone(),
        }
    }
}

impl<T: Copy> Dimension for TensorData<T> {
    #[inline]
    fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T> PartialEq for TensorData<T>
where
    T: Copy + PartialEq + Display,
{
    fn eq(&self, other: &Self) -> bool {
        if self.layout.len() != other.layout.len() {
            return false;
        }

        for (el1, el2) in zip(self.iter(), other.iter()) {
            if *el1 != *el2 {
                return false;
            }
        }

        true
    }
}

impl_display!(TensorData<T>);
