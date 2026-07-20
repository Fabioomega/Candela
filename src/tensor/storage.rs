use std::fmt::Display;
use std::iter::zip;
use std::ops::Index;
use std::sync::Arc;

use crate::tensor::MutIter;
use crate::tensor::definitions::ChunkedIter;
use crate::tensor::iter::{
    ChunkedContiguousIter, ChunkedSliceIter, ContiguousIter, InformedIter, Iter, MutSliceIter,
    StepInfo,
};
use crate::tensor::mem_formats::layout::{Layout, validate_shape};
use crate::tensor::traits::Dimension;
use crate::{OpError, SliceRange};

pub enum IterImpl<C, N> {
    Contiguous(C),
    NotContiguous(N),
}

//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct Storage<T> {
    pub(crate) buffer: Arc<Vec<T>>,
}

impl<T: Clone> Storage<T> {
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
    pub fn data(&self) -> &[T] {
        &self.buffer
    }

    #[inline]
    pub fn mut_data(&mut self) -> Option<&mut [T]> {
        Arc::get_mut(&mut self.buffer).map(|buffer| buffer.as_mut_slice())
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.buffer.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> Option<*mut T> {
        Arc::get_mut(&mut self.buffer).map(|buffer| buffer.as_mut_ptr())
    }

    #[inline]
    pub fn deep_clone(&self) -> Self {
        let buffer = self.buffer.to_vec();
        Storage::from_vec(buffer)
    }
}

impl<T: Clone> Clone for Storage<T> {
    fn clone(&self) -> Self {
        Storage::from_arc(self.buffer.clone())
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct TensorData<T> {
    pub(crate) storage: Storage<T>,
    layout: Layout,
}

impl<T: Clone> TensorData<T> {
    #[inline]
    pub fn new(storage: Storage<T>, layout: Layout) -> Self {
        Self { storage, layout }
    }

    #[inline]
    pub fn from_scalar(scalar: T, shape: &[usize]) -> Self {
        validate_shape(shape).unwrap_or_else(|e| panic!("{}", e));
        let len: usize = shape.iter().product();

        Self {
            storage: Storage::from_scalar(scalar, len),
            layout: Layout::new(shape),
        }
    }

    #[inline]
    pub fn from_arc(buffer: Arc<Vec<T>>, shape: &[usize]) -> Self {
        validate_shape(shape).unwrap_or_else(|e| panic!("{}", e));
        Self {
            storage: Storage::from_arc(buffer),
            layout: Layout::new(shape),
        }
    }

    #[inline]
    pub fn from_vec(vector: Vec<T>, shape: &[usize], offset: usize) -> Self {
        validate_shape(shape).unwrap_or_else(|e| panic!("{}", e));
        let expected: usize = shape.iter().product();
        assert!(
            vector.len() == expected,
            "buffer length {} does not match shape {:?} (product {})",
            vector.len(),
            shape,
            expected
        );

        Self {
            storage: Storage::from_vec(vector),
            layout: Layout::new(shape).with_offset(offset),
        }
    }

    #[inline]
    pub fn from_vec_with_layout(vector: Vec<T>, layout: Layout) -> Self {
        assert!(
            layout.stride().iter().all(|s| *s >= 0),
            "negative strides are not supported"
        );
        assert!(
            layout.last() < vector.len(),
            "the layout indexes outside of the buffer"
        );

        Self {
            storage: Storage::from_vec(vector),
            layout,
        }
    }

    #[inline]
    pub fn from_iter<I>(iter: I, shape: &[usize]) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        validate_shape(shape).unwrap_or_else(|e| panic!("{}", e));
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
    pub fn data(&self) -> &[T] {
        self.storage.data()
    }

    #[inline]
    pub fn mut_data(&mut self) -> Option<&mut [T]> {
        self.storage.mut_data()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.storage.as_ptr().wrapping_add(self.offset())
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> Option<*mut T> {
        self.storage
            .as_mut_ptr()
            .map(|ptr| ptr.wrapping_add(self.offset()))
    }

    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter::new(&self.storage.buffer, self.layout())
    }

    #[inline]
    pub fn iter_slice_mut(&mut self) -> Option<MutSliceIter<'_, T>> {
        if let Some(data) = Arc::get_mut(&mut self.storage.buffer) {
            Some(MutSliceIter::new(data, self.layout.len, &self.layout))
        } else {
            None
        }
    }

    #[inline]
    pub fn iter_mut(&mut self) -> Option<MutIter<'_, T>> {
        if let Some(data) = Arc::get_mut(&mut self.storage.buffer) {
            Some(MutIter::new(data, &self.layout))
        } else {
            None
        }
    }

    /// Iterate over the backing buffer using `layout` instead of this storage's own
    /// layout. Useful for traversals more exotic than the safe interface exposes.
    ///
    /// # Safety
    ///
    /// `layout` must be a valid transformation of this storage's current layout -
    /// every index it addresses must fall within the backing buffer. A layout
    /// derived from this storage's layout (a view, slice, transpose, or broadcast
    /// of it) upholds this; an unrelated layout may read out of bounds and is
    /// undefined behaviour. The `debug_assert!` below is only a coarse guard, not
    /// a full bounds check.
    #[inline]
    pub unsafe fn iter_as_layout<'a>(&'a self, layout: &'a Layout) -> Iter<'a, T> {
        debug_assert!(
            self.layout().len() >= layout.len() && self.layout.offset() >= layout.offset()
        );
        Iter::new(&self.storage.buffer, layout)
    }

    #[inline]
    pub fn fast_iter(&self) -> IterImpl<ContiguousIter<'_, T>, Iter<'_, T>> {
        let buffer = &self.storage.buffer;

        if self.is_contiguous() {
            IterImpl::Contiguous(ContiguousIter::new(buffer, self.offset(), self.len()))
        } else {
            IterImpl::NotContiguous(Iter::new(buffer, self.layout()))
        }
    }

    #[inline]
    pub fn informed_iter(&self) -> InformedIter<'_, T> {
        InformedIter::new(&self.storage.buffer, &self.layout)
    }

    #[inline]
    pub fn deep_clone(&self) -> Self {
        Self {
            storage: self.storage.deep_clone(),
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
    pub fn get(&self, index: &[usize]) -> Result<&T, OpError> {
        if self.layout.shape().len() != index.len() {
            return Err(OpError::NotEnoughAxes(self.layout.shape().len(), index.len()));
        }

        let mut pos: i64 = 0;

        for (i, (&stride, &step)) in zip(self.layout.stride(), index).enumerate() {
            if step >= self.shape()[i] {
                return Err(OpError::IndexOutOfBounds);
            }

            pos += stride as i64 * step as i64;
        }

        Ok(unsafe { &(*self.as_ptr().wrapping_add(pos as usize)) })
    }

    #[inline]
    pub fn item(&self) -> &T {
        unsafe { &(*self.as_ptr()) }
    }

    #[inline]
    pub fn as_contiguous(&self) -> Self {
        if !self.is_contiguous() {
            Self::from_iter(self.iter().cloned(), self.shape())
        } else {
            self.deep_clone()
        }
    }

    #[inline]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T: Clone + Default> TensorData<T> {
    #[inline]
    pub fn packed_iter(&self, packing_buffer_size: usize) -> ChunkedIter<'_, T> {
        ChunkedSliceIter::new(self.iter().cloned(), packing_buffer_size)
    }

    #[inline]
    pub fn fast_packed_iter(
        &self,
        packing_buffer_size: usize,
    ) -> IterImpl<ChunkedContiguousIter<'_, T>, ChunkedIter<'_, T>> {
        if self.is_contiguous() {
            let start = self.offset();
            IterImpl::Contiguous(ChunkedContiguousIter::new(
                &self.data()[start..start + self.len()],
                packing_buffer_size,
            ))
        } else {
            IterImpl::NotContiguous(ChunkedSliceIter::new(
                self.iter().cloned(),
                packing_buffer_size,
            ))
        }
    }
}

impl<T: Clone> Clone for TensorData<T> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            layout: self.layout.clone(),
        }
    }
}

impl<T> Dimension for TensorData<T> {
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

impl<T> Index<&[usize]> for TensorData<T>
where
    T: Copy,
{
    type Output = T;

    fn index(&self, index: &[usize]) -> &Self::Output {
        self.get(index).expect("index is out of bounds, probably")
    }
}

impl<T: std::fmt::Display + Copy> std::fmt::Display for TensorData<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut indent = 0;
        let mut in_seq = false;

        debug_assert!(!self.shape().is_empty(), "TensorData rank must be >= 1");
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
