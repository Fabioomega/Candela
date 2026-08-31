use std::iter::zip;
use std::ops::Index;
use std::sync::Arc;

use crate::tensor::iter::{InformedIter, Iter, StepInfo};
use crate::tensor::mem_formats::layout::{Layout, validate_shape};
use crate::tensor::traits::Dimension;
use crate::tensor::walker::all2;
use crate::{OpError, SliceRange};

//////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: When handling type casting we need to change this
// to account for alignment.
#[derive(Debug)]
enum StorageKind<T> {
    Global(Arc<Vec<T>>),
    Arena { base: *mut T, len: usize },
}

// SAFETY: The `StorageKind::Arena` never scapes run-plan so any user-observable storage
// is `StorageKind::Global`.
unsafe impl<T: Send> Send for StorageKind<T> {}
unsafe impl<T: Sync> Sync for StorageKind<T> {}

//////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct Storage<T> {
    storage: StorageKind<T>,
}

impl<T: Clone> Storage<T> {
    #[inline]
    pub fn from_scalar(scalar: T, len: usize) -> Self {
        Self {
            storage: StorageKind::Global(Arc::new(vec![scalar; len])),
        }
    }

    #[inline]
    pub fn from_arc(buffer: Arc<Vec<T>>) -> Self {
        Self {
            storage: StorageKind::Global(buffer),
        }
    }

    #[inline]
    pub fn from_vec(vector: Vec<T>) -> Self {
        Self {
            storage: StorageKind::Global(Arc::new(vector)),
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
    pub(crate) unsafe fn from_raw_parts(base: *mut T, len: usize) -> Self {
        Self {
            storage: StorageKind::Arena { base, len },
        }
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        match &self.storage {
            StorageKind::Global(buffer) => buffer,
            StorageKind::Arena { base, len } => unsafe { std::slice::from_raw_parts(*base, *len) },
        }
    }

    #[inline]
    pub fn mut_data(&mut self) -> Option<&mut [T]> {
        match &mut self.storage {
            StorageKind::Global(buffer) => Arc::get_mut(buffer).map(|buffer| buffer.as_mut()),
            StorageKind::Arena { base, len } => unsafe {
                Some(std::slice::from_raw_parts_mut(*base, *len))
            },
        }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data().as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> Option<*mut T> {
        self.mut_data().map(|data| data.as_mut_ptr())
    }

    #[inline]
    pub fn deep_clone(&self) -> Self {
        if let StorageKind::Global(buffer) = &self.storage {
            let b = buffer.to_vec();
            Storage::from_vec(b)
        } else {
            unreachable!("this function can only be called when the buffer is not arena-based");
        }
    }

    #[inline]
    pub fn is_arena_backed(&self) -> bool {
        match self.storage {
            StorageKind::Arena { .. } => true,
            _ => false,
        }
    }
}

impl<T: Clone> Clone for Storage<T> {
    fn clone(&self) -> Self {
        match &self.storage {
            StorageKind::Global(buffer) => Storage::from_arc(buffer.clone()),
            StorageKind::Arena { base, len } => unsafe { Storage::from_raw_parts(*base, *len) },
        }
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
        Iter::new(self.storage.data(), self.layout())
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
        Iter::new(self.storage.data(), layout)
    }

    #[inline]
    pub fn informed_iter(&self) -> InformedIter<'_, T> {
        InformedIter::new(self.storage.data(), &self.layout)
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
            return Err(OpError::NotEnoughAxes(
                self.layout.shape().len(),
                index.len(),
            ));
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
    T: Copy + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if self.layout.len() != other.layout.len() {
            return false;
        }

        all2(
            self.data(),
            self.layout(),
            other.data(),
            other.layout(),
            |&el1, &el2| el1 == el2,
        )
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
