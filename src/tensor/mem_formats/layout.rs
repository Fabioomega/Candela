use std::{hash::Hash, iter::zip};

use crate::tensor::{
    errors::OpError,
    internals::{calculate_adjacent_dim_stride, calculate_dim_stride},
    mem_formats::slice::{SliceInfo, SliceRange},
};

#[derive(Clone, Debug)]
pub struct Layout {
    pub(crate) shape: Box<[usize]>,
    pub(crate) stride: Box<[i32]>,
    pub(crate) adj_stride: Box<[i32]>,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

#[inline]
pub(crate) fn validate_shape(shape: &[usize]) -> Result<(), OpError> {
    if shape.is_empty() {
        return Err(OpError::ZeroRankShape);
    }
    Ok(())
}

impl Layout {
    pub fn new(shape: &[usize]) -> Self {
        validate_shape(shape).unwrap_or_else(|e| panic!("{}", e));
        let len: usize = shape.iter().product();

        Self {
            shape: shape.into(),
            stride: calculate_dim_stride(shape),
            adj_stride: vec![1; shape.len()].into_boxed_slice(),
            offset: 0,
            len,
        }
    }

    pub fn empty() -> Self {
        Self {
            shape: Box::new([0]),
            stride: Box::new([0]),
            adj_stride: Box::new([0]),
            offset: 0,
            len: 0,
        }
    }

    pub fn from_raw_parts(
        shape: Box<[usize]>,
        stride: Box<[i32]>,
        adj_stride: Box<[i32]>,
        offset: usize,
        len: usize,
    ) -> Self {
        Self {
            shape,
            stride,
            adj_stride,
            offset,
            len,
        }
    }

    pub fn from_strided(shape: &[usize], stride: &[i32], offset: usize) -> Self {
        validate_shape(shape).unwrap_or_else(|e| panic!("{}", e));
        debug_assert!(shape.len() == stride.len());

        let len: usize = shape.iter().product();

        Self {
            shape: shape.into(),
            stride: stride.into(),
            adj_stride: calculate_adjacent_dim_stride(stride, shape),
            offset,
            len,
        }
    }

    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;

        self
    }

    pub fn view(&self, shape: &[usize]) -> Result<Self, OpError> {
        if shape.iter().product::<usize>() != self.len() {
            return Err(OpError::InvalidViewShape);
        }
        if !self.is_contiguous() {
            return Err(OpError::NonContiguousView);
        }
        Ok(Layout::new(shape).with_offset(self.offset))
    }

    pub fn slice(&self, range: &[SliceRange]) -> Result<Self, OpError> {
        let info = SliceInfo::from_range(self, range)?;
        let len: usize = info.shape.iter().product();

        Ok(Self {
            shape: info.shape,
            stride: self.stride.clone(),
            adj_stride: info.adj_stride,
            offset: info.offset,
            len,
        })
    }

    pub fn transpose(&self) -> Self {
        let mut stride = self.stride.clone();
        let mut shape = self.shape.clone();

        for i in 0..stride.len() / 2 {
            let last = stride.len() - i - 1;

            let temp = stride[last];
            stride[last] = stride[i];
            stride[i] = temp;

            let temp = shape[last];
            shape[last] = shape[i];
            shape[i] = temp;
        }

        let adj_stride: Box<[i32]> = calculate_adjacent_dim_stride(&stride, &shape);

        Self {
            shape,
            stride,
            adj_stride,
            offset: self.offset,
            len: self.len,
        }
    }

    pub fn transpose_axes(&self, axes: &[usize]) -> Result<Self, OpError> {
        if axes.len() != self.stride.len() {
            return Err(OpError::NotEnoughAxes(self.stride.len(), axes.len()));
        }

        for (i, axis) in axes.iter().enumerate() {
            for axis_other in axes.iter().skip(i + 1) {
                if axis == axis_other {
                    return Err(OpError::AxesOutOfBounds);
                }
            }
        }

        let mut stride: Vec<i32> = Vec::with_capacity(self.stride.len());
        let mut shape: Vec<usize> = Vec::with_capacity(self.stride.len());

        for &axis in axes.iter() {
            if axis >= self.stride.len() {
                return Err(OpError::AxesOutOfBounds);
            }

            stride.push(self.stride[axis]);
            shape.push(self.shape[axis]);
        }

        let adj_stride = calculate_adjacent_dim_stride(&stride, &shape);

        Ok(Self {
            shape: shape.into_boxed_slice(),
            stride: stride.into_boxed_slice(),
            adj_stride,
            offset: self.offset,
            len: self.len,
        })
    }

    pub fn broadcast(&self, shape: &[usize]) -> Result<Self, OpError> {
        if shape.len() < self.shape.len() {
            return Err(OpError::CannotBroadcast);
        }

        for (s1, s2) in zip(shape.iter().rev(), self.shape.iter().rev()) {
            if *s2 != 1 && *s1 != *s2 {
                return Err(OpError::CannotBroadcast);
            }
        }

        let mut new_stride: Vec<i32> = Vec::with_capacity(shape.len());
        new_stride.extend(
            (self.shape.len()..shape.len())
                .map(|_| 0)
                .chain(self.stride.iter().cloned()),
        );

        let len = new_stride.len();

        for (dim, s) in self.shape.iter().rev().enumerate() {
            if *s == 1 {
                new_stride[len - dim - 1] = 0;
            }
        }

        let adj_stride = calculate_adjacent_dim_stride(&new_stride, shape);
        let len: usize = shape.iter().product();

        Ok(Self {
            shape: shape.into(),
            stride: new_stride.into_boxed_slice(),
            adj_stride,
            offset: self.offset,
            len,
        })
    }

    #[inline]
    pub fn shape_as_3d(&self) -> [usize; 3] {
        debug_assert!(!self.shape.is_empty(), "shape_as_3d requires rank >= 1");
        if self.shape.len() == 1 {
            [1, 1, self.shape[0]]
        } else if self.shape.len() == 2 {
            [1, self.shape[0], self.shape[1]]
        } else {
            let len = self.shape.len();

            let mut acc: usize = 1;
            for i in 0..len - 2 {
                acc *= self.shape[i];
            }

            [acc, self.shape[len - 2], self.shape[len - 1]]
        }
    }

    /// Cyclically rotates the axes so that iterating the layout walks along
    /// `axis`: `axis` becomes the innermost (fastest-varying) dimension, and
    /// every axis above it (`0..=axis`) is pulled inward as the surrounding
    /// block, so a flat iterator steps through all of their index combinations
    /// before advancing the trailing axes. The trailing axes (`axis+1..`) stay
    /// outermost in their original order.
    #[inline]
    pub fn rotate_axis_innermost(&self, axis: usize) -> Result<Self, OpError> {
        if axis >= self.shape().len() {
            return Err(OpError::AxesOutOfBounds);
        }

        let mut axes: Vec<usize> = (axis + 1..self.shape().len()).collect();
        axes.extend(0..=axis);

        unsafe { Ok(self.transpose_axes(&axes).unwrap_unchecked()) }
    }

    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.is_contiguous_at_axis(0)
    }

    #[inline]
    pub fn is_contiguous_at_axis(&self, axis: usize) -> bool {
        if axis >= self.shape().len() {
            return false;
        }

        self.adj_stride[axis] == 1 && !self.stride[axis + 1..].contains(&0)
    }

    #[inline]
    pub fn is_transposed(&self) -> bool {
        for (i, &adj_stride) in self.adj_stride.iter().enumerate() {
            if adj_stride < 0 && self.stride[i] != 0 {
                return true;
            }
        }

        false
    }

    #[inline]
    pub fn is_transposed_at_axis(&self, axis: usize) -> bool {
        if axis >= self.shape().len() {
            return false;
        }

        self.adj_stride[axis] < 0 && self.stride[axis] != 0
    }

    // Restricted to 2D on purpose: the matmul kernel uses this to pick the BLAS
    // trans-flag, and its batch-stride handling assumes there is no batch dim.
    // Higher-rank tensors whose last two strides happen to match this pattern
    // would silently feed an incoherent batch stride to GEMM.
    #[inline]
    pub fn is_last_axes_transposed(&self) -> bool {
        if self.shape.len() != 2 {
            return false;
        }

        let rs = self.stride[self.stride.len() - 2];
        let cs = self.stride[self.stride.len() - 1];

        // Gives false on broadcasting
        if rs == 0 || cs == 0 {
            return false;
        }

        // cs must be > 1: a contiguous [m, 1] matrix has rs=cs=1 and is not transposed
        rs == 1 && cs > 1
    }

    #[inline]
    pub fn shape(&self) -> &'_ [usize] {
        &self.shape
    }

    #[inline]
    pub fn stride(&self) -> &'_ [i32] {
        &self.stride
    }

    #[inline]
    pub fn adj_stride(&self) -> &'_ [i32] {
        &self.adj_stride
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl PartialEq for Layout {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.stride == other.stride
    }
}

impl Eq for Layout {}

impl Hash for Layout {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape.hash(state);
        self.stride.hash(state);
    }
}

#[cfg(test)]
#[path = "layout_tests.rs"]
mod tests;

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layout {{ shape: {:?}, stride: {:?}, offset: {} }}",
            &self.shape, &self.stride, self.offset
        )
    }
}
