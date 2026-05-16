use std::iter::zip;

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

impl Layout {
    pub fn new(
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

    pub fn empty() -> Self {
        Self {
            shape: Box::new([0]),
            stride: Box::new([0]),
            adj_stride: Box::new([0]),
            offset: 0,
            len: 0,
        }
    }

    pub fn from_shape(shape: &[usize], offset: usize) -> Self {
        let len: usize = shape.iter().product();

        Self {
            shape: shape.into(),
            stride: calculate_dim_stride(shape),
            adj_stride: vec![1; shape.len()].into_boxed_slice(),
            offset,
            len,
        }
    }

    pub fn from_slice(shape: &[usize], stride: &[i32], offset: usize) -> Self {
        let len: usize = shape.iter().product();

        Self {
            shape: shape.into(),
            stride: stride.into(),
            adj_stride: calculate_adjacent_dim_stride(stride, shape),
            offset,
            len,
        }
    }

    pub fn view(&self, shape: &[usize]) -> Result<Self, OpError> {
        if shape.iter().product::<usize>() != self.len() {
            return Err(OpError::InvalidViewShape);
        }
        if !self.is_contiguous() {
            return Err(OpError::NonContiguousView);
        }
        Ok(Layout::from_shape(shape, self.offset))
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
                    return Err(OpError::OutOfBoundAxes);
                }
            }
        }

        let mut stride: Vec<i32> = Vec::with_capacity(self.stride.len());
        let mut shape: Vec<usize> = Vec::with_capacity(self.stride.len());

        for &axis in axes.iter() {
            if axis >= self.stride.len() {
                return Err(OpError::OutOfBoundAxes);
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

        if shape.len() == self.shape.len() {
            for (s1, s2) in zip(shape.iter(), self.shape.iter()) {
                if *s1 % *s2 != 0 {
                    return Err(OpError::CannotBroadcast);
                }
            }
        }

        for (s1, s2) in zip(shape.iter().rev(), self.shape.iter().rev()) {
            if *s1 % *s2 != 0 {
                return Err(OpError::CannotBroadcast);
            }
        }

        let mut new_stride: Vec<i32> = Vec::with_capacity(shape.len());
        new_stride.extend(
            (self.shape.len()..shape.len())
                .into_iter()
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

    #[inline]
    pub fn to_dim_stride(&self, dim: usize) -> Result<Self, OpError> {
        if dim >= self.shape().len() {
            return Err(OpError::OutOfBoundAxes);
        }

        let mut axes = self.shape.to_vec();
        axes.remove(dim);
        axes.push(dim);

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

    // This function is needed because the .is_transposed_at_axis just tells if a tensor is transposed in any way.
    // We are checking for a specific case.
    #[inline]
    pub fn is_last_axes_transposed(&self) -> bool {
        if self.shape.len() < 2 {
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
