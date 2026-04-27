use crate::tensor::{
    errors::OpError,
    internals::{calculate_adjacent_dim_stride, calculate_dim_stride},
    mem_formats::slice::{SliceInfo, SliceRange},
};

use crate::cfg_debug_only;

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
        cfg_debug_only!({
            let size: usize = shape.iter().product();
            if size != self.len() {
                return Err(OpError::InvalidViewShape);
            }

            if !self.is_contiguous() {
                return Err(OpError::NonContiguousView);
            }
        });

        Ok(Layout::from_shape(shape, self.offset))
    }

    pub fn slice(&self, range: &[SliceRange]) -> Result<Self, OpError> {
        let info = SliceInfo::from_range(self, range);

        cfg_debug_only!(if let Err(err) = info {
            return Err(err);
        });

        let unwrapped_info = unsafe { info.unwrap_unchecked() };

        let len: usize = unwrapped_info.shape.iter().product();

        Ok(Self {
            shape: unwrapped_info.shape,
            stride: self.stride.clone(),
            adj_stride: unwrapped_info.adj_stride,
            offset: unwrapped_info.offset,
            len,
        })
    }

    pub fn transpose(&self) -> Self {
        let mut stride = self.stride.clone();
        let mut shape = self.shape.clone();

        for i in 0..stride.len() / 2 {
            let last = stride.len() - i - 1;

            let mut temp = stride[last];
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
        cfg_debug_only!(if axes.len() != self.stride.len() {
            return Err(OpError::NotEnoughAxes(self.stride.len(), axes.len()));
        });

        let mut stride: Vec<i32> = Vec::with_capacity(self.stride.len());
        let mut shape: Vec<usize> = Vec::with_capacity(self.stride.len());

        for &axis in axes.iter() {
            cfg_debug_only!(if axis >= self.stride.len() {
                return Err(OpError::OutOfBoundAxes);
            });

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

    // pub fn broadcast_to_shape(&self, shape: &[usize]) -> Result<Self, OpError> {
    //     cfg_debug_only!(
    //         if shape.len() > self.shape.len() && shape[0] % self.shape[0] == 0 {
    //             return Err(OpError::CannotBroadcast);
    //         }
    //     );
    //     let diff = shape.len() - self.shape.len();

    //     let mut stride: Vec<i32> = Vec::new();
    //     stride.extend((0..diff).map(|_| 0));
    //     stride.extend_from_slice(shape);

    //     let adj_stride = calculate_adjacent_dim_stride(&stride, shape);
    //     let len: i32 = self.shape().iter().product();
    //     let len: usize = len as usize;

    //     Ok(Self {
    //         shape: shape.into(),
    //         stride: stride.into_boxed_slice(),
    //         adj_stride,
    //         offset: self.offset,
    //         len,
    //     })
    // }

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

    pub fn to_dim_stride(&self, dim: usize) -> Result<Self, OpError> {
        if dim >= self.shape().len() {
            return Err(OpError::OutOfBoundAxes);
        }

        let mut temp = self.clone();
        let mut new_len: usize = 1;
        for i in 0..temp.shape().len() {
            if i > dim && self.shape[i] > 0 {
                temp.shape[i] = 1;
            }
            if i <= dim {
                new_len *= self.shape[i] as usize;
            }
        }

        temp.adj_stride = calculate_adjacent_dim_stride(&temp.stride, &temp.shape);
        temp.len = new_len;

        Ok(temp)
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

        self.adj_stride[axis] == 1
    }

    #[inline]
    pub fn is_transposed(&self) -> bool {
        for &adj_stride in &self.adj_stride {
            if adj_stride < 0 {
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

        self.adj_stride[axis] < 0
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

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layout {{ shape: {:?}, stride: {:?}, offset: {} }}",
            &self.shape, &self.stride, self.offset
        )
    }
}
