use std::{hash::Hash, iter::zip};

use crate::tensor::{
    IntoShape, MAX_DIMS,
    errors::OpError,
    internals::{calculate_adjacent_dim_stride, calculate_dim_stride},
    mem_formats::slice::{SliceInfo, SliceRange},
};

/// How a tensor's logical shape maps onto its flat backing buffer.
///
/// A `Layout` bundles the `shape`, the per-axis `stride` (how many buffer
/// elements to step to advance one index along that axis), an `offset` into the
/// buffer, and a cached total `len`. Views, slices, transposes, and broadcasts
/// are all just new layouts over the *same* buffer, which is what makes those
/// operations zero-copy. See the [layout docs](crate::docs::layout) for the full
/// model, including the `adj_stride` iteration trick.
///
/// You rarely build one by hand: a tensor's layout fields are reachable directly
/// through the [`Dimension`](crate::Dimension) trait (`t.shape()`, `t.stride()`,
/// `t.is_contiguous()`, …), and `t.layout()` hands back the whole `Layout`.
/// Constructing one explicitly is mostly useful for shaping a skeleton slot.
///
/// # Examples
///
/// ```
/// use candela::{Dimension, Layout, Tensor};
///
/// // Shape/stride are available straight off the tensor via `Dimension`.
/// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
/// assert_eq!(t.shape(), &[2, 3]);
/// assert_eq!(t.layout(), &Layout::new(&[2, 3]));
///
/// // Or build one directly.
/// let l = Layout::new(&[2, 3]);
/// assert_eq!(l.stride(), &[3, 1]);
/// assert!(l.is_contiguous());
/// ```
#[derive(Clone, Debug)]
pub struct Layout {
    pub(crate) shape: [usize; MAX_DIMS],
    pub(crate) stride: [i32; MAX_DIMS],
    pub(crate) adj_stride: [i32; MAX_DIMS],
    pub(crate) offset: usize,
    pub(crate) rank: usize,
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
    /// Build a contiguous, row-major layout for `shape`.
    ///
    /// # Panics
    ///
    /// Panics if `shape` is empty (a tensor must have rank >= 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// let l = Layout::new((2, 3));
    /// assert_eq!(l.shape(), &[2, 3]);
    /// assert_eq!(l.stride(), &[3, 1]);
    /// assert_eq!(l.len(), 6);
    /// ```
    pub fn new(shape: impl IntoShape) -> Self {
        let (rank, shape) = shape.into_shape();

        validate_shape(&shape[..rank]).unwrap_or_else(|e| panic!("{}", e));
        let len: usize = shape[..rank].iter().product();

        let stride = calculate_dim_stride(&shape[..rank]);

        Self {
            shape,
            stride,
            adj_stride: [1; MAX_DIMS],
            offset: 0,
            rank,
            len,
        }
    }

    /// Build the empty layout: shape `[0]`, length `0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// assert!(Layout::empty().is_empty());
    /// ```
    pub fn empty() -> Self {
        Self {
            shape: [0; MAX_DIMS],
            stride: [0; MAX_DIMS],
            adj_stride: [0; MAX_DIMS],
            offset: 0,
            rank: 1,
            len: 0,
        }
    }

    /// Assemble a layout from already-computed fields, without validation.
    ///
    /// An escape hatch for callers that have already worked out every field,
    /// including the `adj_stride` iteration helper. Prefer [`new`](Self::new) or
    /// [`from_strided`](Self::from_strided), which derive those for you.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// // A hand-built 2x3 contiguous layout, equivalent to `Layout::new(&[2, 3])`.
    /// let l = Layout::from_raw_parts(&[2, 3], &[3, 1], &[1, 1], 0, 6);
    /// assert_eq!(l.shape(), &[2, 3]);
    /// assert_eq!(l.len(), 6);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics (in debug builds) if `shape`, `stride`, and `adj_stride` do not all
    /// have the same length, or if that length exceeds the maximum supported rank.
    pub fn from_raw_parts(
        shape: &[usize],
        stride: &[i32],
        adj_stride: &[i32],
        offset: usize,
        len: usize,
    ) -> Self {
        let rank = shape.len();
        debug_assert!(
            rank == stride.len() && rank == adj_stride.len(),
            "shape, stride, and adj_stride must share the same rank"
        );
        debug_assert!(
            rank <= MAX_DIMS,
            "only tensors upto {} dims are supported",
            MAX_DIMS
        );

        let mut shape_arr = [0usize; MAX_DIMS];
        let mut stride_arr = [0i32; MAX_DIMS];
        let mut adj_stride_arr = [0i32; MAX_DIMS];

        shape_arr[..rank].copy_from_slice(shape);
        stride_arr[..rank].copy_from_slice(stride);
        adj_stride_arr[..rank].copy_from_slice(adj_stride);

        Self {
            shape: shape_arr,
            stride: stride_arr,
            adj_stride: adj_stride_arr,
            offset,
            rank,
            len,
        }
    }

    /// Build a layout for `shape` with an explicit `stride` and `offset`.
    ///
    /// The `adj_stride` iteration helper is derived for you. Use this to describe
    /// non-contiguous data - a stride of `0` on an axis, for instance, repeats
    /// that axis (broadcasting).
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// // Column-major 2x3: advancing a column steps 1, advancing a row steps 2.
    /// let l = Layout::from_strided(&[2, 3], &[1, 2], 0);
    /// assert_eq!(l.shape(), &[2, 3]);
    /// assert_eq!(l.stride(), &[1, 2]);
    /// ```
    pub fn from_strided(shape: &[usize], stride: &[i32], offset: usize) -> Self {
        validate_shape(shape).unwrap_or_else(|e| panic!("{}", e));
        debug_assert!(shape.len() == stride.len());

        let rank = shape.len();
        let len: usize = shape.iter().product();

        let mut shape_arr = [0usize; MAX_DIMS];
        let mut stride_arr = [0i32; MAX_DIMS];
        shape_arr[..rank].copy_from_slice(shape);
        stride_arr[..rank].copy_from_slice(stride);

        Self {
            shape: shape_arr,
            stride: stride_arr,
            adj_stride: calculate_adjacent_dim_stride(stride, shape),
            offset,
            rank,
            len,
        }
    }

    /// Return this layout with its `offset` into the backing buffer replaced.
    ///
    /// Builder-style, usually chained onto [`new`](Self::new).
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// let l = Layout::new(4).with_offset(3);
    /// assert_eq!(l.offset(), 3);
    /// ```
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;

        self
    }

    /// Derive a layout with a new `shape` but the same element count.
    ///
    /// # Errors
    ///
    /// Returns [`OpError::InvalidViewShape`] if `shape`'s element count differs
    /// from this layout's, or [`OpError::NonContiguousView`] if this layout is
    /// not contiguous (viewing needs a contiguous source).
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// let l = Layout::new(&[2, 3]);
    /// assert_eq!(l.view(&[3, 2])?.shape(), &[3, 2]);
    /// assert!(l.view(&[4, 4]).is_err()); // 16 != 6 elements
    /// # Ok::<(), candela::OpError>(())
    /// ```
    pub fn view(&self, shape: impl IntoShape) -> Result<Self, OpError> {
        let (rank, shape) = shape.into_shape();

        if shape[..rank].iter().product::<usize>() != self.len() {
            return Err(OpError::InvalidViewShape);
        }
        if !self.is_contiguous() {
            return Err(OpError::NonContiguousView);
        }
        Ok(Layout::new(&shape[..rank]).with_offset(self.offset))
    }

    /// Derive the layout of a sub-region, one [`SliceRange`] per leading axis.
    ///
    /// Axes without a range are taken in full. Build the range list with the
    /// [`s!`](crate::s) macro.
    ///
    /// # Errors
    ///
    /// Returns [`OpError::AxesOutOfBounds`] if `range` has more entries than the
    /// layout has axes, or a slice error if a range is empty or runs past its axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::{s, Layout};
    /// let l = Layout::new((2, 3));
    /// let sub = l.slice(s![0..1, 1..3])?;
    /// assert_eq!(sub.shape(), &[1, 2]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
    pub fn slice(&self, range: &[SliceRange]) -> Result<Self, OpError> {
        let info = SliceInfo::from_range(self, range)?;
        let len: usize = info.shape[..self.rank].iter().product();

        Ok(Self {
            shape: info.shape,
            stride: self.stride,
            adj_stride: info.adj_stride,
            offset: info.offset,
            rank: self.rank,
            len,
        })
    }

    /// Reverse the order of every axis (a full transpose), swapping both the
    /// shape and the stride end for end.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// let t = Layout::new((2, 3)).transpose();
    /// assert_eq!(t.shape(), &[3, 2]);
    /// assert_eq!(t.stride(), &[1, 3]);
    /// ```
    pub fn transpose(&self) -> Self {
        let rank = self.rank;
        let mut stride = self.stride;
        let mut shape = self.shape;

        for i in 0..rank / 2 {
            let last = rank - i - 1;

            stride.swap(i, last);
            shape.swap(i, last);
        }

        let adj_stride = calculate_adjacent_dim_stride(&stride[..rank], &shape[..rank]);

        Self {
            shape,
            stride,
            adj_stride,
            offset: self.offset,
            rank,
            len: self.len,
        }
    }

    /// Reorders the axes by an explicit permutation.
    ///
    /// `axes` must list every axis index exactly once: `transpose_axes(&[1, 0])`
    /// is the plain 2-D [`.transpose()`][Self::transpose], while `&[0, 2, 1]`
    /// swaps only the last two axes of a rank-3 layout and leaves the first alone.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    ///
    /// let l = Layout::new(&[1, 2, 3]);
    /// let s = l.transpose_axes(&[0, 2, 1])?;
    /// assert_eq!(s.shape(), &[1, 3, 2]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`OpError::NotEnoughAxes`] if `axes` doesn't have one entry per
    /// axis, or [`OpError::AxesOutOfBounds`] if an index is out of range or
    /// repeated (so the list isn't a valid permutation).
    pub fn transpose_axes(&self, axes: impl IntoShape) -> Result<Self, OpError> {
        let (axes_rank, axes) = axes.into_shape();
        let rank = self.rank;

        if axes_rank != rank {
            return Err(OpError::NotEnoughAxes(rank, axes_rank));
        }

        for (i, axis) in axes[..axes_rank].iter().enumerate() {
            for axis_other in axes[..axes_rank].iter().skip(i + 1) {
                if axis == axis_other {
                    return Err(OpError::AxesOutOfBounds);
                }
            }
        }

        let mut stride = [0i32; MAX_DIMS];
        let mut shape = [0usize; MAX_DIMS];

        for (new_axis, &axis) in axes[..axes_rank].iter().enumerate() {
            if axis >= rank {
                return Err(OpError::AxesOutOfBounds);
            }

            stride[new_axis] = self.stride[axis];
            shape[new_axis] = self.shape[axis];
        }

        let adj_stride = calculate_adjacent_dim_stride(&stride[..rank], &shape[..rank]);

        Ok(Self {
            shape,
            stride,
            adj_stride,
            offset: self.offset,
            rank,
            len: self.len,
        })
    }

    /// Expands the layout to a larger shape along new or size-1 axes.
    ///
    /// Broadcasting follows NumPy's right-aligned rules: a target axis must
    /// either match the source or expand from size 1, and extra leading axes are
    /// added on the left. The repeated axes are faked with zero strides.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    ///
    /// let row = Layout::new((1, 3));
    /// let b = row.broadcast((2, 3))?;
    /// assert_eq!(b.shape(), &[2, 3]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`OpError::CannotBroadcast`] if the target shape has fewer axes
    /// than the source, or an axis is neither equal to the source nor expandable
    /// from 1.
    pub fn broadcast(&self, shape: impl IntoShape) -> Result<Self, OpError> {
        let (rank, shape) = shape.into_shape();
        let src_rank = self.rank;

        if rank < src_rank {
            return Err(OpError::CannotBroadcast);
        }

        for (s1, s2) in zip(
            shape[..rank].iter().rev(),
            self.shape[..src_rank].iter().rev(),
        ) {
            if *s2 != 1 && *s1 != *s2 {
                return Err(OpError::CannotBroadcast);
            }
        }

        // Right-align the source strides under the target shape: the extra leading
        // axes keep their zero stride, the rest copy the source stride.
        let mut new_stride = [0i32; MAX_DIMS];
        let lead = rank - src_rank;
        new_stride[lead..rank].copy_from_slice(&self.stride[..src_rank]);

        for (dim, s) in self.shape[..src_rank].iter().rev().enumerate() {
            if *s == 1 {
                new_stride[rank - dim - 1] = 0;
            }
        }

        let adj_stride = calculate_adjacent_dim_stride(&new_stride[..rank], &shape[..rank]);
        let len: usize = shape[..rank].iter().product();

        Ok(Self {
            shape,
            stride: new_stride,
            adj_stride,
            offset: self.offset,
            rank,
            len,
        })
    }

    /// Collapses the shape into a canonical `[batch, rows, cols]` triple.
    ///
    /// The last two axes become `rows` and `cols`; everything above them is
    /// folded into a single `batch` count, and ranks below 3 are padded with
    /// leading ones. The matmul kernel uses this to treat any rank uniformly.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// assert_eq!(Layout::new(5).shape_as_3d(), [1, 1, 5]);
    /// assert_eq!(Layout::new((2, 3)).shape_as_3d(), [1, 2, 3]);
    /// assert_eq!(Layout::new((2, 3, 4)).shape_as_3d(), [2, 3, 4]);
    /// assert_eq!(Layout::new((6, 2, 3, 4)).shape_as_3d(), [12, 3, 4]);
    /// ```
    #[inline]
    pub fn shape_as_3d(&self) -> [usize; 3] {
        debug_assert!(self.rank >= 1, "shape_as_3d requires rank >= 1");
        if self.rank == 1 {
            [1, 1, self.shape[0]]
        } else if self.rank == 2 {
            [1, self.shape[0], self.shape[1]]
        } else {
            let len = self.rank;

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
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// let r = Layout::new((2, 3, 4)).rotate_axis_innermost(0)?;
    /// assert_eq!(r.shape(), &[3, 4, 2]);
    /// # Ok::<(), candela::OpError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`OpError::AxesOutOfBounds`] if `axis` is past the layout's rank.
    #[inline]
    pub fn rotate_axis_innermost(&self, axis: usize) -> Result<Self, OpError> {
        if axis >= self.shape().len() {
            return Err(OpError::AxesOutOfBounds);
        }

        let mut axes: Vec<usize> = (axis + 1..self.shape().len()).collect();
        axes.extend(0..=axis);

        unsafe { Ok(self.transpose_axes(&axes).unwrap_unchecked()) }
    }

    /// Returns `true` if a flat walk visits every element in row-major order with
    /// no gaps - i.e. the layout has not been transposed, broadcast, or sliced
    /// down the inner axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// assert!(Layout::new((3, 4)).is_contiguous());
    /// assert!(!Layout::new((3, 4)).transpose().is_contiguous());
    /// ```
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.is_contiguous_at_axis(0)
    }

    /// Like [`is_contiguous`](Self::is_contiguous), but only checks the axes from
    /// `axis` inward, ignoring how the outer axes are arranged.
    ///
    /// An out-of-range `axis` returns `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// let l = Layout::new((2, 3));
    /// assert!(l.is_contiguous_at_axis(0));
    /// assert!(!l.is_contiguous_at_axis(9)); // out of range
    /// ```
    #[inline]
    pub fn is_contiguous_at_axis(&self, axis: usize) -> bool {
        if axis >= self.rank {
            return false;
        }

        self.adj_stride[axis] == 1 && !self.stride[axis + 1..self.rank].contains(&0)
    }

    /// Returns `true` if any axis runs backwards relative to a contiguous layout -
    /// the signature a transpose leaves behind. Broadcast axes (zero stride) are
    /// not counted.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// assert!(!Layout::new((3, 4)).is_transposed());
    /// assert!(Layout::new((3, 4)).transpose().is_transposed());
    /// ```
    #[inline]
    pub fn is_transposed(&self) -> bool {
        for (i, &adj_stride) in self.adj_stride[..self.rank].iter().enumerate() {
            if adj_stride < 0 && self.stride[i] != 0 {
                return true;
            }
        }

        false
    }

    /// Like [`is_transposed`](Self::is_transposed), but tests a single `axis`.
    ///
    /// An out-of-range `axis` returns `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// let l = Layout::new((3, 4)); // fresh: no axis is transposed
    /// assert!(!l.is_transposed_at_axis(0));
    /// assert!(!l.is_transposed_at_axis(9)); // out of range
    /// ```
    #[inline]
    pub fn is_transposed_at_axis(&self, axis: usize) -> bool {
        if axis >= self.rank {
            return false;
        }

        self.adj_stride[axis] < 0 && self.stride[axis] != 0
    }

    /// Returns `true` for a 2-D layout whose two axes are transposed; always
    /// `false` for any other rank.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// assert!(Layout::new((3, 4)).transpose().is_last_axes_transposed());
    /// assert!(!Layout::new((3, 4)).is_last_axes_transposed());
    /// ```
    // Restricted to 2D on purpose: the matmul kernel uses this to pick the BLAS
    // trans-flag, and its batch-stride handling assumes there is no batch dim.
    // Higher-rank tensors whose last two strides happen to match this pattern
    // would silently feed an incoherent batch stride to GEMM.
    #[inline]
    pub fn is_last_axes_transposed(&self) -> bool {
        if self.rank != 2 {
            return false;
        }

        let rs = self.stride[self.rank - 2];
        let cs = self.stride[self.rank - 1];

        // Gives false on broadcasting
        if rs == 0 || cs == 0 {
            return false;
        }

        // cs must be > 1: a contiguous [m, 1] matrix has rs=cs=1 and is not transposed
        rs == 1 && cs > 1
    }

    /// The size of each axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// assert_eq!(Layout::new((2, 3)).shape(), &[2, 3]);
    /// ```
    #[inline]
    pub fn shape(&self) -> &'_ [usize] {
        &self.shape[..self.rank]
    }

    /// The per-axis stride: how many buffer elements to step to advance one
    /// index along that axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// assert_eq!(Layout::new((2, 3)).stride(), &[3, 1]);
    /// ```
    #[inline]
    pub fn stride(&self) -> &'_ [i32] {
        &self.stride[..self.rank]
    }

    /// The adjacent stride: the per-axis step a flat iterator applies when it
    /// rolls over into the next axis. See the [layout docs](crate::docs::layout).
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// // All ones for a freshly built contiguous layout.
    /// assert_eq!(Layout::new((2, 3)).adj_stride(), &[1, 1]);
    /// ```
    #[inline]
    pub fn adj_stride(&self) -> &'_ [i32] {
        &self.adj_stride[..self.rank]
    }

    /// The starting index into the backing buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// assert_eq!(Layout::new(4).with_offset(3).offset(), 3);
    /// ```
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The total number of elements, i.e. the product of the shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// assert_eq!(Layout::new((2, 3, 4)).len(), 24);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the layout has no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::Layout;
    /// assert!(Layout::empty().is_empty());
    /// assert!(!Layout::new(3).is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the memory position of the last element from offset.
    ///
    /// This is akin to (offset + (shape - 1) * stride).
    #[inline]
    pub fn last(&self) -> usize {
        let mut acc: usize = 0;

        for (d, s) in zip(&self.shape[..self.rank], &self.stride[..self.rank]) {
            acc = acc.wrapping_add_signed((*d - 1) as isize * *s as isize);
        }

        self.offset + acc
    }
}

impl PartialEq for Layout {
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape() && self.stride() == other.stride()
    }
}

impl Eq for Layout {}

impl Hash for Layout {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.shape().hash(state);
        self.stride().hash(state);
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
            self.shape(),
            self.stride(),
            self.offset
        )
    }
}
