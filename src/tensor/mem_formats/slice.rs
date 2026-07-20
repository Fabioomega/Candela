use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use crate::tensor::MAX_DIMS;
use crate::tensor::mem_formats::layout::Layout;

use crate::tensor::errors::OpError;
use crate::tensor::internals::calculate_adjacent_dim_stride;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SliceBounds {
    Beginning,
    Index(usize),
    ReverseIndex(usize),
    End,
}

/// A per-axis range for [`slice`](crate::Tensor::slice), one entry per axis.
///
/// You rarely name `SliceRange` directly - the [`s!`](crate::s) macro builds the
/// list from ordinary range syntax. It converts from `a..b`, `a..`, `..b`, `..`,
/// and bare integers (a single index); negative bounds count from the end.
///
/// # Examples
///
/// ```
/// use candela::{s, Dimension, Tensor};
///
/// let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
/// // row 0, columns 1..3
/// let sub = t.slice(s![0..1, 1..3])?.materialize();
/// assert_eq!(sub.shape(), &[1, 2]);
/// let vals: Vec<f64> = sub.iter().copied().collect();
/// assert_eq!(vals, [1.0, 2.0]);
/// # Ok::<(), candela::OpError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SliceRange {
    start: SliceBounds,
    end: SliceBounds,
}

impl From<i32> for SliceRange {
    #[inline]
    fn from(value: i32) -> Self {
        if value >= 0 {
            Self {
                start: SliceBounds::Index(value as usize),
                end: SliceBounds::Index((value + 1) as usize),
            }
        } else {
            Self {
                start: SliceBounds::ReverseIndex((-value) as usize),
                end: SliceBounds::ReverseIndex((-(value + 1)) as usize),
            }
        }
    }
}

impl From<RangeFrom<i32>> for SliceRange {
    #[inline]
    fn from(value: RangeFrom<i32>) -> Self {
        if value.start >= 0 {
            Self {
                start: SliceBounds::Index(value.start as usize),
                end: SliceBounds::End,
            }
        } else {
            Self {
                start: SliceBounds::ReverseIndex((-value.start) as usize),
                end: SliceBounds::End,
            }
        }
    }
}

impl From<RangeTo<i32>> for SliceRange {
    #[inline]
    fn from(value: RangeTo<i32>) -> Self {
        if value.end >= 0 {
            Self {
                start: SliceBounds::Beginning,
                end: SliceBounds::Index(value.end as usize),
            }
        } else {
            Self {
                start: SliceBounds::Beginning,
                end: SliceBounds::ReverseIndex((-value.end) as usize),
            }
        }
    }
}

impl From<RangeFull> for SliceRange {
    #[inline]
    fn from(_: RangeFull) -> Self {
        Self {
            start: SliceBounds::Beginning,
            end: SliceBounds::End,
        }
    }
}

impl From<Range<i32>> for SliceRange {
    #[inline]
    fn from(value: Range<i32>) -> Self {
        let start = if value.start >= 0 {
            SliceBounds::Index(value.start as usize)
        } else {
            SliceBounds::ReverseIndex((-value.start) as usize)
        };

        let end = if value.end >= 0 {
            SliceBounds::Index(value.end as usize)
        } else {
            SliceBounds::ReverseIndex((-value.end) as usize)
        };

        Self { start, end }
    }
}

/////////////////////////////////////////////////////

#[derive(Debug)]
pub struct SliceInfo {
    pub(crate) offset: usize,
    pub(crate) shape: [usize; MAX_DIMS],
    pub(crate) adj_stride: [i32; MAX_DIMS],
}

impl SliceInfo {
    pub(crate) fn from_range(layout: &Layout, range: &[SliceRange]) -> Result<Self, OpError> {
        let rank = layout.shape().len();

        if range.len() > rank {
            return Err(OpError::AxesOutOfBounds);
        }

        let mut offset: i64 = layout.offset() as i64;
        let mut new_shape = [0usize; MAX_DIMS];
        new_shape[..rank].copy_from_slice(layout.shape());

        for (dim, r) in range.iter().enumerate() {
            let start = match r.start {
                SliceBounds::Beginning => 0,
                SliceBounds::Index(i) => {
                    offset += (i as i64) * layout.stride()[dim] as i64;

                    i
                }
                SliceBounds::ReverseIndex(i) => {
                    let true_index = layout.shape()[dim] - i;
                    offset += true_index as i64 * layout.stride()[dim] as i64;

                    true_index
                }
                _ => unreachable!("a new variation of SliceBounds was implemented"),
            };

            let end = match r.end {
                SliceBounds::End => layout.shape()[dim],
                SliceBounds::Index(i) => i,
                SliceBounds::ReverseIndex(i) => layout.shape()[dim] - i,
                _ => unreachable!("a new variation of SliceBounds was implemented"),
            };

            if end <= start {
                return Err(OpError::SliceOutOfBounds);
            }

            new_shape[dim] = end - start;
        }

        let len: usize = new_shape[..rank].iter().product();
        if len + (offset as usize) > layout.len() {
            return Err(OpError::InvalidSliceShape(layout.len(), len));
        }

        let adj_stride = calculate_adjacent_dim_stride(layout.stride(), &new_shape[..rank]);

        Ok(Self {
            offset: offset as usize,
            shape: new_shape,
            adj_stride,
        })
    }
}
