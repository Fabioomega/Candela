use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use crate::tensor::mem_formats::layout::Layout;

use crate::cfg_debug_only;
use crate::tensor::errors::OpError;
use crate::tensor::internals::calculate_adjacent_dim_stride;

enum SliceBounds {
    Beginning,
    Index(usize),
    ReverseIndex(usize),
    End,
}

pub struct SliceRange {
    start: SliceBounds,
    end: SliceBounds,
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
    pub(crate) shape: Box<[i32]>,
    pub(crate) adj_stride: Box<[i32]>,
}

impl SliceInfo {
    pub(crate) fn from_range(layout: &Layout, range: &[SliceRange]) -> Result<Self, OpError> {
        debug_assert!(layout.shape().len() >= range.len());

        let mut offset: i64 = layout.offset() as i64;
        let mut new_shape: Vec<i32> = layout.shape().into();

        for (dim, r) in range.iter().enumerate() {
            let start = match r.start {
                SliceBounds::Beginning => 0,
                SliceBounds::Index(i) => {
                    offset += (i as i64) * layout.stride()[dim] as i64;

                    i as i32
                }
                SliceBounds::ReverseIndex(i) => {
                    let true_index = layout.shape()[dim] as i64 - i as i64;
                    offset += true_index * layout.stride()[dim] as i64;

                    true_index as i32
                }
                _ => unreachable!("a new variation of SliceBounds was implemented"),
            };

            let end = match r.end {
                SliceBounds::End => layout.shape()[dim],
                SliceBounds::Index(i) => i as i32,
                SliceBounds::ReverseIndex(i) => {
                    let true_index = layout.shape()[dim] as i64 - i as i64;
                    true_index as i32
                }
                _ => unreachable!("a new variation of SliceBounds was implemented"),
            };

            cfg_debug_only!(if end <= start {
                return Err(OpError::OutOfBoundSlice);
            });

            new_shape[dim] = end - start;
        }

        cfg_debug_only!({
            let len: i32 = new_shape.iter().product();
            let len = len as usize;

            if len + (offset as usize) > layout.len() {
                return Err(OpError::InvalidSliceShape(layout.len(), len));
            }
        });

        let adj_stride = calculate_adjacent_dim_stride(layout.stride(), &new_shape);

        Ok(Self {
            offset: offset as usize,
            shape: new_shape.into_boxed_slice(),
            adj_stride,
        })
    }
}
