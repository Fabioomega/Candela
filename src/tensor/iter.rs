use std::iter::{FusedIterator, zip};

use crate::tensor::MAX_DIMS;
use crate::tensor::internals::calculate_adjacent_dim_stride;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::traits::StreamingIterator;
use crate::tensor::walker::fold_ref;

pub struct ContiguousIter<'a, T: Clone> {
    data: &'a [T],
    offset: usize,
    left_over: usize,
}

impl<'a, T: Clone> ContiguousIter<'a, T> {
    pub fn new(data: &'a [T], offset: usize, len: usize) -> Self {
        Self {
            data,
            offset,
            left_over: len,
        }
    }
}

impl<'a, T: Clone> Iterator for ContiguousIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.left_over == 0 {
            return None;
        }

        let item = &self.data[self.offset] as *const T;
        self.offset += 1;
        self.left_over -= 1;

        Some(unsafe { &*item })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left_over, Some(self.left_over))
    }
}

impl<'a, T: Clone> ExactSizeIterator for ContiguousIter<'a, T> {}

impl<'a, T: Clone> FusedIterator for ContiguousIter<'a, T> {}

///////////////////////////////////////////////////////////////

/// Iterator over a tensor's elements in logical (row-major) order.
///
/// Returned by [`Tensor::iter`](crate::Tensor::iter). It walks the backing
/// buffer following the tensor's [`Layout`], so a sliced or transposed tensor
/// yields its elements in the order its shape implies rather than in storage
/// order.
#[derive(Debug, Clone)]
pub struct Iter<'a, T> {
    data: &'a [T],
    pos: usize,
    counter: [usize; MAX_DIMS],
    adj_stride: [i32; MAX_DIMS],
    layout: &'a Layout,
    left_over: usize,
}

impl<'a, T> Iter<'a, T> {
    pub fn new(data: &'a [T], layout: &'a Layout) -> Self {
        debug_assert!(
            layout.shape().len() <= MAX_DIMS,
            "dimensions higher than 8 are not support for iteration!"
        );

        Self {
            data,
            pos: layout.offset(),
            layout,
            counter: [0; MAX_DIMS],
            adj_stride: calculate_adjacent_dim_stride(layout.stride(), layout.shape()),
            left_over: layout.len(),
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.left_over == 0 {
            return None;
        }

        let last = self.layout.shape().len() - 1;
        self.counter[last] += 1;
        let mut step_dim = last;

        for dim in (1..self.layout.shape().len()).rev() {
            if self.counter[dim] == self.layout.shape()[dim] {
                self.counter[dim] = 0;
                self.counter[dim - 1] += 1;

                step_dim = dim - 1;
                continue;
            }
            break;
        }

        let pos = self.pos;

        let item = &self.data[pos];

        self.pos = self
            .pos
            .wrapping_add_signed(self.adj_stride[step_dim] as isize);

        self.left_over -= 1;

        Some(item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left_over, Some(self.left_over))
    }

    #[inline]
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self::Item) -> B,
    {
        fold_ref(self.data, self.layout, init, &mut f)
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

impl<'a, T> FusedIterator for Iter<'a, T> {}

///////////////////////////////////////////////////////////////

pub struct MutSliceIter<'a, T> {
    data: &'a mut [T],
    pos: isize,
    counter: [usize; MAX_DIMS],
    adj_stride: [i32; MAX_DIMS],
    layout: &'a Layout,
    left_over: usize,
}

impl<'a, T: Clone> MutSliceIter<'a, T> {
    pub fn new(data: &'a mut Vec<T>, data_len: usize, layout: &'a Layout) -> Self {
        debug_assert!(
            layout.shape().len() <= MAX_DIMS,
            "dimensions higher than 8 are not support for iteration!"
        );

        Self {
            data,
            pos: layout.offset() as isize,
            layout,
            counter: [0; MAX_DIMS],
            adj_stride: calculate_adjacent_dim_stride(layout.stride(), layout.shape()),
            left_over: data_len,
        }
    }
}

impl<'a, T: Clone> Iterator for MutSliceIter<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.left_over == 0 {
            return None;
        }

        let last = self.layout.shape().len() - 1;
        self.counter[last] += 1;
        let mut step_dim = last;

        for dim in (1..self.layout.shape().len()).rev() {
            if self.counter[dim] == self.layout.shape()[dim] {
                self.counter[dim] = 0;
                self.counter[dim - 1] += 1;

                step_dim = dim - 1;
                continue;
            }
            break;
        }

        let pos = self.pos as usize;

        unsafe {
            let item = &mut self.data[pos] as *mut T;
            self.pos += self.adj_stride[step_dim] as isize;
            self.left_over -= 1;

            Some(&mut *item)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left_over, Some(self.left_over))
    }
}

impl<'a, T: Clone> ExactSizeIterator for MutSliceIter<'a, T> {}

impl<'a, T: Clone> FusedIterator for MutSliceIter<'a, T> {}

///////////////////////////////////////////////////////////////

/// A single event in a structural walk of a tensor, produced by
/// [`Tensor::informed_iter`](crate::Tensor::informed_iter).
///
/// A walk is the nested loop it would take to visit every element - one loop per
/// dimension, the innermost yielding values. Each variant marks a point in that
/// loop: [`EnterDimension`] when a loop opens, [`Value`] for an element the
/// innermost loop reads, [`ExitDimension`] when a loop closes, and [`End`] once
/// every loop has finished.
///
/// The walk follows the tensor's logical layout, so a sliced or transposed
/// tensor is visited in the order its shape implies.
///
/// [`EnterDimension`]: StepInfo::EnterDimension
/// [`ExitDimension`]: StepInfo::ExitDimension
/// [`Value`]: StepInfo::Value
/// [`End`]: StepInfo::End
///
/// # Examples
///
/// ```
/// use candela::{StepInfo, Tensor};
///
/// let t = Tensor::from_slice(&[1.0, 2.0], &[2]);
/// let events: Vec<StepInfo<f64>> = t.informed_iter().collect();
/// assert_eq!(events, vec![
///     StepInfo::EnterDimension(0),
///     StepInfo::Value(1.0),
///     StepInfo::Value(2.0),
///     StepInfo::ExitDimension(0),
/// ]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepInfo<T: Clone> {
    /// A dimension's loop has opened; the payload is its index, `0` being the
    /// outermost.
    EnterDimension(usize),
    /// A dimension's loop has closed; the payload is its index.
    ExitDimension(usize),
    /// An element, read by the innermost loop in logical order.
    Value(T),
    /// Every loop has finished. Iteration terminates by returning `None`, so the
    /// iterator does not yield this variant.
    End,
}

/// Structural walk over a tensor, yielding a [`StepInfo`] per element and per
/// dimension boundary.
///
/// Returned by [`Tensor::informed_iter`](crate::Tensor::informed_iter). Unlike
/// [`Iter`], which yields a flat stream of values, its events also mark where
/// each sub-array opens and closes, which is what lets you reconstruct the
/// tensor's nesting. See [`StepInfo`] for the event kinds and
/// [`Tensor::informed_iter`](crate::Tensor::informed_iter) for a worked example.
#[derive(Debug, Clone)]
pub struct InformedIter<'a, T: Clone> {
    buffer: &'a [T],
    layout: &'a Layout,
    next_state: StepInfo<T>,
    pos: i64,
    counter: [usize; MAX_DIMS],
    adj_stride: [i32; MAX_DIMS],
}

impl<'a, T: Clone> InformedIter<'a, T> {
    pub fn new(data: &'a [T], layout: &'a Layout) -> Self {
        debug_assert!(
            layout.shape().len() <= MAX_DIMS,
            "dimensions higher than 8 are not support for iteration!"
        );

        Self {
            buffer: data,
            layout,
            next_state: StepInfo::<T>::EnterDimension(0),
            pos: layout.offset() as i64,
            counter: [0; MAX_DIMS],
            adj_stride: calculate_adjacent_dim_stride(layout.stride(), layout.shape()),
        }
    }
}

impl<'a, T: Copy> Iterator for InformedIter<'a, T> {
    type Item = StepInfo<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_state {
            StepInfo::EnterDimension(dim) => {
                if dim == self.layout.shape().len() - 1 {
                    self.next_state = StepInfo::Value(self.buffer[self.pos as usize]);

                    return Some(StepInfo::EnterDimension(dim));
                }

                self.next_state = StepInfo::EnterDimension(dim + 1);

                Some(StepInfo::EnterDimension(dim))
            }
            StepInfo::ExitDimension(dim) => {
                if dim == 0 {
                    self.next_state = StepInfo::End;
                    return Some(StepInfo::ExitDimension(dim));
                }

                self.counter[dim] = 0;
                self.counter[dim - 1] += 1;

                if self.counter[dim - 1] == self.layout.shape()[dim - 1] {
                    self.next_state = StepInfo::ExitDimension(dim - 1);
                    return Some(StepInfo::ExitDimension(dim));
                }

                self.pos += self.adj_stride[dim - 1] as i64;
                self.next_state = StepInfo::EnterDimension(dim);

                Some(StepInfo::ExitDimension(dim))
            }
            StepInfo::Value(v) => {
                let counter_last = self.layout.shape().len() - 1;

                if self.counter[counter_last] == *self.layout.shape().last().unwrap() - 1 {
                    self.next_state = StepInfo::ExitDimension(counter_last);
                    self.counter[counter_last] = 0;

                    return Some(StepInfo::Value(v));
                }

                self.pos += self.adj_stride[counter_last] as i64;
                self.counter[counter_last] += 1;

                self.next_state = StepInfo::Value(self.buffer[self.pos as usize]);

                Some(StepInfo::Value(v))
            }
            StepInfo::End => None,
        }
    }
}

impl<'a, T: Copy> FusedIterator for InformedIter<'a, T> {}

/////////////////////////////////////////////////////////////
pub struct PackedBuffer<'a, T: Clone> {
    pub packing_buffer: &'a [T],
    pub absolute_buffer_position: usize,
}

pub struct ChunkedSliceIter<I, T: Clone>
where
    I: IntoIterator<Item = T>,
{
    iter: I::IntoIter,
    packing_buffer: Vec<T>,
    absolute_buffer_position: usize,
}

impl<I, T: Clone + Default> ChunkedSliceIter<I, T>
where
    I: Iterator<Item = T>,
{
    pub fn new(iter: I, packing_buffer_size: usize) -> Self {
        Self {
            iter,
            packing_buffer: vec![T::default(); packing_buffer_size],
            absolute_buffer_position: 0,
        }
    }
}

impl<I, T: Clone> StreamingIterator for ChunkedSliceIter<I, T>
where
    I: IntoIterator<Item = T>,
{
    type Item<'a>
        = PackedBuffer<'a, T>
    where
        Self: 'a;

    fn next_stream<'a>(&'a mut self) -> Option<Self::Item<'a>> {
        let mut len = 0;

        for slot in &mut self.packing_buffer {
            match self.iter.next() {
                Some(v) => {
                    *slot = v;
                    len += 1;
                }
                None => break,
            }
        }

        if len == 0 {
            return None;
        }

        let pos = self.absolute_buffer_position;
        self.absolute_buffer_position += len;

        Some(PackedBuffer {
            packing_buffer: &self.packing_buffer[..len],
            absolute_buffer_position: pos,
        })
    }
}

/////////////////////////////////////////////////////////////
pub struct ChunkedContiguousIter<'a, T: Clone> {
    data: &'a [T],
    packing_buffer_size: usize,
    absolute_buffer_position: usize,
}

impl<'a, T: Clone> ChunkedContiguousIter<'a, T> {
    pub fn new(data: &'a [T], packing_buffer_size: usize) -> Self {
        Self {
            data,
            packing_buffer_size,
            absolute_buffer_position: 0,
        }
    }
}

impl<'b, T: Clone> StreamingIterator for ChunkedContiguousIter<'b, T> {
    type Item<'a>
        = PackedBuffer<'a, T>
    where
        Self: 'a;

    fn next_stream<'a>(&'a mut self) -> Option<Self::Item<'a>> {
        if self.absolute_buffer_position >= self.data.len() {
            return None;
        }

        let start = self.absolute_buffer_position;
        let end = (self.absolute_buffer_position + self.packing_buffer_size).min(self.data.len());
        self.absolute_buffer_position = end;

        Some(PackedBuffer {
            packing_buffer: &self.data[start..end],
            absolute_buffer_position: start,
        })
    }
}

impl<'a, T: Clone> Iterator for ChunkedContiguousIter<'a, T> {
    type Item = PackedBuffer<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.absolute_buffer_position >= self.data.len() {
            return None;
        }

        let start = self.absolute_buffer_position;
        let end = (self.absolute_buffer_position + self.packing_buffer_size).min(self.data.len());
        self.absolute_buffer_position = end;

        Some(PackedBuffer {
            packing_buffer: &self.data[start..end],
            absolute_buffer_position: start,
        })
    }
}

impl<'a, T: Clone> FusedIterator for ChunkedContiguousIter<'a, T> {}
