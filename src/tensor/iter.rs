use std::iter::FusedIterator;

use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::traits::StreamingIterator;

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

const MAX_DIMS: usize = 8;

#[derive(Debug)]
pub struct ChunkIter<'a, T> {
    data: &'a [T],
    pos: usize,
    shape: [usize; MAX_DIMS],
    adj_stride: [i32; MAX_DIMS],
    counter: [usize; MAX_DIMS],
    rank: usize,
    step: isize,
    left_over: usize,
}

#[derive(Debug)]
pub enum ChunkKind<'a, T> {
    Contiguous {
        data: &'a [T],
        start: usize,
        times: usize,
    },
    Strided {
        data: &'a [T],
        start: usize,
        times: usize,
        step: isize,
    },
}

fn simplify_layout(layout: &Layout) -> (usize, [usize; MAX_DIMS], [i32; MAX_DIMS]) {
    let rank: usize = layout.shape().len();
    let mut shape = [0usize; MAX_DIMS];
    let mut adj_stride = [0i32; MAX_DIMS];

    let mut w: usize = 0;

    shape[0] = layout.shape()[0];
    adj_stride[0] = layout.adj_stride()[0];
    for i in 1..rank {
        if layout.adj_stride()[i] == layout.adj_stride()[i - 1] {
            shape[w] *= layout.shape()[i];
        } else {
            w += 1;
            shape[w] = layout.shape()[i];
            adj_stride[w] = layout.adj_stride()[i];
        }
    }

    (w + 1, shape, adj_stride)
}

impl<'a, T> ChunkIter<'a, T> {
    pub fn new(data: &'a [T], layout: &Layout) -> Self {
        debug_assert!(
            layout.shape().len() <= MAX_DIMS,
            "dimensions higher than 8 are not support for iteration!"
        );

        let (mut rank, mut shape, mut adj_stride) = simplify_layout(layout);

        if rank == 1 {
            shape[1] = shape[0];
            shape[0] = 1;

            adj_stride[1] = adj_stride[0];

            rank += 1;
        }

        let last = rank - 1;
        let step = adj_stride[last] as isize * (shape[last] - 1) as isize;

        let left_over: usize = shape[0..last].iter().product();

        Self {
            data,
            pos: layout.offset(),
            shape,
            adj_stride,
            counter: [0; MAX_DIMS],
            step,
            rank,
            left_over,
        }
    }
}

impl<'a, T> Iterator for ChunkIter<'a, T> {
    type Item = ChunkKind<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.left_over == 0 {
            return None;
        }

        let last = self.rank - 1;

        // TODO: self.adj_stride[last] is a constant, we can store that.
        let chunk = if self.adj_stride[last] == 1 {
            ChunkKind::Contiguous {
                data: self.data,
                start: self.pos as usize,
                times: self.shape[last],
            }
        } else {
            ChunkKind::Strided {
                data: self.data,
                start: self.pos as usize,
                times: self.shape[last],
                step: self.adj_stride[last] as isize,
            }
        };

        self.left_over -= 1;

        let last_counter = last - 1;

        self.counter[last_counter] += 1;
        let mut step_dim = last_counter;
        for dim in (1..last).rev() {
            if self.counter[dim] == self.shape[dim] {
                self.counter[dim] = 0;
                self.counter[dim - 1] += 1;

                step_dim = dim - 1;
                continue;
            }
            break;
        }

        // TODO: We can change the adj_stride so it has self.step already summed on it
        self.pos = self
            .pos
            .wrapping_add_signed(self.adj_stride[step_dim] as isize + self.step);

        Some(chunk)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left_over, Some(self.left_over))
    }
}

impl<'a, T> ExactSizeIterator for ChunkIter<'a, T> {}

impl<'a, T> FusedIterator for ChunkIter<'a, T> {}

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

        let pos = self.pos as usize;

        let item = &self.data[pos];

        self.pos = self
            .pos
            .wrapping_add_signed(self.layout.adj_stride()[step_dim] as isize);

        self.left_over -= 1;

        Some(&item)
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
        let mut acc = init;

        if self.layout.is_contiguous() {
            for el in
                self.data[self.layout.offset()..self.layout.offset() + self.layout.len()].iter()
            {
                acc = f(acc, el);
            }

            return acc;
        }

        let (mut rank, mut shape, mut adj_stride) = simplify_layout(self.layout);

        if rank == 1 {
            shape[1] = shape[0];
            shape[0] = 1;

            adj_stride[1] = adj_stride[0];

            rank += 1;
        }

        let last = rank - 1;
        let mut counter: [usize; MAX_DIMS] = [0; MAX_DIMS];
        let mut pos = self.layout.offset();
        let left_over: usize = shape[0..last].iter().product();

        let n = shape[last];
        let step = adj_stride[last] as isize * (shape[last] - 1) as isize;
        let stride = adj_stride[last] as isize;

        let next_chunk = |counter: &mut [usize; MAX_DIMS]| -> isize {
            let last_counter = last - 1;
            counter[last_counter] += 1;
            let mut step_dim = last_counter;
            for dim in (1..last).rev() {
                if counter[dim] == shape[dim] {
                    counter[dim] = 0;
                    counter[dim - 1] += 1;
                    step_dim = dim - 1;
                    continue;
                }
                break;
            }
            adj_stride[step_dim] as isize + step
        };

        if stride == 1 {
            for _ in 0..left_over {
                for el in self.data[pos..pos + n].iter() {
                    acc = f(acc, el);
                }

                pos = pos.wrapping_add_signed(next_chunk(&mut counter));
            }
        } else {
            for _ in 0..left_over {
                let mut pos_inner = pos;
                for _ in 0..n {
                    debug_assert!(pos_inner < self.data.len());
                    // SAFETY: a well-formed layout only ever visits in-bounds
                    // positions of its own buffer, so `pos_inner` is a valid
                    // index. Dropping the bounds check keeps the strided read
                    // from stalling memory-level parallelism on gather-heavy
                    // layouts (e.g. transposed).
                    acc = f(acc, unsafe { self.data.get_unchecked(pos_inner) });

                    pos_inner = pos_inner.wrapping_add_signed(stride);
                }
                pos = pos.wrapping_add_signed(next_chunk(&mut counter));
            }
        }

        acc
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

impl<'a, T> FusedIterator for Iter<'a, T> {}

///////////////////////////////////////////////////////////////

pub struct MutSliceIter<'a, T> {
    data: &'a mut Vec<T>,
    pos: isize,
    counter: Box<[usize]>,
    layout: &'a Layout,
    left_over: usize,
}

impl<'a, T: Clone> MutSliceIter<'a, T> {
    pub fn new(data: &'a mut Vec<T>, data_len: usize, layout: &'a Layout) -> Self {
        let counter = vec![0; layout.shape().len()].into_boxed_slice();

        Self {
            data,
            pos: layout.offset() as isize,
            layout,
            counter,
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

        let last = self.counter.len() - 1;
        self.counter[last] += 1;
        let mut step_dim = last;

        for dim in (1..self.counter.len()).rev() {
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
            self.pos += self.layout.adj_stride()[step_dim] as isize;
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
    counter: Vec<usize>,
}

impl<'a, T: Clone> InformedIter<'a, T> {
    pub fn new(data: &'a [T], layout: &'a Layout) -> Self {
        let len = layout.shape().len();

        Self {
            buffer: data,
            layout,
            next_state: StepInfo::<T>::EnterDimension(0),
            pos: layout.offset() as i64,
            counter: vec![0; len],
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

                self.pos += self.layout.adj_stride()[dim - 1] as i64;
                self.next_state = StepInfo::EnterDimension(dim);

                Some(StepInfo::ExitDimension(dim))
            }
            StepInfo::Value(v) => {
                let counter_last = self.counter.len() - 1;

                if *self.counter.last().unwrap() == *self.layout.shape().last().unwrap() - 1 {
                    self.next_state = StepInfo::ExitDimension(self.counter.len() - 1);
                    self.counter[counter_last] = 0;

                    return Some(StepInfo::Value(v));
                }

                self.pos += *self.layout.adj_stride().last().unwrap() as i64;
                self.counter[counter_last] += 1;

                self.next_state = StepInfo::Value(self.buffer[self.pos as usize]);

                Some(StepInfo::Value(v))
            }
            StepInfo::End => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.layout.len() - self.pos as usize;

        (len, Some(len))
    }
}

impl<'a, T: Copy> ExactSizeIterator for InformedIter<'a, T> {}

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
