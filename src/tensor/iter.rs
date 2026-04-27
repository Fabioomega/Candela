use std::iter::FusedIterator;
use std::sync::Arc;

use crate::debug_assert_positive;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::traits::StreamingIterator;

pub struct ContiguousIter<'a, T: Copy> {
    data: &'a Arc<Vec<T>>,
    offset: usize,
    left_over: usize,
}

impl<'a, T: Copy> ContiguousIter<'a, T> {
    pub fn new(data: &'a Arc<Vec<T>>, offset: usize, len: usize) -> Self {
        Self {
            data,
            offset,
            left_over: len,
        }
    }
}

impl<'a, T: Copy> Iterator for ContiguousIter<'a, T> {
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

impl<'a, T: Copy> ExactSizeIterator for ContiguousIter<'a, T> {}

impl<'a, T: Copy> FusedIterator for ContiguousIter<'a, T> {}

///////////////////////////////////////////////////////////////

// TODO: This impl is not correct anymore. If used please fix it like
// the non-mut one.

// pub struct MutContiguousIter<'a, T: Copy> {
//     data: RwLockWriteGuard<'a, Vec<T>>,
//     index: usize,
// }
//
// impl<'a, T: Copy> MutContiguousIter<'a, T> {
//     pub fn new(lock: &'a RwLock<Vec<T>>) -> Self {
//         let data: RwLockWriteGuard<'_, Vec<T>> = lock.write();
//         Self { data, index: 0 }
//     }
// }
//
// impl<'a, T: Copy> Iterator for MutContiguousIter<'a, T> {
//     type Item = &'a mut T;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index >= self.data.len() {
//             return None;
//         }
//
//         let mut item = NonNull::new(&mut self.data[self.index] as *mut T).unwrap();
//         self.index += 1;
//
//         return Some(unsafe { item.as_mut() });
//     }
//
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         let len = self.data.len() - self.index;
//
//         (len, Some(len))
//     }
// }
//
// impl<'a, T: Copy> ExactSizeIterator for MutContiguousIter<'a, T> {}
//
// impl<'a, T: Copy> FusedIterator for MutContiguousIter<'a, T> {}
//
///////////////////////////////////////////////////////////////

pub struct CopiedContiguousIter<'a, T: Copy> {
    data: &'a Arc<Vec<T>>,
    offset: usize,
    left_over: usize,
}

impl<'a, T: Copy> CopiedContiguousIter<'a, T> {
    pub fn new(data: &'a Arc<Vec<T>>, offset: usize, len: usize) -> Self {
        Self {
            data,
            offset,
            left_over: len,
        }
    }
}

impl<'a, T: Copy> Iterator for CopiedContiguousIter<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.left_over == 0 {
            return None;
        }

        let item = self.data[self.offset];
        self.offset += 1;
        self.left_over -= 1;

        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left_over, Some(self.left_over))
    }
}

impl<'a, T: Copy> ExactSizeIterator for CopiedContiguousIter<'a, T> {}

impl<'a, T: Copy> FusedIterator for CopiedContiguousIter<'a, T> {}

///////////////////////////////////////////////////////////////

pub struct SliceIter<'a, T: Copy> {
    data: &'a Arc<Vec<T>>,
    pos: isize,
    counter: Box<[usize]>,
    layout: &'a Layout,
    left_over: usize,
}

impl<'a, T: Copy> SliceIter<'a, T> {
    pub fn new(data: &'a Arc<Vec<T>>, data_len: usize, layout: &'a Layout) -> Self {
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

impl<'a, T: Copy> Iterator for SliceIter<'a, T> {
    type Item = &'a T;

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
            let item = &self.data[pos] as *const T;
            self.pos += self.layout.adj_stride()[step_dim] as isize;
            self.left_over -= 1;

            Some(&*item)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left_over, Some(self.left_over))
    }
}

impl<'a, T: Copy> ExactSizeIterator for SliceIter<'a, T> {}

impl<'a, T: Copy> FusedIterator for SliceIter<'a, T> {}

///////////////////////////////////////////////////////////////

// pub struct MutSliceIter<'a, T: Copy> {
//     data: RwLockWriteGuard<'a, Vec<T>>,
//     pos: isize,
//     counter: Box<[i32]>,
//     layout: &'a Layout,
//     left_over: usize,
// }
//
// impl<'a, T: Copy> MutSliceIter<'a, T> {
//     pub fn new(lock: &'a RwLock<Vec<T>>, data_len: usize, layout: &'a Layout) -> Self {
//         let counter = vec![0; layout.shape().len()].into_boxed_slice();
//
//         Self {
//             data: lock.write(),
//             pos: layout.offset() as isize,
//             layout,
//             counter,
//             left_over: data_len,
//         }
//     }
// }
//
// impl<'a, T: Copy> Iterator for MutSliceIter<'a, T> {
//     type Item = &'a mut T;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.left_over == 0 {
//             return None;
//         }
//
//         let last = self.counter.len() - 1;
//         self.counter[last] += 1;
//         let mut step_dim = last;
//
//         for dim in (1..self.counter.len()).rev() {
//             if self.counter[dim] == self.layout.shape[dim] {
//                 self.counter[dim] = 0;
//                 self.counter[dim - 1] += 1;
//
//                 step_dim = dim - 1;
//                 continue;
//             }
//             break;
//         }
//
//         let step = self.layout.adj_stride()[step_dim];
//         unsafe {
//             let item_ptr = &mut self.data[self.pos as usize] as *mut T;
//             self.pos += step as isize;
//             self.left_over -= 1;
//
//             Some(&mut *item_ptr)
//         }
//     }
//
//     fn size_hint(&self) -> (usize, Option<usize>) {
//         (self.left_over, Some(self.left_over))
//     }
// }
//
// impl<'a, T: Copy> ExactSizeIterator for MutSliceIter<'a, T> {}
//
// impl<'a, T: Copy> FusedIterator for MutSliceIter<'a, T> {}
//
///////////////////////////////////////////////////////////////

pub struct CopiedSliceIter<'a, T: Copy> {
    data: &'a Arc<Vec<T>>,
    pos: isize,
    counter: Box<[usize]>,
    layout: &'a Layout,
    left_over: usize,
}

impl<'a, T: Copy> CopiedSliceIter<'a, T> {
    pub fn new(data: &'a Arc<Vec<T>>, data_len: usize, layout: &'a Layout) -> Self {
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

impl<'a, T: Copy> Iterator for CopiedSliceIter<'a, T> {
    type Item = T;

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

        let item = self.data[pos];
        self.pos += self.layout.adj_stride()[step_dim] as isize;
        self.left_over -= 1;

        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left_over, Some(self.left_over))
    }
}

impl<'a, T: Copy> ExactSizeIterator for CopiedSliceIter<'a, T> {}

impl<'a, T: Copy> FusedIterator for CopiedSliceIter<'a, T> {}

///////////////////////////////////////////////////////////////

pub enum StepInfo<T: Copy> {
    EnterDimension(usize),
    ExitDimension(usize),
    Value(T),
    End,
}

pub struct InformedSliceIter<'a, T: Copy> {
    buffer: &'a Arc<Vec<T>>,
    layout: &'a Layout,
    next_state: StepInfo<T>,
    pos: i64,
    counter: Vec<usize>,
}

impl<'a, T: Copy> InformedSliceIter<'a, T> {
    pub fn new(data: &'a Arc<Vec<T>>, layout: &'a Layout) -> Self {
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

impl<'a, T: Copy> Iterator for InformedSliceIter<'a, T> {
    type Item = StepInfo<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_state {
            StepInfo::EnterDimension(dim) => {
                if dim == self.layout.shape().len() - 1 {
                    debug_assert_positive!(self.pos);

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

                debug_assert_positive!(self.pos);

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

impl<'a, T: Copy> ExactSizeIterator for InformedSliceIter<'a, T> {}

impl<'a, T: Copy> FusedIterator for InformedSliceIter<'a, T> {}

/////////////////////////////////////////////////////////////
pub struct PackedBuffer<'a, T: Copy> {
    pub packing_buffer: &'a [T],
    pub absolute_buffer_position: usize,
}

pub struct ChunkedSliceIter<I, T: Copy, const N: usize>
where
    I: IntoIterator<Item = T>,
{
    iter: I::IntoIter,
    packing_buffer: [T; N],
    absolute_buffer_position: usize,
}

impl<I, T: Copy + Default, const N: usize> ChunkedSliceIter<I, T, N>
where
    I: Iterator<Item = T>,
{
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            packing_buffer: [T::default(); N],
            absolute_buffer_position: 0,
        }
    }
}

impl<I, T: Copy, const N: usize> StreamingIterator for ChunkedSliceIter<I, T, N>
where
    I: IntoIterator<Item = T>,
{
    type Item<'a>
        = PackedBuffer<'a, T>
    where
        Self: 'a;

    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>> {
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
