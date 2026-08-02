use std::convert::Infallible;
use std::iter::zip;
use std::ops::ControlFlow;

use crate::{Layout, tensor::MAX_DIMS};

fn calculate_adjacent_dim_stride(stride: &[i32], slice_shape: &[usize]) -> [i32; MAX_DIMS] {
    let rank = stride.len();
    debug_assert!(rank >= 1, "stride must have rank >= 1");

    let mut v = [0i32; MAX_DIMS];
    v[..rank].copy_from_slice(stride);

    let mut accum: i32 = 0;
    for i in (0..rank - 1).rev() {
        accum += stride[i + 1] * (slice_shape[i + 1] as i32 - 1);
        v[i] -= accum;
    }

    v
}

fn simplify_layout<const N: usize>(
    layouts: [&Layout; N],
    l_adj_strides: [[i32; MAX_DIMS]; N],
) -> (usize, [usize; MAX_DIMS], [[i32; MAX_DIMS]; N]) {
    let rank: usize = layouts[0].shape().len();
    let mut shape = [0usize; MAX_DIMS];
    let mut adj_strides = [[0i32; MAX_DIMS]; N];

    let mut w: usize = 0;

    shape[0] = layouts[0].shape()[0];
    for i in 0..N {
        adj_strides[i][0] = l_adj_strides[i][0];
    }

    for i in 1..rank {
        let mut mergeable = true;
        for n in 0..N {
            if l_adj_strides[n][i] != l_adj_strides[n][i - 1] {
                mergeable = false;
                break;
            }
        }

        if mergeable {
            shape[w] *= layouts[0].shape()[i];
        } else {
            w += 1;
            shape[w] = layouts[0].shape()[i];
            for x in 0..N {
                adj_strides[x][w] = l_adj_strides[x][i];
            }
        }
    }

    (w + 1, shape, adj_strides)
}

pub struct DimWalker<'a, const N: usize> {
    rank: usize,
    layouts: [&'a Layout; N],
    shape: [usize; MAX_DIMS],
    baked_stride: [[isize; MAX_DIMS]; N],
    chunk_len: usize,
    is_fully_contiguous: bool,
}

impl<'a, const N: usize> DimWalker<'a, N> {
    pub fn new(layouts: [&'a Layout; N]) -> Self {
        if layouts.iter().all(|l| l.is_contiguous()) {
            let rank = layouts[0].shape().len();
            let baked_stride = [[1isize; MAX_DIMS]; N];

            return Self {
                rank,
                layouts,
                shape: [0; MAX_DIMS],
                baked_stride,
                chunk_len: layouts[0].len(),
                is_fully_contiguous: true,
            };
        }

        let l_adj_strides = layouts.map(|l| calculate_adjacent_dim_stride(l.stride(), l.shape()));

        let (mut rank, mut shape, mut adj_strides) = simplify_layout(layouts, l_adj_strides);

        if rank == 1 {
            shape[1] = shape[0];
            shape[0] = 1;

            for i in 0..N {
                adj_strides[i][1] = adj_strides[i][0];
            }

            rank += 1;
        }

        let last = rank - 1;

        let steps =
            adj_strides.map(|adj_stride| adj_stride[last] as isize * (shape[last] - 1) as isize);

        let mut i: usize = 0;
        let baked_stride: [[isize; MAX_DIMS]; N] = adj_strides.map(|adj_stride| {
            let mut temp = [0isize; MAX_DIMS];
            let step = steps[i];
            let chunk_stride = adj_stride[last - 1] as isize + step;
            temp[last] = adj_stride[last] as isize;
            temp[last - 1] = chunk_stride;

            for x in 0..rank - 2 {
                temp[x] = adj_stride[x] as isize + step - chunk_stride;
            }
            i += 1;

            temp
        });

        Self {
            rank,
            layouts,
            shape,
            baked_stride,
            chunk_len: shape[rank - 1],
            is_fully_contiguous: false,
        }
    }

    pub fn strides(&self) -> (usize, [isize; N]) {
        (
            self.chunk_len,
            self.baked_stride
                .map(|adj_stride| adj_stride[self.rank - 1]),
        )
    }

    pub fn is_fully_contiguous(&self) -> bool {
        self.is_fully_contiguous
    }

    pub fn try_fold<A, B>(
        &self,
        init: A,
        mut f: impl FnMut(A, [usize; N]) -> ControlFlow<B, A>,
    ) -> ControlFlow<B, A> {
        debug_assert!(!self.is_fully_contiguous);

        let last = self.rank - 1;
        let mut counter: [usize; MAX_DIMS] = [0; MAX_DIMS];

        let n_chunks = self.shape[last - 1];
        let chunk_stride: [isize; N] = self.baked_stride.map(|s| s[last - 1]);

        let mut offsets: [usize; N] = self.layouts.map(|l| l.offset());
        let mut acc = init;

        if self.rank == 2 {
            let left_over = self.shape[0];
            for _ in 0..left_over {
                acc = f(acc, offsets)?;

                for i in 0..N {
                    offsets[i] = offsets[i].wrapping_add_signed(chunk_stride[i]);
                }
            }
            return ControlFlow::Continue(acc);
        }

        if self.rank == 3 {
            let chunks = self.shape[0] * n_chunks;
            let mut count = n_chunks;
            for _ in 0..chunks {
                acc = f(acc, offsets)?;

                for i in 0..N {
                    offsets[i] = offsets[i].wrapping_add_signed(chunk_stride[i]);
                }

                count -= 1;
                if count == 0 {
                    count = n_chunks;
                    for i in 0..N {
                        offsets[i] = offsets[i].wrapping_add_signed(self.baked_stride[i][0]);
                    }
                }
            }
            return ControlFlow::Continue(acc);
        }

        let chunks: usize = self.shape[0..last].iter().product();
        let mut count = n_chunks;
        for _ in 0..chunks {
            acc = f(acc, offsets)?;

            for i in 0..N {
                offsets[i] = offsets[i].wrapping_add_signed(chunk_stride[i]);
            }

            count -= 1;
            if count == 0 {
                count = n_chunks;

                let last_counter = last - 2;
                counter[last_counter] += 1;
                let mut step_dim = last_counter;
                for dim in (1..last - 1).rev() {
                    if counter[dim] == self.shape[dim] {
                        counter[dim] = 0;
                        counter[dim - 1] += 1;
                        step_dim = dim - 1;
                        continue;
                    }
                    break;
                }

                for i in 0..N {
                    offsets[i] = offsets[i].wrapping_add_signed(self.baked_stride[i][step_dim]);
                }
            }
        }

        ControlFlow::Continue(acc)
    }

    pub fn fold<A>(&self, init: A, mut f: impl FnMut(A, [usize; N]) -> A) -> A {
        match self.try_fold(init, |acc, offsets| {
            ControlFlow::<Infallible, A>::Continue(f(acc, offsets))
        }) {
            ControlFlow::Continue(acc) => acc,
            ControlFlow::Break(never) => match never {},
        }
    }

    pub fn for_each(&self, mut f: impl FnMut([usize; N])) {
        self.fold((), |(), offsets| f(offsets));
    }
}

///////////////////////////////////////////////////////////////

#[inline]
pub fn fold_ref<'a, T, B, F>(inp: &'a [T], l: &Layout, init: B, mut f: F) -> B
where
    F: FnMut(B, &'a T) -> B,
{
    let walker = DimWalker::new([l]);

    if walker.is_fully_contiguous {
        let offset = l.offset();
        return inp[offset..offset + l.len()].iter().fold(init, &mut f);
    }

    let (len, strides) = walker.strides();
    match strides {
        [1] => walker.fold(init, |acc, offsets| {
            let offset = offsets[0];
            inp[offset..offset + len].iter().fold(acc, &mut f)
        }),
        [0] => walker.fold(init, |mut acc, offsets| {
            let offset = offsets[0];
            // SAFETY: a well-formed layout only ever visits in-bounds
            // positions of its own buffer.
            let el = unsafe { inp.get_unchecked(offset) };
            for _ in 0..len {
                acc = f(acc, el);
            }
            acc
        }),
        [s] => walker.fold(init, |mut acc, offsets| {
            let mut offset = offsets[0];
            for _ in 0..len {
                debug_assert!(offset < inp.len());
                // SAFETY: a well-formed layout only ever visits in-bounds
                // positions of its own buffer.
                acc = f(acc, unsafe { inp.get_unchecked(offset) });
                offset = offset.wrapping_add_signed(s);
            }
            acc
        }),
    }
}

#[inline]
pub fn fold<T: Clone, B, F>(inp: &[T], l: &Layout, init: B, mut f: F) -> B
where
    F: FnMut(B, T) -> B,
{
    let walker = DimWalker::new([l]);

    if walker.is_fully_contiguous {
        let offset = l.offset();
        return inp[offset..offset + l.len()]
            .iter()
            .cloned()
            .fold(init, &mut f);
    }

    let (len, strides) = walker.strides();
    match strides {
        [1] => walker.fold(init, |acc, offsets| {
            let offset = offsets[0];
            inp[offset..offset + len].iter().cloned().fold(acc, &mut f)
        }),
        [0] => walker.fold(init, |mut acc, offsets| {
            let offset = offsets[0];
            // SAFETY: a well-formed layout only ever visits in-bounds
            // positions of its own buffer.
            let el = unsafe { inp.get_unchecked(offset) }.clone();
            for _ in 0..len {
                acc = f(acc, el.clone());
            }
            acc
        }),
        [s] => walker.fold(init, |mut acc, offsets| {
            let mut offset = offsets[0];
            for _ in 0..len {
                debug_assert!(offset < inp.len());
                // SAFETY: a well-formed layout only ever visits in-bounds
                // positions of its own buffer.
                acc = f(acc, unsafe { inp.get_unchecked(offset) }.clone());
                offset = offset.wrapping_add_signed(s);
            }
            acc
        }),
    }
}

#[inline]
pub fn fold_chunk<'a, 'b, T: Clone, B, R, E>(
    inp: &'a [T],
    l: &Layout,
    init: B,
    mut ch: R,
    mut elem: E,
) -> B
where
    R: FnMut(B, &'a [T]) -> B,
    E: FnMut(B, T) -> B,
{
    let walker = DimWalker::new([l]);
    let acc = init;

    if walker.is_fully_contiguous {
        let offset = l.offset();
        return ch(acc, &inp[offset..offset + l.len()]);
    }

    let (len, strides) = walker.strides();
    match strides {
        [1] => walker.fold(acc, |acc, offsets| {
            let offset = offsets[0];
            ch(acc, &inp[offset..offset + len])
        }),
        [0] => walker.fold(acc, |mut acc, offsets| {
            let temp = inp[offsets[0]].clone();
            for _ in 0..len {
                acc = elem(acc, temp.clone())
            }

            acc
        }),
        [s] => walker.fold(acc, |mut acc, offsets| {
            let mut offset = offsets[0];
            for _ in 0..len {
                debug_assert!(offset < inp.len());
                // SAFETY: a well-formed layout only ever visits in-bounds
                // positions of its own buffer.
                acc = elem(acc, unsafe { inp.get_unchecked(offset) }.clone());
                offset = offset.wrapping_add_signed(s);
            }

            acc
        }),
    }
}

///////////////////////////////////////////////////////////////

#[inline]
pub fn map_chunk<'a, 'b, T: Clone, R, E>(
    inp: &'a [T],
    l: &Layout,
    out: &'b mut [T],
    mut ch: R,
    mut elem: E,
) where
    R: FnMut(&'a [T], &'b mut [T]),
    E: FnMut(T) -> T,
{
    let walker = DimWalker::new([l]);

    if walker.is_fully_contiguous {
        let offset = l.offset();
        ch(&inp[offset..offset + l.len()], out);
        return;
    }

    let (len, strides) = walker.strides();
    let mut chunks = out.chunks_exact_mut(len);
    match strides {
        [1] => walker.for_each(|offsets| {
            let chunk = unsafe { next_chunk(&mut chunks) };
            let offset = offsets[0];
            ch(&inp[offset..offset + len], chunk);
        }),
        [0] => walker.for_each(|offsets| {
            let chunk = unsafe { next_chunk(&mut chunks) };
            let temp = inp[offsets[0]].clone();

            chunk.fill(elem(temp));
        }),
        [s] => walker.for_each(|offsets| {
            let chunk = unsafe { next_chunk(&mut chunks) };
            let mut offset = offsets[0];
            for o in chunk.iter_mut() {
                debug_assert!(offset < inp.len());
                // SAFETY: a well-formed layout only ever visits in-bounds
                // positions of its own buffer.
                *o = elem(unsafe { inp.get_unchecked(offset) }.clone());
                offset = offset.wrapping_add_signed(s);
            }
        }),
    }
}

#[inline]
pub fn map_chunk_inplace<'a, T: Clone, R, E>(out: &mut [T], l: &Layout, mut ch: R, mut elem: E)
where
    R: FnMut(&mut [T]),
    E: FnMut(T) -> T,
{
    let walker = DimWalker::new([l]);

    if walker.is_fully_contiguous {
        let offset = l.offset();
        ch(&mut out[offset..offset + l.len()]);
        return;
    }

    let (len, strides) = walker.strides();
    match strides {
        [1] => walker.for_each(|offsets| {
            let offset = offsets[0];
            ch(&mut out[offset..offset + len]);
        }),
        [0] => walker.for_each(|offsets| {
            let offset = offsets[0];
            let temp = out[offset].clone();

            out[offset..offset + len].fill(elem(temp));
        }),
        [s] => walker.for_each(|offsets| {
            let mut offset = offsets[0];
            for _ in 0..len {
                debug_assert!(offset < out.len());
                // SAFETY: a well-formed layout only ever visits in-bounds
                // positions of its own buffer.
                out[offset] = elem(unsafe { out.get_unchecked(offset) }.clone());
                offset = offset.wrapping_add_signed(s);
            }
        }),
    }
}

///////////////////////////////////////////////////////////////

#[inline(always)]
unsafe fn next_chunk<'a, T>(chunks: &mut std::slice::ChunksExactMut<'a, T>) -> &'a mut [T] {
    let chunk = chunks.next();
    debug_assert!(chunk.is_some(), "walk emitted more chunks than `out` holds");
    // SAFETY: the caller drives this iterator in lockstep with the walk, which
    // yields exactly as many chunks as `out` was split into.
    unsafe { chunk.unwrap_unchecked() }
}

pub fn map2<T: Clone, F: Fn(T, T) -> T>(
    inp1: &[T],
    l1: &Layout,
    inp2: &[T],
    l2: &Layout,
    out: &mut [T],
    f: F,
) {
    let walker = DimWalker::new([l1, l2]);

    if walker.is_fully_contiguous {
        let (o1, o2) = (l1.offset(), l2.offset());
        let len = l1.len();
        let it1 = inp1[o1..o1 + len].iter();
        let it2 = inp2[o2..o2 + len].iter();

        for (o, (x, y)) in out.iter_mut().zip(zip(it1, it2)) {
            *o = f(x.clone(), y.clone());
        }
        return;
    }

    let (len, strides) = walker.strides();
    let mut chunks = out.chunks_exact_mut(len);
    match strides {
        [1, 1] => {
            walker.for_each(|offsets| {
                let chunk = unsafe { next_chunk(&mut chunks) };
                let it1 = inp1[offsets[0]..offsets[0] + len].iter();
                let it2 = inp2[offsets[1]..offsets[1] + len].iter();

                for (o, (x, y)) in chunk.iter_mut().zip(zip(it1, it2)) {
                    *o = f(x.clone(), y.clone());
                }
            });
        }
        [0, 0] => {
            walker.for_each(|offsets| {
                let chunk = unsafe { next_chunk(&mut chunks) };
                chunk.fill(f(inp1[offsets[0]].clone(), inp2[offsets[1]].clone()));
            });
        }
        [0, _] => {
            walker.for_each(|offsets| {
                let chunk = unsafe { next_chunk(&mut chunks) };
                let x = inp1[offsets[0]].clone();

                for (o, y) in chunk
                    .iter_mut()
                    .zip(inp2[offsets[1]..offsets[1] + len].iter())
                {
                    *o = f(x.clone(), y.clone());
                }
            });
        }
        [_, 0] => {
            walker.for_each(|offsets| {
                let chunk = unsafe { next_chunk(&mut chunks) };
                let y = inp2[offsets[1]].clone();

                for (o, x) in chunk
                    .iter_mut()
                    .zip(inp1[offsets[0]..offsets[0] + len].iter())
                {
                    *o = f(x.clone(), y.clone());
                }
            });
        }
        [s1, s2] => {
            walker.for_each(|offsets| {
                let chunk = unsafe { next_chunk(&mut chunks) };
                let mut pos1 = offsets[0];
                let mut pos2 = offsets[1];

                for o in chunk.iter_mut() {
                    debug_assert!(pos1 < inp1.len());
                    debug_assert!(pos2 < inp2.len());
                    // SAFETY: a well-formed layout only ever visits in-bounds
                    // positions of its own buffer.
                    *o = unsafe {
                        f(
                            inp1.get_unchecked(pos1).clone(),
                            inp2.get_unchecked(pos2).clone(),
                        )
                    };
                    pos1 = pos1.wrapping_add_signed(s1);
                    pos2 = pos2.wrapping_add_signed(s2);
                }
            });
        }
    }
}

/// Assumes that the output is contiguous! Don't run on this otherwise!
///
/// Assumes the ordering (out, inp) for `f`
#[inline]
pub fn map2_inplace<'a, T: Clone, F>(out: &mut [T], inp: &[T], l: &Layout, f: F)
where
    F: Fn(T, T) -> T,
{
    let walker = DimWalker::new([l]);

    if walker.is_fully_contiguous {
        let o = l.offset();
        let it = inp[o..o + l.len()].iter();

        for (o, x) in out.iter_mut().zip(it) {
            *o = f(o.clone(), x.clone());
        }
        return;
    }

    let (len, strides) = walker.strides();
    let mut chunks = out.chunks_exact_mut(len);
    match strides {
        [1] => {
            walker.for_each(|offsets| {
                let chunk = unsafe { next_chunk(&mut chunks) };
                let it = inp[offsets[0]..offsets[0] + len].iter();

                for (o, x) in chunk.iter_mut().zip(it) {
                    *o = f(o.clone(), x.clone());
                }
            });
        }
        [0] => {
            walker.for_each(|offsets| {
                let chunk = unsafe { next_chunk(&mut chunks) };
                let x = inp[offsets[0]].clone();

                chunk.iter_mut().for_each(|o| *o = f(o.clone(), x.clone()));
            });
        }
        [s] => {
            walker.for_each(|offsets| {
                let chunk = unsafe { next_chunk(&mut chunks) };
                let mut pos = offsets[0];

                for o in chunk.iter_mut() {
                    debug_assert!(pos < inp.len());
                    // SAFETY: a well-formed layout only ever visits in-bounds
                    // positions of its own buffer.
                    *o = unsafe { f(o.clone(), inp.get_unchecked(pos).clone()) };
                    pos = pos.wrapping_add_signed(s);
                }
            });
        }
    }
}

#[inline]
pub fn all2<T, F>(inp1: &[T], l1: &Layout, inp2: &[T], l2: &Layout, mut f: F) -> bool
where
    F: FnMut(&T, &T) -> bool,
{
    let walker = DimWalker::new([l1, l2]);

    if walker.is_fully_contiguous {
        let (o1, o2) = (l1.offset(), l2.offset());
        let len = l1.len();
        let it1 = inp1[o1..o1 + len].iter();
        let it2 = inp2[o2..o2 + len].iter();

        return zip(it1, it2).all(|(x, y)| f(x, y));
    }

    let (len, strides) = walker.strides();
    let walk = match strides {
        [1, 1] => walker.try_fold((), |(), offsets| {
            let it1 = inp1[offsets[0]..offsets[0] + len].iter();
            let it2 = inp2[offsets[1]..offsets[1] + len].iter();

            if zip(it1, it2).all(|(x, y)| f(x, y)) {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            }
        }),
        [0, 0] => walker.try_fold((), |(), offsets| {
            if f(&inp1[offsets[0]], &inp2[offsets[1]]) {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            }
        }),
        [0, _] => walker.try_fold((), |(), offsets| {
            let x = &inp1[offsets[0]];

            if inp2[offsets[1]..offsets[1] + len].iter().all(|y| f(x, y)) {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            }
        }),
        [_, 0] => walker.try_fold((), |(), offsets| {
            let y = &inp2[offsets[1]];

            if inp1[offsets[0]..offsets[0] + len].iter().all(|x| f(x, y)) {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            }
        }),
        [s1, s2] => walker.try_fold((), |(), offsets| {
            let mut pos1 = offsets[0];
            let mut pos2 = offsets[1];

            for _ in 0..len {
                debug_assert!(pos1 < inp1.len());
                debug_assert!(pos2 < inp2.len());
                // SAFETY: a well-formed layout only ever visits in-bounds
                // positions of its own buffer.
                let (x, y) = unsafe { (inp1.get_unchecked(pos1), inp2.get_unchecked(pos2)) };

                if !f(x, y) {
                    return ControlFlow::Break(());
                }

                pos1 = pos1.wrapping_add_signed(s1);
                pos2 = pos2.wrapping_add_signed(s2);
            }

            ControlFlow::Continue(())
        }),
    };

    walk.is_continue()
}

///////////////////////////////////////////////////////////////
