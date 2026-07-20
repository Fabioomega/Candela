use candela::backend::{Backend, ComputeFor, CpuPure};
use candela::skeleton::{Skeleton, SkeletonSlot};
use candela::{Dimension, Layout};
use criterion::{
    AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group, criterion_main,
};
use std::hint::black_box;
use std::iter::FusedIterator;
use std::iter::successors;

mod common;
use common::{FillConfig, ShapePolicy, SizeSpec, Variant};

//////////////////////////////////////////////////////////////////

const MAX_DIMS: usize = 8;

#[derive(Debug)]
struct ChunkIter<'a, T> {
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
enum ChunkKind<'a, T> {
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

fn collect_chunks<T: Clone>(it: ChunkIter<'_, T>) -> Vec<T> {
    let mut v = Vec::with_capacity(it.len());

    for chunk in it {
        match chunk {
            ChunkKind::Contiguous { data, start, times } => {
                v.extend_from_slice(&data[start..start + times]);
            }
            ChunkKind::Strided {
                data,
                start,
                times,
                step,
            } => {
                for i in successors(Some(start), |&i| i.checked_add_signed(step)).take(times) {
                    v.push(data[i].clone());
                }
            }
        }
    }

    v
}

//////////////////////////////////////////////////////////////////

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
        let mut left_over: usize = shape[0..last - 1].iter().product();
        left_over = left_over.max(1);

        let chunk_size = shape[last];
        let n_chunks = shape[last - 1];
        let step = adj_stride[last] as isize * (shape[last] - 1) as isize;
        let stride = adj_stride[last] as isize;
        let chunk_stride = adj_stride[last - 1] as isize + step;

        let baked_stride: [isize; MAX_DIMS] = adj_stride.map(|x| x as isize + step - chunk_stride);

        let next_chunk = |counter: &mut [usize; MAX_DIMS]| -> usize {
            let last_counter = last - 2;
            counter[last_counter] += 1;
            let mut step_dim = last_counter;
            for dim in (1..last - 1).rev() {
                if counter[dim] == shape[dim] {
                    counter[dim] = 0;
                    counter[dim - 1] += 1;
                    step_dim = dim - 1;
                    continue;
                }
                break;
            }

            step_dim
        };

        if stride == 1 {
            for _ in 0..left_over {
                for _ in 0..n_chunks {
                    for el in self.data[pos..pos + chunk_size].iter() {
                        acc = f(acc, el);
                    }

                    pos = pos.wrapping_add_signed(chunk_stride);
                }

                let step_dim = next_chunk(&mut counter);
                pos = pos.wrapping_add_signed(baked_stride[step_dim]);
            }
        } else if stride == 0 {
            for _ in 0..left_over {
                for _ in 0..n_chunks {
                    for _ in 0..chunk_size {
                        acc = f(acc, &self.data[pos]);
                    }

                    pos = pos.wrapping_add_signed(chunk_stride);
                }

                let step_dim = next_chunk(&mut counter);
                pos = pos.wrapping_add_signed(baked_stride[step_dim]);
            }
        } else {
            for _ in 0..left_over {
                for _ in 0..n_chunks {
                    let mut pos_inner = pos;
                    for _ in 0..chunk_size {
                        debug_assert!(pos_inner < self.data.len());
                        // SAFETY: a well-formed layout only ever visits in-bounds
                        // positions of its own buffer, so `pos_inner` is a valid
                        // index. Dropping the bounds check keeps the strided read
                        // from stalling memory-level parallelism on gather-heavy
                        // layouts (e.g. transposed).
                        acc = f(acc, unsafe { self.data.get_unchecked(pos_inner) });

                        pos_inner = pos_inner.wrapping_add_signed(stride);
                    }

                    pos = pos.wrapping_add_signed(chunk_stride);
                }

                let step_dim = next_chunk(&mut counter);
                pos = pos.wrapping_add_signed(baked_stride[step_dim]);
            }
        }

        acc
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

impl<'a, T> FusedIterator for Iter<'a, T> {}

//////////////////////////////////////////////////////////////////

// Used only to get the cases out nicely
fn scalar_add<B: Backend>(layouts: &[Layout]) -> Skeleton<f32, B>
where
    f32: ComputeFor<B>,
{
    let a: SkeletonSlot<f32, B> = SkeletonSlot::new(layouts[0].clone());
    (&a + 0.0f32)
        .into_skeleton(std::slice::from_ref(&a))
        .unwrap()
}

/// Hand-rolled vectorizable copy iterator
fn copy_kernel(a: &[f32], layout: &Layout, out: &mut [f32]) {
    let a = &a[layout.offset()..];

    if layout.is_contiguous() {
        out.copy_from_slice(&a[..out.len()]);
    } else if layout.shape().len() == 1 && layout.stride()[0] > 1 {
        let step = layout.stride()[0] as usize;
        let mut i = 0usize;
        for o in out.iter_mut() {
            debug_assert!(i < a.len());
            // SAFETY: the buffer is sized to hold the layout's last position,
            // so every index this walk produces is in bounds. `step_by` would
            // be the safe spelling, but it optimizes poorly enough to make this
            // kernel slower than the iterator it is supposed to be the ceiling
            // for.
            *o = unsafe { *a.get_unchecked(i) };
            i += step;
        }
    } else if layout.shape().len() == 2 && layout.stride()[1] == 0 {
        let cols = layout.shape()[1];
        let pitch = layout.stride()[0] as usize;
        for (r, row_out) in out.chunks_exact_mut(cols).enumerate() {
            row_out.fill(a[r * pitch]);
        }
    } else if layout.shape().len() == 2 && layout.stride()[0] == 0 {
        // outer-broadcast: one source row reused across every output row.
        let cols = layout.shape()[1];
        let inner = layout.stride()[1] as usize;
        if inner == 1 {
            for row_out in out.chunks_exact_mut(cols) {
                row_out.copy_from_slice(&a[..cols]);
            }
        } else {
            for row_out in out.chunks_exact_mut(cols) {
                for (j, o) in row_out.iter_mut().enumerate() {
                    *o = a[j * inner];
                }
            }
        }
    } else if layout.shape().len() == 2 && layout.stride()[1] == 1 {
        let cols = layout.shape()[1];
        let pitch = layout.stride()[0] as usize;
        for (r, row_out) in out.chunks_exact_mut(cols).enumerate() {
            row_out.copy_from_slice(&a[r * pitch..r * pitch + cols]);
        }
    } else if layout.shape().len() == 2 && layout.stride()[0] == 1 {
        let (rows, cols) = (layout.shape()[0], layout.shape()[1]);
        let inner = layout.stride()[1] as usize;
        for i in 0..rows {
            for j in 0..cols {
                out[i * cols + j] = a[j * inner + i];
            }
        }
    } else {
        panic!("no best copy kernel for layout {layout:?}");
    }
}

/// Hand-rolled vectorizable `* 2.0` kernel. Unlike [`copy_kernel`] it can't fall
/// back to `copy_from_slice`, so it's the fair best for a non-copy op: every
/// path multiplies element-by-element, the same work `fold` has to do.
fn mul_kernel(a: &[f32], layout: &Layout, out: &mut [f32]) {
    let a = &a[layout.offset()..];

    if layout.is_contiguous() {
        for (o, x) in out.iter_mut().zip(a.iter()) {
            *o = *x * 2.0;
        }
    } else if layout.shape().len() == 1 && layout.stride()[0] > 1 {
        let step = layout.stride()[0] as usize;
        let mut i = 0usize;
        for o in out.iter_mut() {
            debug_assert!(i < a.len());
            // SAFETY: see `copy_kernel` - the buffer holds the layout's last
            // position, so every index this walk produces is in bounds.
            *o = unsafe { *a.get_unchecked(i) } * 2.0;
            i += step;
        }
    } else if layout.shape().len() == 2 && layout.stride()[1] == 0 {
        let cols = layout.shape()[1];
        let pitch = layout.stride()[0] as usize;
        for (r, row_out) in out.chunks_exact_mut(cols).enumerate() {
            row_out.fill(a[r * pitch] * 2.0);
        }
    } else if layout.shape().len() == 2 && layout.stride()[0] == 0 {
        let cols = layout.shape()[1];
        let inner = layout.stride()[1] as usize;
        let (first, rest) = out.split_at_mut(cols);
        for (j, o) in first.iter_mut().enumerate() {
            *o = a[j * inner] * 2.0;
        }
        for row_out in rest.chunks_exact_mut(cols) {
            row_out.copy_from_slice(first);
        }
    } else if layout.shape().len() == 2 && layout.stride()[1] == 1 {
        let cols = layout.shape()[1];
        let pitch = layout.stride()[0] as usize;
        for (r, row_out) in out.chunks_exact_mut(cols).enumerate() {
            for (o, x) in row_out
                .iter_mut()
                .zip(a[r * pitch..r * pitch + cols].iter())
            {
                *o = *x * 2.0;
            }
        }
    } else if layout.shape().len() == 2 && layout.stride()[0] == 1 {
        let (rows, cols) = (layout.shape()[0], layout.shape()[1]);
        let inner = layout.stride()[1] as usize;
        for i in 0..rows {
            for j in 0..cols {
                out[i * cols + j] = a[j * inner + i] * 2.0;
            }
        }
    } else {
        panic!("no best mul kernel for layout {layout:?}");
    }
}

/// The layout/size sweep shared by every bench in this file.
fn standard_config() -> FillConfig {
    FillConfig::new(1)
        .variants(&[
            Variant::Contig,
            Variant::Step,
            Variant::Padded,
            Variant::Transposed,
        ])
        .sizes(&[
            SizeSpec::Elems(64),
            SizeSpec::Elems(1024),
            SizeSpec::L1,
            SizeSpec::L2,
            SizeSpec::L3,
            SizeSpec::Dram,
        ])
}

fn iteration(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("iteration");
    group.plot_config(plot_config);

    let cfg = standard_config();

    for case in common::fill_cases(&cfg, scalar_add::<CpuPure>) {
        group.throughput(case.throughput);

        let input = &case.inputs[0];
        let data = input.data();
        let layout = input.layout();
        let n = case.skeleton.len();

        let it = ChunkIter::new(data, layout);
        let it2 = Iter::new(data, layout);

        let mut expected = vec![0.0f32; n];
        copy_kernel(data, layout, &mut expected);
        let from_elem: Vec<f32> = collect_chunks(it);
        let from_elem2: Vec<f32> = it2.cloned().collect();
        assert_eq!(
            from_elem, expected,
            "ChunkIter order wrong at {}",
            case.label
        );
        assert_eq!(from_elem2, expected, "Iter order wrong at {}", case.label);

        let from_fold: Vec<f32> = {
            let mut v = Vec::with_capacity(n);
            Iter::new(data, layout).for_each(|x| v.push(*x));
            v
        };
        assert_eq!(
            from_fold, expected,
            "Iter::fold order wrong at {}",
            case.label
        );

        let from_old: Vec<f32> = input.iter().copied().collect();
        assert_eq!(from_old, expected, "Iter order wrong at {}", case.label);

        let mut out = vec![0.0f32; n];

        group.bench_function(BenchmarkId::new("best", &case.label), |bencher| {
            bencher.iter(|| {
                copy_kernel(black_box(data), layout, &mut out);
                black_box(&out);
            });
        });

        group.bench_function(BenchmarkId::new("elem", &case.label), |bencher| {
            bencher.iter(|| {
                let it = black_box(ChunkIter::new(data, layout));

                let mut current: usize = 0;

                for chunk in it {
                    match chunk {
                        ChunkKind::Contiguous { data, start, times } => {
                            out[current..current + times]
                                .copy_from_slice(&data[start..start + times]);
                            current += times;
                        }
                        ChunkKind::Strided {
                            data,
                            start,
                            times,
                            step,
                        } => {
                            let mut pos = start;
                            for _ in 0..times {
                                out[current] = data[pos];
                                pos = pos.wrapping_add_signed(step);

                                current += 1;
                            }
                        }
                    }
                }

                black_box(&out);
            });
        });

        group.bench_function(BenchmarkId::new("iter", &case.label), |bencher| {
            bencher.iter(|| {
                let it = black_box(Iter::new(data, layout));

                for (o, x) in out.iter_mut().zip(it) {
                    *o = *x;
                }
                black_box(&out);
            });
        });

        group.bench_function(BenchmarkId::new("fold", &case.label), |bencher| {
            bencher.iter(|| {
                let it = black_box(Iter::new(data, layout));

                let mut i = 0usize;
                it.for_each(|x| {
                    out[i] = *x;
                    i += 1;
                });
                black_box(&out);
            });
        });

        group.bench_function(BenchmarkId::new("old", &case.label), |bencher| {
            bencher.iter(|| {
                for (o, x) in out.iter_mut().zip(black_box(input).iter()) {
                    *o = *x;
                }
                black_box(&out);
            });
        });
    }

    group.finish();
}

/// Same sweep as [`iteration`], but the op is `* 2.0` instead of a copy. Multiply
/// has no `copy_from_slice` shortcut, so `best` must go element-wise too - this
/// isolates `fold`'s per-element cost against a fair element-wise best,
/// without a `memcpy` handing `best` a free win.
fn multiplication(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("multiplication");
    group.plot_config(plot_config);

    let cfg = standard_config();

    for case in common::fill_cases(&cfg, scalar_add::<CpuPure>) {
        group.throughput(case.throughput);

        let input = &case.inputs[0];
        let data = input.data();
        let layout = input.layout();
        let n = case.skeleton.len();

        // Multiply by a power of two is exact for f32, so this compares
        // byte-for-byte against the hand-rolled kernel.
        let mut expected = vec![0.0f32; n];
        mul_kernel(data, layout, &mut expected);
        let from_fold: Vec<f32> = {
            let mut v = vec![0.0f32; n];
            Iter::new(data, layout)
                .enumerate()
                .for_each(|(i, x)| v[i] = *x * 2.0);
            v
        };
        assert_eq!(from_fold, expected, "fold *2.0 wrong at {}", case.label);

        let mut out = vec![0.0f32; n];

        group.bench_function(BenchmarkId::new("best", &case.label), |bencher| {
            bencher.iter(|| {
                mul_kernel(black_box(data), black_box(layout), &mut out);
                black_box(&out);
            });
        });

        group.bench_function(BenchmarkId::new("elem", &case.label), |bencher| {
            bencher.iter(|| {
                let it = black_box(ChunkIter::new(data, layout));

                let mut current: usize = 0;

                for chunk in it {
                    match chunk {
                        ChunkKind::Contiguous { data, start, times } => {
                            data[start..start + times]
                                .iter()
                                .enumerate()
                                .for_each(|(i, x)| out[current + i] = *x * 2.0);
                            current += times;
                        }
                        ChunkKind::Strided {
                            data,
                            start,
                            times,
                            step,
                        } => {
                            let mut pos = start;
                            for _ in 0..times {
                                out[current] = data[pos] * 2.0;
                                pos = pos.wrapping_add_signed(step);

                                current += 1;
                            }
                        }
                    }
                }

                black_box(&out);
            });
        });

        group.bench_function(BenchmarkId::new("iter", &case.label), |bencher| {
            bencher.iter(|| {
                let it = black_box(Iter::new(data, layout));

                for (o, x) in out.iter_mut().zip(it) {
                    *o = *x * 2.0;
                }
                black_box(&out);
            });
        });

        group.bench_function(BenchmarkId::new("fold", &case.label), |bencher| {
            bencher.iter(|| {
                let it = black_box(Iter::new(data, layout));

                it.enumerate().for_each(|(i, x)| {
                    out[i] = *x * 2.0;
                });
                black_box(&out);
            });
        });
    }

    group.finish();
}

fn broadcasting(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("broadcasting");
    group.plot_config(plot_config);

    let sizes = &[SizeSpec::L1, SizeSpec::L2, SizeSpec::L3, SizeSpec::Dram];

    for &inner in &[64usize, 1024usize] {
        let cfg = FillConfig::new(1)
            .shape(ShapePolicy::Rows { inner })
            .variants(&[Variant::BcastInner, Variant::BcastOuter])
            .sizes(sizes)
            .variant_sizes(sizes);

        for case in common::fill_cases(&cfg, scalar_add::<CpuPure>) {
            group.throughput(case.throughput);

            let input = &case.inputs[0];
            let data = input.data();
            let layout = input.layout();
            let n = case.skeleton.len();

            let mut expected = vec![0.0f32; n];
            mul_kernel(data, layout, &mut expected);

            let from_fold: Vec<f32> = {
                let mut v = vec![0.0f32; n];
                Iter::new(data, layout)
                    .enumerate()
                    .for_each(|(i, x)| v[i] = *x * 2.0);
                v
            };
            assert_eq!(
                from_fold, expected,
                "broadcast *2.0 wrong at n{inner} {}",
                case.label
            );

            let mut out = vec![0.0f32; n];
            let tag = format!("n{inner}");

            group.bench_function(BenchmarkId::new(format!("best_{tag}"), &case.label), |b| {
                b.iter(|| {
                    mul_kernel(black_box(data), black_box(layout), &mut out);
                    black_box(&out);
                });
            });

            group.bench_function(BenchmarkId::new(format!("iter_{tag}"), &case.label), |b| {
                b.iter(|| {
                    let it = black_box(Iter::new(data, layout));
                    for (o, x) in out.iter_mut().zip(it) {
                        *o = *x * 2.0;
                    }
                    black_box(&out);
                });
            });

            group.bench_function(BenchmarkId::new(format!("fold_{tag}"), &case.label), |b| {
                b.iter(|| {
                    let it = black_box(Iter::new(data, layout));
                    let mut i = 0usize;
                    it.for_each(|x| {
                        out[i] = *x * 2.0;
                        i += 1;
                    });
                    black_box(&out);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, iteration, multiplication, broadcasting);
criterion_main!(benches);
