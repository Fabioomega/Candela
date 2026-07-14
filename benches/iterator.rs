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
use common::{FillConfig, SizeSpec, Variant};

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
    let mut rank: usize = layout.shape().len();
    let mut shape = [0usize; MAX_DIMS];
    let mut adj_stride = [0i32; MAX_DIMS];

    let mut i = rank - 1;

    shape[0] = layout.shape()[0];
    adj_stride[0] = layout.adj_stride()[0];
    while 0 < i {
        if adj_stride[i] == layout.adj_stride()[i - 1] {
            shape[i - 1] = shape[i] * layout.shape()[i - 1];
            rank -= 1;
        } else {
            shape[i - 1] = layout.shape()[i - 1];
            adj_stride[i - 1] = layout.adj_stride()[i - 1];
        }

        i -= 1;
    }

    (rank, shape, adj_stride)
}

impl<'a, T> ChunkIter<'a, T> {
    pub fn new(data: &'a [T], layout: &Layout) -> Self {
        // TODO: Add support for dimensions higher than 8 (use smallvec or something similar)
        debug_assert!(
            layout.shape().len() <= 8,
            "dimensions higher than 8 are not support for iteration!"
        );

        let (mut rank, mut shape, mut adj_stride) = simplify_layout(layout);

        if rank == 1 {
            shape[1] = shape[0];
            shape[0] = 1;

            adj_stride[1] = adj_stride[0];

            rank += 1;
        }

        let last = layout.shape().len() - 1;
        let step = layout.adj_stride()[last] as isize * (layout.shape()[last] - 1) as isize;

        let left_over: usize = layout.shape()[0..last].iter().product();

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

    pub fn len(&self) -> usize {
        self.rank
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
        for dim in (1..last_counter).rev() {
            if self.counter[dim] == self.shape[dim] {
                self.counter[dim] = 0;
                self.counter[dim - 1] += 1;

                step_dim = dim;
                continue;
            }
            break;
        }

        self.pos = unsafe {
            self.pos
                .checked_add_signed(self.adj_stride[step_dim] as isize + self.step)
                .unwrap_unchecked()
        };

        Some(chunk)
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

struct Iter<'a, T> {
    it: ChunkIter<'a, T>,
    data: &'a [T],
    pos: usize,
    stride: isize,
    times: usize,
}

impl<'a, T> Iter<'a, T> {
    fn new(data: &'a [T], layout: &Layout) -> Self {
        Self {
            it: ChunkIter::new(data, layout),
            data,
            pos: 0,
            stride: 0,
            times: 0,
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.times == 0 {
            match self.it.next()? {
                ChunkKind::Contiguous { data, start, times } => {
                    self.data = data;
                    self.pos = start;
                    self.stride = 1;
                    self.times = times;
                }
                ChunkKind::Strided {
                    data,
                    start,
                    times,
                    step,
                } => {
                    self.data = data;
                    self.pos = start;
                    self.stride = step;
                    self.times = times;
                }
            }
        }

        let output = unsafe { self.data.get_unchecked(self.pos) };
        self.pos = unsafe { self.pos.checked_add_signed(self.stride).unwrap_unchecked() };
        self.times -= 1;

        Some(output)
    }

    #[inline]
    fn fold<Acc, F>(mut self, init: Acc, mut f: F) -> Acc
    where
        F: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut acc = init;

        // Clear starting run (if it exists)
        while self.times > 0 {
            acc = f(acc, unsafe { self.data.get_unchecked(self.pos) });
            self.pos = self.pos.wrapping_add_signed(self.stride);
            self.times -= 1;
        }

        while let Some(chunk) = self.it.next() {
            match chunk {
                ChunkKind::Contiguous { data, start, times } => {
                    for x in &data[start..start + times] {
                        acc = f(acc, x);
                    }
                }
                ChunkKind::Strided {
                    data,
                    start,
                    times,
                    step,
                } => {
                    for i in successors(Some(start), |&i| i.checked_add_signed(step)).take(times) {
                        acc = f(acc, unsafe { data.get_unchecked(i) });
                    }
                }
            }
        }

        acc
    }
}

// impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

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
        for (o, x) in out.iter_mut().zip(a.iter().step_by(step)) {
            *o = *x;
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
        panic!("no baseline copy kernel for layout {layout:?}");
    }
}

fn iteration(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let mut group = c.benchmark_group("iteration");
    group.plot_config(plot_config);

    let cfg = FillConfig::new(1)
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
        ]);

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

        let from_old: Vec<f32> = input.iter().copied().collect();
        assert_eq!(from_old, expected, "Iter order wrong at {}", case.label);

        let mut out = vec![0.0f32; n];

        group.bench_function(BenchmarkId::new("base", &case.label), |bencher| {
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
                            for i in
                                successors(Some(start), |&i| i.checked_add_signed(step)).take(times)
                            {
                                out[current] = data[i];
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

criterion_group!(benches, iteration);
criterion_main!(benches);
