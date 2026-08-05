use candela::backend::{Backend, ComputeFor};
use candela::skeleton::{Skeleton, SkeletonSlot};
use candela::{Layout, Tensor};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use rand::distr::{Distribution, StandardUniform};
use std::hint::black_box;

use crate::common::{ShapeCase, shape_cases};
use crate::ops::{BenchScalar, Tier, bench_cells, group};

const SQUARE: [usize; 3] = [128, 512, 2048];

fn flops(m: usize, n: usize, k: usize, batch: usize) -> Throughput {
    Throughput::Elements((2 * m * n * k * batch) as u64)
}

fn build_matmul<T, B>(layouts: &[Layout]) -> Skeleton<T, B>
where
    T: BenchScalar + ComputeFor<B>,
    B: Backend,
{
    let a: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[0].clone());
    let b: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[1].clone());
    a.matmul(&b).unwrap().into_skeleton(&[a, b]).unwrap()
}

/// `a @ b + c`, which fusion rewrites into one `MatMulSum` node.
fn build_matmul_sum<T, B>(layouts: &[Layout]) -> Skeleton<T, B>
where
    T: BenchScalar + ComputeFor<B>,
    B: Backend,
{
    let a: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[0].clone());
    let b: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[1].clone());
    let c: SkeletonSlot<T, B> = SkeletonSlot::new(layouts[2].clone());
    (a.matmul(&b).unwrap() + &c)
        .into_skeleton(&[a, b, c])
        .unwrap()
}

fn square_cases() -> Vec<ShapeCase> {
    SQUARE
        .iter()
        .map(|&n| ShapeCase {
            label: format!("{n}x{n}x{n}"),
            layouts: vec![Layout::new([n, n]), Layout::new([n, n])],
            throughput: flops(n, n, n, 1),
        })
        .collect()
}

fn batched_cases() -> Vec<ShapeCase> {
    [(32usize, 128usize), (8, 512)]
        .iter()
        .map(|&(batch, n)| ShapeCase {
            label: format!("{batch}x{n}x{n}x{n}"),
            layouts: vec![Layout::new([batch, n, n]), Layout::new([batch, n, n])],
            throughput: flops(n, n, n, batch),
        })
        .collect()
}

fn transposed_cases() -> Vec<ShapeCase> {
    SQUARE
        .iter()
        .map(|&n| ShapeCase {
            label: format!("{n}x{n}x{n}"),
            layouts: vec![
                Layout::new([n, n]),
                Layout::from_strided(&[n, n], &[1, n as i32], 0),
            ],
            throughput: flops(n, n, n, 1),
        })
        .collect()
}

fn matmul_sum_cases() -> Vec<ShapeCase> {
    SQUARE
        .iter()
        .map(|&n| ShapeCase {
            label: format!("{n}x{n}x{n}"),
            layouts: vec![
                Layout::new([n, n]),
                Layout::new([n, n]),
                Layout::new([n, n]),
            ],
            throughput: flops(n, n, n, 1),
        })
        .collect()
}

fn run_shapes<T, B, F>(
    c: &mut Criterion,
    name: &str,
    cell: &str,
    specs: &[ShapeCase],
    builder: F,
    report: bool,
) where
    T: BenchScalar + ComputeFor<B>,
    StandardUniform: Distribution<T>,
    B: Backend,
    F: Fn(&[Layout]) -> Skeleton<T, B>,
{
    let mut g = group(c, name, Tier::Heavy);
    for case in shape_cases::<T, B, F>(specs, builder) {
        if report {
            let m = case.skeleton.memory_report();
            eprintln!(
                "[{name}] {cell} {}: buffers={:?} peak={}B",
                case.label, m.allocated_buffers_size, m.peak_memory_usage
            );
        }

        g.throughput(case.throughput.clone());
        let refs: Vec<&Tensor<T, B>> = case.inputs.iter().collect();
        g.bench_function(BenchmarkId::new(cell, &case.label), |bencher| {
            bencher.iter(|| case.skeleton.run(black_box(&refs)).unwrap());
        });
    }
    g.finish();
}

fn run<T, B>(c: &mut Criterion, cell: &str)
where
    T: BenchScalar + ComputeFor<B>,
    StandardUniform: Distribution<T>,
    B: Backend,
{
    run_shapes::<T, B, _>(
        c,
        "matmul/square",
        cell,
        &square_cases(),
        build_matmul::<T, B>,
        false,
    );
    run_shapes::<T, B, _>(
        c,
        "matmul/batched",
        cell,
        &batched_cases(),
        build_matmul::<T, B>,
        false,
    );
    run_shapes::<T, B, _>(
        c,
        "matmul/transposed_b",
        cell,
        &transposed_cases(),
        build_matmul::<T, B>,
        true,
    );
    run_shapes::<T, B, _>(
        c,
        "matmul/matmul_sum",
        cell,
        &matmul_sum_cases(),
        build_matmul_sum::<T, B>,
        false,
    );
}

fn matmul(c: &mut Criterion) {
    bench_cells!(c, run);
}

criterion_group!(benches, matmul);
