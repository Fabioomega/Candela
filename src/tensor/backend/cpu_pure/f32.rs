use core::f32;
use std::iter::zip;

use wide::f32x16;

use crate::Dimension;
use crate::tensor::backend::common::{clone_to_buffer, normalize_axis};
use crate::tensor::backend::common_kernels::{
    compute_elementwise, compute_elementwise_inplace, compute_mean, compute_mean_axis,
    compute_reduction, compute_reduction_axis, compute_scalar, compute_scalar_inplace,
};

use crate::tensor::backend::cpu_pure::kernels::compute_matmul_sum;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::{OpKind, OpKindScalar, Sign};
use crate::tensor::storage::TensorData;

const LANE_WIDTH: usize = 16;
const TILE_SIZE: usize = 2048;

#[inline]
fn unary_simd<F: Fn(f32x16) -> f32x16, U: Fn(f32) -> f32>(
    src: &[f32],
    dst: &mut [f32],
    f_simd: F,
    f: U,
) {
    let (in_chunks, in_remainder) = src.as_chunks::<LANE_WIDTH>();
    let (out_chunks, out_remainder) = dst.as_chunks_mut::<LANE_WIDTH>();

    for (chunk_in, chunk_out) in in_chunks.iter().zip(out_chunks) {
        let v = wide::f32x16::from(*chunk_in);

        let result = f_simd(v);

        chunk_out.copy_from_slice(&result.to_array());
    }

    for (x, y) in in_remainder.iter().zip(out_remainder) {
        *y = f(*x);
    }
}

#[inline]
fn unary_simd_inplace<F: Fn(f32x16) -> f32x16, U: Fn(f32) -> f32>(
    out: &mut [f32],
    f_simd: F,
    f: U,
) {
    let (out_chunks, out_remainder) = out.as_chunks_mut::<LANE_WIDTH>();

    for chunk_out in out_chunks.iter_mut() {
        let v = wide::f32x16::from(*chunk_out);

        let result = f_simd(v);

        chunk_out.copy_from_slice(&result.to_array());
    }

    for x in out_remainder.iter_mut() {
        *x = f(*x);
    }
}

#[inline]
fn compute_out(op: &OpKindScalar<f32>, src: &[f32], dst: &mut [f32]) {
    match op {
        OpKindScalar::AxBy(a, b) => {
            for (i, o) in zip(src, dst) {
                *o = *a * *i + *b;
            }
        }
        OpKindScalar::Exp => {
            unary_simd(src, dst, |x| x.exp(), |x| x.exp());
        }
        OpKindScalar::Ln => {
            unary_simd(src, dst, |x| x.ln(), |x| x.ln());
        }
        OpKindScalar::Log2 => {
            unary_simd(src, dst, |x| x.log2(), |x| x.log2());
        }
        OpKindScalar::Recip => {
            for (i, o) in zip(src, dst) {
                *o = 1.0 / i;
            }
        }
        OpKindScalar::ReLU => {
            unary_simd(
                src,
                dst,
                |x| x.fast_max(f32x16::splat(0.0)),
                |x| if x > 0.0 { x } else { 0.0 },
            );
        }
        OpKindScalar::Tanh => {
            unary_simd(src, dst, |x| x.tanh(), |x| x.tanh());
        }
    }
}

#[inline]
fn compute_inplace(op: &OpKindScalar<f32>, dst: &mut [f32]) {
    match op {
        OpKindScalar::AxBy(a, b) => {
            for o in dst {
                *o = *a * *o + *b;
            }
        }
        OpKindScalar::Exp => {
            unary_simd_inplace(dst, |x| x.exp(), |x| x.exp());
        }
        OpKindScalar::Ln => {
            unary_simd_inplace(dst, |x| x.ln(), |x| x.ln());
        }
        OpKindScalar::Log2 => {
            unary_simd_inplace(dst, |x| x.log2(), |x| x.log2());
        }
        OpKindScalar::Recip => {
            for o in dst {
                *o = 1.0 / *o;
            }
        }
        OpKindScalar::ReLU => {
            unary_simd_inplace(
                dst,
                |x| x.fast_max(f32x16::splat(0.0)),
                |x| if x > 0.0 { x } else { 0.0 },
            );
        }
        OpKindScalar::Tanh => {
            unary_simd_inplace(dst, |x| x.tanh(), |x| x.tanh());
        }
    }
}

fn compute_element(ops: &[OpKindScalar<f32>], el: f32) -> f32 {
    let mut temp = el;

    for op in ops {
        match op {
            OpKindScalar::AxBy(a, b) => temp = temp * *a + *b,
            OpKindScalar::Exp => temp = temp.exp(),
            OpKindScalar::Ln => temp = temp.ln(),
            OpKindScalar::Log2 => temp = temp.log2(),
            OpKindScalar::Recip => temp = 1.0 / temp,
            OpKindScalar::ReLU => temp = if temp > 0.0 { temp } else { 0.0 },
            OpKindScalar::Tanh => temp = temp.tanh(),
        }
    }

    temp
}

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        skip(inputs, output_buffer, output_layout),
        fields(op = op.as_str(), out_len = output_layout.len())
    )
)]
pub(crate) fn compute_op(
    op: &OpKind<f32>,
    output_buffer: &mut [f32],
    output_layout: &Layout,
    inputs: &[TensorData<f32>],
) {
    match op {
        OpKind::ScalarOp(s) => compute_scalar::<TILE_SIZE, f32, _, _, _>(
            inputs[0].data(),
            output_buffer,
            inputs[0].layout(),
            std::slice::from_ref(s),
            compute_out,
            compute_element,
            compute_inplace,
        ),
        OpKind::FusedScalar(ss) => compute_scalar::<TILE_SIZE, f32, _, _, _>(
            inputs[0].data(),
            output_buffer,
            inputs[0].layout(),
            ss,
            compute_out,
            compute_element,
            compute_inplace,
        ),
        OpKind::AsContiguous => clone_to_buffer(&inputs[0], output_buffer),
        OpKind::Add => compute_elementwise(inputs, output_buffer, |a, b| a + b),
        OpKind::Sub => compute_elementwise(inputs, output_buffer, |a, b| a - b),
        OpKind::Mul => compute_elementwise(inputs, output_buffer, |a, b| a * b),
        OpKind::Div => compute_elementwise(inputs, output_buffer, |a, b| a / b),
        OpKind::MatMul(a) => {
            compute_matmul_sum(inputs, *a, 0.0, output_buffer, false, matrixmultiply::sgemm)
        }
        OpKind::MatMulSum(a, b, sign) => {
            let beta = if *sign == Sign::Minus { -*b } else { *b };
            compute_matmul_sum(inputs, *a, beta, output_buffer, true, matrixmultiply::sgemm)
        }
        OpKind::Sum => compute_reduction(inputs, output_buffer, 0.0, |x, y| x + y),
        OpKind::SumAxis(axis, _) => {
            let axis = normalize_axis(*axis, inputs[0].shape().len());

            compute_reduction_axis(
                inputs,
                axis,
                output_buffer,
                inputs[0].layout(),
                0.0,
                |x, y| x + y,
            );
        }
        OpKind::Max => compute_reduction(inputs, output_buffer, f32::NEG_INFINITY, |x, y| {
            if x > y { x } else { y }
        }),
        OpKind::MaxAxis(axis, _) => {
            let axis = normalize_axis(*axis, inputs[0].shape().len());

            compute_reduction_axis(
                inputs,
                axis,
                output_buffer,
                output_layout,
                f32::NEG_INFINITY,
                |x, y| if x > y { x } else { y },
            );
        }
        OpKind::Mean => compute_mean(inputs, output_buffer),
        OpKind::MeanAxis(axis, _) => {
            let axis = normalize_axis(*axis, inputs[0].shape().len());

            compute_mean_axis(inputs, axis, output_buffer, output_layout);
        }
        OpKind::Slice
        | OpKind::View
        | OpKind::TransposeAxes
        | OpKind::Broadcast
        | OpKind::Transpose
        | OpKind::NoOp => {
            unreachable!("a reference should never appear here");
        }
    };
}

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        skip(inputs, output_layout),
        fields(op = op.as_str(), out_len = output_layout.len())
    )
)]
pub(crate) fn compute_op_inplace(
    op: &OpKind<f32>,
    output_buffer: &mut [f32],
    output_layout: &Layout,
    inputs: &[TensorData<f32>],
    output_idx: usize,
) {
    match op {
        OpKind::ScalarOp(s) => compute_scalar_inplace::<TILE_SIZE, f32, _, _>(
            std::slice::from_ref(s),
            output_buffer,
            output_layout,
            compute_element,
            compute_inplace,
        ),
        OpKind::FusedScalar(ss) => compute_scalar_inplace::<TILE_SIZE, f32, _, _>(
            ss,
            output_buffer,
            output_layout,
            compute_element,
            compute_inplace,
        ),
        OpKind::Add => compute_elementwise_inplace(inputs, output_buffer, output_idx, |x, y| x + y),
        OpKind::Sub => compute_elementwise_inplace(inputs, output_buffer, output_idx, |x, y| x - y),
        OpKind::Mul => compute_elementwise_inplace(inputs, output_buffer, output_idx, |x, y| x * y),
        OpKind::Div => compute_elementwise_inplace(inputs, output_buffer, output_idx, |x, y| x / y),
        OpKind::Slice
        | OpKind::View
        | OpKind::TransposeAxes
        | OpKind::Broadcast
        | OpKind::Transpose
        | OpKind::NoOp
        | OpKind::AsContiguous => {
            unreachable!("a reference should never appear here");
        }
        _ => todo!("not implemented {}", op.as_str()),
    }
}
