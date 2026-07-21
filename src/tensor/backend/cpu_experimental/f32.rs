use std::iter::zip;

use crate::Dimension;
use crate::tensor::backend::common::{clone_to_buffer, normalize_axis};
use crate::tensor::backend::common_kernels::{
    compute_max_axis_tensor, compute_max_tensor, compute_mean_axis_tensor, compute_mean_tensor,
    compute_sum_axis_tensor, compute_sum_tensor,
};
use crate::tensor::backend::cpu_experimental::kernels::{
    CommonBLASOps, compute_elementwise_tensor_tensor, compute_elementwise_tensor_tensor_inplace,
    compute_matmul_sum, compute_scalar, compute_scalar_inplace,
};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::{OpKind, OpKindScalar, Sign};
use crate::tensor::storage::{Storage, TensorData};
use crate::tensor::walker::{map_chunk, map_chunk_inplace};

use wide::f32x8;

const BLAS: CommonBLASOps<f32> = CommonBLASOps {
    fma: |a, b, c| a.mul_add(b, c),
    exp: |a| a.exp(),
    ln: |a| a.ln(),
    log2: |a| a.log2(),
    max: |a, b| a.max(b),
    tanh: |a| a.tanh(),
    matmul: matrixmultiply::sgemm,
};

const LANE_WIDTH: usize = 8;

#[inline]
fn unary_simd<F: Fn(f32x8) -> f32x8, U: Fn(f32) -> f32>(
    src: &[f32],
    dst: &mut [f32],
    f_simd: F,
    f: U,
) {
    let (in_chunks, in_remainder) = src.as_chunks::<LANE_WIDTH>();
    let (out_chunks, out_remainder) = dst.as_chunks_mut::<LANE_WIDTH>();

    for (chunk_in, chunk_out) in in_chunks.iter().zip(out_chunks) {
        let v = wide::f32x8::from(*chunk_in);

        let result = f_simd(v);

        chunk_out.copy_from_slice(&result.to_array());
    }

    for (x, y) in in_remainder.iter().zip(out_remainder) {
        *y = f(*x);
    }
}

#[inline]
fn unary_simd_inplace<F: Fn(f32x8) -> f32x8, U: Fn(f32) -> f32>(out: &mut [f32], f_simd: F, f: U) {
    let (out_chunks, out_remainder) = out.as_chunks_mut::<LANE_WIDTH>();

    for chunk_out in out_chunks.iter_mut() {
        let v = wide::f32x8::from(*chunk_out);

        let result = f_simd(v);

        chunk_out.copy_from_slice(&result.to_array());
    }

    for x in out_remainder.iter_mut() {
        *x = f(*x);
    }
}

/// Applies one scalar op reading `src`, writing `dst` (same length). `AxBy` is a
/// plain multiply-add (`s * a + b`, not `mul_add`): on a target without the `fma`
/// feature `mul_add` lowers to a libm `fmaf` call per element, which blocks
/// vectorization, whereas separate multiply/add lower to `mulps`/`addps`.
#[inline]
fn apply_op_out(op: &OpKindScalar<f32>, src: &[f32], dst: &mut [f32]) {
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
        OpKindScalar::Inv => {
            for (i, o) in zip(src, dst) {
                *o = 1.0 / i;
            }
        }
        OpKindScalar::ReLU => {
            unary_simd(src, dst, |x| x.fast_max(f32x8::splat(0.0)), |x| x.max(0.0));
        }
        OpKindScalar::Tanh => {
            unary_simd(src, dst, |x| x.tanh(), |x| x.tanh());
        }
    }
}

/// Applies one scalar op in place over `dst`.
#[inline]
fn apply_op_inplace(op: &OpKindScalar<f32>, dst: &mut [f32]) {
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
        OpKindScalar::Inv => {
            unary_simd_inplace(dst, |x| f32x8::splat(1.0) / x, |x| 1.0 / x);
        }
        OpKindScalar::ReLU => {
            unary_simd_inplace(dst, |x| x.fast_max(f32x8::splat(0.0)), |x| x.max(0.0));
        }
        OpKindScalar::Tanh => {
            unary_simd_inplace(dst, |x| x.tanh(), |x| x.tanh());
        }
    }
}

/// Applies a fused scalar chain over one contiguous run, `src -> dst` (equal
/// length). The first op reads `src`; every later op folds in place over `dst`,
/// so the ops compose. Each op is a single fixed-shape loop the compiler can
/// vectorize.
#[inline]
fn apply_chain(ops: &[OpKindScalar<f32>], src: &[f32], dst: &mut [f32]) {
    // SAFETY: A ops always has at least a single operation otherwise it
    // cannot exist.
    let (first, rest) = unsafe { ops.split_first().unwrap_unchecked() };

    apply_op_out(first, src, dst);
    for op in rest {
        apply_op_inplace(op, dst);
    }
}

/// Applies a whole chain to one scalar - the per-element fallback for gather
/// layouts, which cannot vectorize. `AxBy` is `temp * a + b`, not `mul_add`, to
/// avoid a libm `fmaf` call on targets without the `fma` feature.
fn compute_inline(ops: &[OpKindScalar<f32>], el: f32) -> f32 {
    let mut temp = el;

    for op in ops {
        match op {
            OpKindScalar::AxBy(a, b) => temp = temp * *a + *b,
            OpKindScalar::Exp => temp = temp.exp(),
            OpKindScalar::Ln => temp = temp.ln(),
            OpKindScalar::Log2 => temp = temp.log2(),
            OpKindScalar::Inv => temp = 1.0 / temp,
            OpKindScalar::ReLU => temp = temp.max(0.0),
            OpKindScalar::Tanh => temp = temp.tanh(),
        }
    }

    temp
}

const TILE_SIZE: usize = 2048;

fn for_each_scalar_inplace(dst: &mut [f32], layout: &Layout, ops: &[OpKindScalar<f32>]) {
    let ch = |dst: &mut [f32]| {
        let (o_tile, o_remainder) = dst.as_chunks_mut::<TILE_SIZE>();

        for tile in o_tile {
            for op in ops {
                apply_op_inplace(op, tile);
            }
        }

        for op in ops {
            apply_op_inplace(op, o_remainder);
        }
    };

    let elem = |src| compute_inline(ops, src);

    map_chunk_inplace(dst, layout, ch, elem);
}

fn for_each_scalar(src: &[f32], dst: &mut [f32], layout: &Layout, ops: &[OpKindScalar<f32>]) {
    let ch = |src: &[f32], dst: &mut [f32]| {
        let (i_tile, i_remainder) = src.as_chunks::<TILE_SIZE>();
        let (o_tile, o_remainder) = dst.as_chunks_mut::<TILE_SIZE>();

        for (i, o) in zip(i_tile, o_tile) {
            apply_chain(ops, i, o);
        }

        apply_chain(ops, i_remainder, o_remainder);
    };

    let elem = |src| compute_inline(ops, src);

    map_chunk(src, layout, dst, ch, elem);
}

///////////////////////////////////////////////////////////////

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        skip(inputs, _output_buffer, output_layout),
        fields(op = op.as_str(), out_len = output_layout.len())
    )
)]
pub(crate) fn compute_op(
    op: &OpKind<f32>,
    mut _output_buffer: Vec<f32>,
    output_layout: &Layout,
    inputs: &[TensorData<f32>],
) -> TensorData<f32> {
    let output_buffer = &mut _output_buffer;

    let layout = output_layout.clone();

    match op {
        OpKind::ScalarOp(s) => compute_scalar(
            std::slice::from_ref(s),
            inputs,
            output_buffer,
            for_each_scalar,
        ),
        OpKind::FusedScalar(ss) => compute_scalar(ss, inputs, output_buffer, for_each_scalar),
        OpKind::AsContiguous => clone_to_buffer(&inputs[0], output_buffer),
        OpKind::Add => compute_elementwise_tensor_tensor(inputs, output_buffer, |a, b| a + b),
        OpKind::Sub => compute_elementwise_tensor_tensor(inputs, output_buffer, |a, b| a - b),
        OpKind::Mul => compute_elementwise_tensor_tensor(inputs, output_buffer, |a, b| a * b),
        OpKind::Div => compute_elementwise_tensor_tensor(inputs, output_buffer, |a, b| a / b),
        OpKind::MatMul(a) => compute_matmul_sum(inputs, *a, 0.0, output_buffer, false, BLAS),
        OpKind::MatMulSum(a, b, sign) => {
            let beta = if *sign == Sign::Minus { -*b } else { *b };
            compute_matmul_sum(inputs, *a, beta, output_buffer, true, BLAS)
        }
        OpKind::Slice
        | OpKind::View
        | OpKind::TransposeAxes
        | OpKind::Broadcast
        | OpKind::Transpose
        | OpKind::NoOp => {
            unreachable!("a reference should never appear here");
        }
        OpKind::Sum => compute_sum_tensor(inputs, output_buffer),
        OpKind::SumAxis(axis, _) => {
            let axis = normalize_axis(*axis, inputs[0].shape().len());

            compute_sum_axis_tensor(inputs, axis, output_buffer)
        }
        OpKind::Max => compute_max_tensor(inputs, output_buffer, BLAS.max),
        OpKind::MaxAxis(axis, _) => {
            let axis = normalize_axis(*axis, inputs[0].shape().len());

            compute_max_axis_tensor(inputs, axis, output_buffer, BLAS.max)
        }
        OpKind::Mean => compute_mean_tensor(inputs, output_buffer, |a, b| a / b as f32),
        OpKind::MeanAxis(axis, _) => {
            let axis = normalize_axis(*axis, inputs[0].shape().len());

            compute_mean_axis_tensor(inputs, axis, output_buffer, |a, b| a / b as f32)
        }
    };

    TensorData::new(Storage::from_vec(_output_buffer), layout)
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
    output_layout: &Layout,
    mut inputs: Vec<TensorData<f32>>,
    output_idx: usize,
) -> TensorData<f32> {
    match op {
        OpKind::ScalarOp(s) => compute_scalar_inplace(
            std::slice::from_ref(s),
            inputs,
            output_layout,
            for_each_scalar_inplace,
        ),
        OpKind::FusedScalar(ss) => {
            compute_scalar_inplace(ss, inputs, output_layout, for_each_scalar_inplace)
        }
        OpKind::Add => {
            let b = inputs.pop().unwrap();
            let a = inputs.pop().unwrap();
            if output_idx == 0 {
                compute_elementwise_tensor_tensor_inplace(a, b, |a, b| a + b)
            } else {
                compute_elementwise_tensor_tensor_inplace(b, a, |a, b| a + b)
            }
        }
        OpKind::Sub => {
            let b = inputs.pop().unwrap();
            let a = inputs.pop().unwrap();
            if output_idx == 0 {
                compute_elementwise_tensor_tensor_inplace(a, b, |a, b| a - b)
            } else {
                compute_elementwise_tensor_tensor_inplace(b, a, |b, a| a - b)
            }
        }
        OpKind::Mul => {
            let b = inputs.pop().unwrap();
            let a = inputs.pop().unwrap();
            if output_idx == 0 {
                compute_elementwise_tensor_tensor_inplace(a, b, |a, b| a * b)
            } else {
                compute_elementwise_tensor_tensor_inplace(b, a, |a, b| a * b)
            }
        }
        OpKind::Div => {
            let b = inputs.pop().unwrap();
            let a = inputs.pop().unwrap();
            if output_idx == 0 {
                compute_elementwise_tensor_tensor_inplace(a, b, |a, b| a / b)
            } else {
                compute_elementwise_tensor_tensor_inplace(b, a, |b, a| a / b)
            }
        }
        OpKind::Slice
        | OpKind::View
        | OpKind::TransposeAxes
        | OpKind::Broadcast
        | OpKind::Transpose => {
            let input = unsafe { inputs.pop().unwrap_unchecked() };
            let offset = input.offset();

            input.into_layout(
                output_layout
                    .clone()
                    .with_offset(offset + output_layout.offset()),
            )
        }
        OpKind::NoOp | OpKind::AsContiguous => unsafe { inputs.pop().unwrap_unchecked() },
        _ => todo!("not implemented {}", op.as_str()),
    }
}
