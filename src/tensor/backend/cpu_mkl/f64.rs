use crate::Dimension;
use crate::tensor::backend::common::{clone_to_buffer, normalize_axis};
use crate::tensor::backend::common_kernels::{
    compute_elementwise, compute_elementwise_inplace, compute_mean, compute_mean_axis,
    compute_reduction, compute_reduction_axis, compute_scalar, compute_scalar_inplace,
};
use crate::tensor::backend::cpu_mkl::kernels::compute_matmul_sum;
use crate::tensor::backend::cpu_mkl::mkl_extension::cblas_dgemm_batch_strided;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::{OpKind, OpKindScalar, Sign};
use crate::tensor::storage::TensorData;
use intel_mkl_sys::{vdExp, vdLn, vdLog2, vdTanh};
use wide::f64x8;

const LANE_WIDTH: usize = 8;
const TILE_SIZE: usize = 2048;

#[inline]
fn unary_simd<F: Fn(f64x8) -> f64x8, U: Fn(f64) -> f64>(
    src: &[f64],
    dst: &mut [f64],
    f_simd: F,
    f: U,
) {
    let (in_chunks, in_remainder) = src.as_chunks::<LANE_WIDTH>();
    let (out_chunks, out_remainder) = dst.as_chunks_mut::<LANE_WIDTH>();

    for (chunk_in, chunk_out) in in_chunks.iter().zip(out_chunks) {
        let v = wide::f64x8::from(*chunk_in);

        let result = f_simd(v);

        chunk_out.copy_from_slice(&result.to_array());
    }

    for (x, y) in in_remainder.iter().zip(out_remainder) {
        *y = f(*x);
    }
}

#[inline]
fn unary_simd_inplace<F: Fn(f64x8) -> f64x8, U: Fn(f64) -> f64>(out: &mut [f64], f_simd: F, f: U) {
    let (out_chunks, out_remainder) = out.as_chunks_mut::<LANE_WIDTH>();

    for chunk_out in out_chunks.iter_mut() {
        let v = wide::f64x8::from(*chunk_out);

        let result = f_simd(v);

        chunk_out.copy_from_slice(&result.to_array());
    }

    for x in out_remainder.iter_mut() {
        *x = f(*x);
    }
}

#[inline]
fn compute_out(op: &OpKindScalar<f64>, src: &[f64], dst: &mut [f64]) {
    match op {
        OpKindScalar::AxBy(a, b) => {
            for (o, i) in dst.iter_mut().zip(src) {
                *o = *a * *i + *b;
            }
        }
        OpKindScalar::Exp => {
            unsafe { vdExp(dst.len() as i32, src.as_ptr(), dst.as_mut_ptr()) };
        }
        OpKindScalar::Ln => {
            unsafe { vdLn(dst.len() as i32, src.as_ptr(), dst.as_mut_ptr()) };
        }
        OpKindScalar::Log2 => {
            unsafe { vdLog2(dst.len() as i32, src.as_ptr(), dst.as_mut_ptr()) };
        }
        OpKindScalar::Recip => {
            for (o, i) in dst.iter_mut().zip(src) {
                *o = 1.0 / *i;
            }
        }
        OpKindScalar::ReLU => {
            unary_simd(
                src,
                dst,
                |x| x.fast_max(f64x8::splat(0.0)),
                |x| {
                    if x > 0.0 { x } else { 0.0 }
                },
            );
        }
        OpKindScalar::Tanh => {
            unsafe { vdTanh(dst.len() as i32, src.as_ptr(), dst.as_mut_ptr()) };
        }
    }
}

#[inline]
fn compute_inplace(op: &OpKindScalar<f64>, dst: &mut [f64]) {
    match op {
        OpKindScalar::AxBy(a, b) => {
            for o in dst {
                *o = *a * *o + *b;
            }
        }
        OpKindScalar::Exp => {
            unsafe { vdExp(dst.len() as i32, dst.as_ptr(), dst.as_mut_ptr()) };
        }
        OpKindScalar::Ln => {
            unsafe { vdLn(dst.len() as i32, dst.as_ptr(), dst.as_mut_ptr()) };
        }
        OpKindScalar::Log2 => {
            unsafe { vdLog2(dst.len() as i32, dst.as_ptr(), dst.as_mut_ptr()) };
        }
        OpKindScalar::Recip => {
            for o in dst {
                *o = 1.0 / *o;
            }
        }
        OpKindScalar::ReLU => {
            unary_simd_inplace(
                dst,
                |x| x.fast_max(f64x8::splat(0.0)),
                |x| if x > 0.0 { x } else { 0.0 },
            );
        }
        OpKindScalar::Tanh => {
            unsafe { vdTanh(dst.len() as i32, dst.as_ptr(), dst.as_mut_ptr()) };
        }
    }
}

fn compute_element(ops: &[OpKindScalar<f64>], el: f64) -> f64 {
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
    op: &OpKind<f64>,
    output_buffer: &mut [f64],
    output_layout: &Layout,
    inputs: &[TensorData<f64>],
) {
    match op {
        OpKind::ScalarOp(s) => compute_scalar::<TILE_SIZE, f64, _, _, _>(
            inputs[0].data(),
            output_buffer,
            inputs[0].layout(),
            std::slice::from_ref(s),
            compute_out,
            compute_element,
            compute_inplace,
        ),
        OpKind::FusedScalar(ss) => compute_scalar::<TILE_SIZE, f64, _, _, _>(
            inputs[0].data(),
            output_buffer,
            inputs[0].layout(),
            ss,
            compute_out,
            compute_element,
            compute_inplace,
        ),
        OpKind::AsContiguous => {
            clone_to_buffer(&inputs[0], output_buffer);
        }
        OpKind::Add => compute_elementwise(inputs, output_buffer, |x, y| x + y),
        OpKind::Sub => compute_elementwise(inputs, output_buffer, |x, y| x - y),
        OpKind::Mul => compute_elementwise(inputs, output_buffer, |x, y| x * y),
        OpKind::Div => compute_elementwise(inputs, output_buffer, |x, y| x / y),
        OpKind::MatMul(a) => compute_matmul_sum(
            inputs,
            *a,
            0.0,
            output_buffer,
            false,
            cblas_dgemm_batch_strided,
        ),
        OpKind::MatMulSum(a, b, sign) => {
            let beta = if *sign == Sign::Minus { -*b } else { *b };
            compute_matmul_sum(
                inputs,
                *a,
                beta,
                output_buffer,
                true,
                cblas_dgemm_batch_strided,
            )
        }
        OpKind::Slice
        | OpKind::View
        | OpKind::TransposeAxes
        | OpKind::Broadcast
        | OpKind::Transpose
        | OpKind::NoOp => {
            unreachable!("a reference should never appear here");
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
        OpKind::Max => compute_reduction(inputs, output_buffer, f64::NEG_INFINITY, |x, y| {
            if x > y { x } else { y }
        }),
        OpKind::MaxAxis(axis, _) => {
            let axis = normalize_axis(*axis, inputs[0].shape().len());

            compute_reduction_axis(
                inputs,
                axis,
                output_buffer,
                output_layout,
                f64::NEG_INFINITY,
                |x, y| if x > y { x } else { y },
            );
        }
        OpKind::Mean => compute_mean(inputs, output_buffer),
        OpKind::MeanAxis(axis, _) => {
            let axis = normalize_axis(*axis, inputs[0].shape().len());

            compute_mean_axis(inputs, axis, output_buffer, output_layout);
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
    op: &OpKind<f64>,
    output_buffer: &mut [f64],
    output_layout: &Layout,
    inputs: &[TensorData<f64>],
    output_idx: usize,
) {
    match op {
        OpKind::ScalarOp(s) => compute_scalar_inplace::<TILE_SIZE, f64, _, _>(
            std::slice::from_ref(s),
            output_buffer,
            output_layout,
            compute_element,
            compute_inplace,
        ),
        OpKind::FusedScalar(ss) => compute_scalar_inplace::<TILE_SIZE, f64, _, _>(
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
