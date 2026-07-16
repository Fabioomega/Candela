use crate::Dimension;
use crate::tensor::backend::common::{clone_to_buffer, normalize_axis};
use crate::tensor::backend::common_kernels::{
    compute_max_axis_tensor, compute_max_tensor, compute_mean_axis_tensor, compute_mean_tensor,
    compute_sum_axis_tensor, compute_sum_tensor,
};
use crate::tensor::backend::cpu_pure::kernels::{
    CommonBLASOps, compute_elementwise_tensor_tensor, compute_elementwise_tensor_tensor_inplace,
    compute_matmul_sum, compute_scalar, compute_scalar_inplace,
};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::{OpKind, Sign};
use crate::tensor::storage::{Storage, TensorData};

const BLAS: CommonBLASOps<f64> = CommonBLASOps {
    fma: |a, b, c| a.mul_add(b, c),
    exp: |a| a.exp(),
    ln: |a| a.ln(),
    log2: |a| a.log2(),
    max: |a, b| a.max(b),
    tanh: |a| a.tanh(),
    matmul: matrixmultiply::dgemm,
};

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        skip(inputs, _output_buffer, output_layout),
        fields(op = op.as_str(), out_len = output_layout.len())
    )
)]
pub(crate) fn compute_op(
    op: &OpKind<f64>,
    mut _output_buffer: Vec<f64>,
    output_layout: &Layout,
    inputs: &[TensorData<f64>],
) -> TensorData<f64> {
    let output_buffer = &mut _output_buffer;

    let layout = output_layout.clone();

    match op {
        OpKind::ScalarOp(s) => compute_scalar(std::slice::from_ref(s), inputs, output_buffer, BLAS),
        OpKind::FusedScalar(ss) => compute_scalar(ss, inputs, output_buffer, BLAS),
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
        OpKind::Mean => compute_mean_tensor(inputs, output_buffer, |a, b| a / b as f64),
        OpKind::MeanAxis(axis, _) => {
            let axis = normalize_axis(*axis, inputs[0].shape().len());

            compute_mean_axis_tensor(inputs, axis, output_buffer, |a, b| a / b as f64)
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
    op: &OpKind<f64>,
    output_layout: &Layout,
    mut inputs: Vec<TensorData<f64>>,
    output_idx: usize,
) -> TensorData<f64> {
    match op {
        OpKind::ScalarOp(s) => {
            compute_scalar_inplace(std::slice::from_ref(s), inputs, output_layout, BLAS)
        }
        OpKind::FusedScalar(ss) => compute_scalar_inplace(ss, inputs, output_layout, BLAS),
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

#[cfg(test)]
#[path = "f64_tests.rs"]
mod tests;
