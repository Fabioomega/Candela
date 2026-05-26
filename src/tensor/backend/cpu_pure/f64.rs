use crate::tensor::backend::common::clone_to_buffer;
use crate::tensor::backend::cpu_pure::kernels::{
    CommonBLASOps, compute_elementwise_tensor_tensor, compute_elementwise_tensor_tensor_inplace,
    compute_scalar, compute_scalar_inplace,
};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::{OpKind, OpKindScalar, Sign};
use crate::tensor::storage::{Storage, TensorData};

const blas: CommonBLASOps<f64> = CommonBLASOps {
    fma: |a, b, c| a.mul_add(b, c),
    exp: |a| a.exp(),
    ln: |a| a.ln(),
    log2: |a| a.log2(),
    max: |a, b| a.max(b),
    tanh: |a| a.tanh(),
};

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
    mut output_buffer: Vec<f64>,
    output_layout: &Layout,
    inputs: &[TensorData<f64>],
) -> TensorData<f64> {
    match op {
        OpKind::ScalarOp(s) => {
            compute_scalar(&[s.clone()], inputs, output_buffer, output_layout, blas)
        }
        OpKind::FusedScalar(ss) => compute_scalar(ss, inputs, output_buffer, output_layout, blas),
        OpKind::AsContiguous => {
            let output_buffer = clone_to_buffer(&inputs[0], output_buffer);

            TensorData::new(Storage::from_vec(output_buffer), output_layout.clone())
        }
        OpKind::Add => compute_elementwise_tensor_tensor(inputs, output_buffer, |a, b| a + b),
        OpKind::Sub => compute_elementwise_tensor_tensor(inputs, output_buffer, |a, b| a - b),
        OpKind::Mul => compute_elementwise_tensor_tensor(inputs, output_buffer, |a, b| a * b),
        OpKind::Div => compute_elementwise_tensor_tensor(inputs, output_buffer, |a, b| a / b),
        OpKind::MatMul(a) => {}
        OpKind::MatMulSum(a, b, sign) => {}
        OpKind::Slice(new_layout)
        | OpKind::View(new_layout)
        | OpKind::TransposeAxes(new_layout)
        | OpKind::Broadcast(new_layout) => inputs[0].as_layout(new_layout.clone()),
        OpKind::Transpose => {
            let layout = inputs[0].layout().transpose();
            inputs[0].as_layout(layout)
        }
        OpKind::Sum => {}
        OpKind::SumAxis(axis, _) => {}
        OpKind::Max => {}
        OpKind::MaxAxis(axis, _) => {}
        OpKind::Mean => {}
        OpKind::MeanAxis(axis, _) => {}
        OpKind::NoOp => inputs[0].clone(),
        _ => todo!("not implemented {}", op.as_str()),
    }
}

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        skip(inputs, output_layout),
        fields(op = op.as_str(), out_len = output_layout.len())
    )
)]
pub(crate) fn compute_op_f64_inplace(
    op: &OpKind<f64>,
    output_layout: &Layout,
    mut inputs: Vec<TensorData<f64>>,
    output_idx: usize,
) -> TensorData<f64> {
    match op {
        OpKind::ScalarOp(s) => compute_scalar_inplace(&[s.clone()], inputs, output_layout, blas),
        OpKind::FusedScalar(ss) => compute_scalar_inplace(ss, inputs, output_layout, blas),
        OpKind::Add => {
            let mut a = inputs.pop().unwrap();
            let mut b = inputs.pop().unwrap();
            if output_idx == 0 {
                compute_elementwise_tensor_tensor_inplace(a, b, |a, b| a + b)
            } else {
                compute_elementwise_tensor_tensor_inplace(b, a, |a, b| a + b)
            }
        }
        OpKind::Sub => {
            let mut a = inputs.pop().unwrap();
            let mut b = inputs.pop().unwrap();
            if output_idx == 0 {
                compute_elementwise_tensor_tensor_inplace(a, b, |a, b| a - b)
            } else {
                compute_elementwise_tensor_tensor_inplace(b, a, |b, a| a - b)
            }
        }
        OpKind::Mul => {
            let mut a = inputs.pop().unwrap();
            let mut b = inputs.pop().unwrap();
            if output_idx == 0 {
                compute_elementwise_tensor_tensor_inplace(a, b, |a, b| a * b)
            } else {
                compute_elementwise_tensor_tensor_inplace(b, a, |a, b| a * b)
            }
        }
        OpKind::Div => {
            let mut a = inputs.pop().unwrap();
            let mut b = inputs.pop().unwrap();
            if output_idx == 0 {
                compute_elementwise_tensor_tensor_inplace(a, b, |a, b| a / b)
            } else {
                compute_elementwise_tensor_tensor_inplace(b, a, |b, a| a / b)
            }
        }
        OpKind::Slice(new_layout)
        | OpKind::View(new_layout)
        | OpKind::TransposeAxes(new_layout)
        | OpKind::Broadcast(new_layout) => {
            unsafe { inputs.pop().unwrap_unchecked() }.into_layout(new_layout.clone())
        }
        OpKind::Transpose => {
            let layout = inputs[0].layout().transpose();
            unsafe { inputs.pop().unwrap_unchecked() }.into_layout(layout)
        }
        OpKind::NoOp => unsafe { inputs.pop().unwrap_unchecked() },
        _ => todo!("not implemented"),
    }
}
