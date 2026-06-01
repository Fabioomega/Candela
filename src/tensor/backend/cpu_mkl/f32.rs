use crate::Dimension;
use crate::tensor::backend::common::{clone_to_buffer, normalize_axis};
use crate::tensor::backend::common_kernels::{
    compute_max_axis_tensor, compute_max_tensor, compute_mean_axis_tensor, compute_mean_tensor,
    compute_sum_axis_tensor, compute_sum_tensor,
};
use crate::tensor::backend::cpu_mkl::kernels::{
    CommonBLASOps, compute_elementwise_tensor_tensor, compute_elementwise_tensor_tensor_inplace,
    compute_matmul_sum, compute_scalar, compute_scalar_inplace,
};
use crate::tensor::backend::cpu_mkl::mkl_extension::{cblas_sgemm_batch_strided, vsAddI};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::{OpKind, Sign};
use crate::tensor::storage::{Storage, TensorData};
use cblas_sys::{cblas_saxpy, cblas_sscal};
use intel_mkl_sys::{vsAdd, vsDiv, vsExp, vsInv, vsLn, vsLog2, vsMul, vsSub, vsTanh};

const BLAS_OPS: CommonBLASOps<f32> = CommonBLASOps {
    add: vsAddI,
    scal: cblas_sscal,
    axby: cblas_saxpy,
    exp: vsExp,
    ln: vsLn,
    log2: vsLog2,
    inv: vsInv,
    tanh: vsTanh,
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
    op: &OpKind<f32>,
    output_buffer: Vec<f32>,
    output_layout: &Layout,
    inputs: &[TensorData<f32>],
) -> TensorData<f32> {
    match op {
        OpKind::ScalarOp(s) => compute_scalar(
            std::slice::from_ref(s),
            output_buffer,
            output_layout,
            inputs,
            BLAS_OPS,
            0.0,
            |x, y| x.max(y),
        ),
        OpKind::FusedScalar(ss) => compute_scalar(
            ss,
            output_buffer,
            output_layout,
            inputs,
            BLAS_OPS,
            0.0,
            |x, y| x.max(y),
        ),
        OpKind::AsContiguous => {
            let output_buffer = clone_to_buffer(&inputs[0], output_buffer);

            TensorData::new(Storage::from_vec(output_buffer), output_layout.clone())
        }
        OpKind::Add => compute_elementwise_tensor_tensor(inputs, output_buffer, vsAdd),
        OpKind::Sub => compute_elementwise_tensor_tensor(inputs, output_buffer, vsSub),
        OpKind::Mul => compute_elementwise_tensor_tensor(inputs, output_buffer, vsMul),
        OpKind::Div => compute_elementwise_tensor_tensor(inputs, output_buffer, vsDiv),
        OpKind::MatMul(a) => compute_matmul_sum(
            inputs,
            *a,
            0.0,
            output_buffer,
            output_layout,
            false,
            cblas_sgemm_batch_strided,
        ),
        OpKind::MatMulSum(a, b, sign) => {
            let beta = if *sign == Sign::Minus { -*b } else { *b };
            compute_matmul_sum(
                inputs,
                *a,
                beta,
                output_buffer,
                output_layout,
                true,
                cblas_sgemm_batch_strided,
            )
        }
        OpKind::Slice(new_layout)
        | OpKind::View(new_layout)
        | OpKind::TransposeAxes(new_layout)
        | OpKind::Broadcast(new_layout) => inputs[0].as_layout(new_layout.clone()),
        OpKind::Transpose => {
            let layout = inputs[0].layout().transpose();
            inputs[0].as_layout(layout)
        }
        OpKind::Sum => compute_sum_tensor(inputs, output_buffer, output_layout),
        OpKind::SumAxis(axis, _) => {
            let axis = normalize_axis(*axis, inputs[0].shape().len());

            compute_sum_axis_tensor(inputs, axis, output_buffer, output_layout)
        }
        OpKind::Max => compute_max_tensor(inputs, output_buffer, output_layout, |a, b| a.max(b)),
        OpKind::MaxAxis(axis, _) => {
            let axis = normalize_axis(*axis, inputs[0].shape().len());

            compute_max_axis_tensor(inputs, axis, output_buffer, output_layout, |a, b| a.max(b))
        }
        OpKind::Mean => {
            compute_mean_tensor(inputs, output_buffer, output_layout, |a, b| a / (b as f32))
        }
        OpKind::MeanAxis(axis, _) => {
            let axis = normalize_axis(*axis, inputs[0].shape().len());

            compute_mean_axis_tensor(inputs, axis, output_buffer, output_layout, |a, b| {
                a / (b as f32)
            })
        }
        OpKind::NoOp => inputs[0].clone(),
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
pub(crate) fn compute_op_inplace(
    op: &OpKind<f32>,
    output_layout: &Layout,
    mut inputs: Vec<TensorData<f32>>,
    output_idx: usize,
) -> TensorData<f32> {
    match op {
        OpKind::ScalarOp(s) => compute_scalar_inplace(
            std::slice::from_ref(s),
            output_layout,
            inputs,
            BLAS_OPS,
            0.0,
            |x, y| x.max(y),
        ),
        OpKind::FusedScalar(ss) => {
            compute_scalar_inplace(ss, output_layout, inputs, BLAS_OPS, 0.0, |x, y| x.max(y))
        }
        OpKind::Add => compute_elementwise_tensor_tensor_inplace(inputs, output_idx, vsAdd),
        OpKind::Sub => compute_elementwise_tensor_tensor_inplace(inputs, output_idx, vsSub),
        OpKind::Mul => compute_elementwise_tensor_tensor_inplace(inputs, output_idx, vsMul),
        OpKind::Div => compute_elementwise_tensor_tensor_inplace(inputs, output_idx, vsDiv),
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
        OpKind::NoOp | OpKind::AsContiguous => unsafe { inputs.pop().unwrap_unchecked() },
        _ => todo!("not implemented {}", op.as_str()),
    }
}

#[cfg(test)]
#[path = "f32_tests.rs"]
mod tests;
