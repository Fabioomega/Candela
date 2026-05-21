use crate::Dimension;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::mkl_extension::{cblas_sgemm_batch_strided, vsAddI};
use crate::tensor::ops::def_op::{OpKind, Sign};
use crate::tensor::ops::impl_compute::cpu_compute_generic::{
    CommonBLASOps, compute_elementwise_tensor_tensor, compute_elementwise_tensor_tensor_inplace,
    compute_scalar, compute_scalar_inplace, cpu_compute_matmul_sum_scaled,
    cpu_compute_max_axis_tensor, cpu_compute_max_tensor, cpu_compute_mean_axis_tensor,
    cpu_compute_mean_tensor, cpu_compute_sum_axis_tensor, cpu_compute_sum_tensor, normalize_axis,
};
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
pub(crate) fn cpu_compute_op_f32(
    op: &OpKind<f32>,
    mut output_buffer: Vec<f32>,
    output_layout: &Layout,
    inputs: &[TensorData<f32>],
) -> TensorData<f32> {
    match op {
        OpKind::ScalarOp(s) => compute_scalar(
            &[s.clone()],
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
            for (i, el) in inputs[0].iter().enumerate() {
                output_buffer[i] = *el;
            }

            TensorData::new(Storage::from_vec(output_buffer), output_layout.clone())
        }
        OpKind::Add => compute_elementwise_tensor_tensor(inputs, output_buffer, vsAdd),
        OpKind::Sub => compute_elementwise_tensor_tensor(inputs, output_buffer, vsSub),
        OpKind::Mul => compute_elementwise_tensor_tensor(inputs, output_buffer, vsMul),
        OpKind::Div => compute_elementwise_tensor_tensor(inputs, output_buffer, vsDiv),
        OpKind::MatMul(a) => cpu_compute_matmul_sum_scaled(
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
            cpu_compute_matmul_sum_scaled(
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
        OpKind::Sum => cpu_compute_sum_tensor(inputs, output_buffer, output_layout, 0.0),
        OpKind::SumAxis(axis, _) => {
            let axis = normalize_axis::<f32>(*axis, inputs[0].shape().len());

            cpu_compute_sum_axis_tensor(inputs, axis, output_buffer, output_layout, 0.0)
        }
        OpKind::Max => cpu_compute_max_tensor(
            inputs,
            output_buffer,
            output_layout,
            f32::NEG_INFINITY,
            |a, b| a.max(b),
        ),
        OpKind::MaxAxis(axis, _) => {
            let axis = normalize_axis::<f32>(*axis, inputs[0].shape().len());

            cpu_compute_max_axis_tensor(
                inputs,
                axis,
                output_buffer,
                output_layout,
                f32::NEG_INFINITY,
                |a, b| a.max(b),
            )
        }
        OpKind::Mean => {
            cpu_compute_mean_tensor(inputs, output_buffer, output_layout, 0.0, |a, b| {
                a / (b as f32)
            })
        }
        OpKind::MeanAxis(axis, _) => {
            let axis = normalize_axis::<f32>(*axis, inputs[0].shape().len());

            cpu_compute_mean_axis_tensor(inputs, axis, output_buffer, output_layout, 0.0, |a, b| {
                a / (b as f32)
            })
        }
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
pub(crate) fn cpu_compute_op_f32_inplace(
    op: &OpKind<f32>,
    output_layout: &Layout,
    mut inputs: Vec<TensorData<f32>>,
    output_idx: usize,
) -> TensorData<f32> {
    match op {
        OpKind::ScalarOp(s) => compute_scalar_inplace(
            &[s.clone()],
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
        OpKind::NoOp => unsafe { inputs.pop().unwrap_unchecked() },
        _ => todo!("not implemented"),
    }
}

// #[cfg(test)]
// #[path = "cpu_f32_tests.rs"]
// mod tests;
