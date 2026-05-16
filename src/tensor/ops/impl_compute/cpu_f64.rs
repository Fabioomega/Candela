use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::mkl_extension::{cblas_dgemm_batch_strided, vdAddI};
use crate::tensor::ops::def_op::{OpKind, Sign};
use crate::tensor::ops::impl_compute::cpu_compute_generic::{
    CommonBLASOps, compute_elementwise_tensor_tensor, compute_elementwise_tensor_tensor_inplace,
    compute_scalar, compute_scalar_inplace, cpu_compute_matmul_sum_scaled,
};
use crate::tensor::storage::{Storage, TensorData};
use cblas_sys::{cblas_daxpy, cblas_dscal};
use intel_mkl_sys::{vdAdd, vdDiv, vdExp, vdLn, vdLog2, vdMul, vdSub};

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        skip(inputs, output_buffer, output_layout),
        fields(op = op.as_str(), out_len = output_layout.len())
    )
)]
pub(crate) fn cpu_compute_op_f64(
    op: &OpKind<f64>,
    mut output_buffer: Vec<f64>,
    output_layout: &Layout,
    inputs: &[TensorData<f64>],
) -> TensorData<f64> {
    const BLAS_OPS: CommonBLASOps<f64> = CommonBLASOps {
        add: vdAddI,
        scal: cblas_dscal,
        axby: cblas_daxpy,
        exp: vdExp,
        ln: vdLn,
        log2: vdLog2,
    };

    match op {
        OpKind::ScalarOp(s) => {
            compute_scalar(&[s.clone()], output_buffer, output_layout, inputs, BLAS_OPS)
        }
        OpKind::FusedScalar(ss) => {
            compute_scalar(ss, output_buffer, output_layout, inputs, BLAS_OPS)
        }
        OpKind::AsContiguous => {
            for (i, el) in inputs[0].iter().enumerate() {
                output_buffer[i] = *el;
            }

            TensorData::new(Storage::from_vec(output_buffer), output_layout.clone())
        }
        OpKind::Add => compute_elementwise_tensor_tensor(inputs, output_buffer, vdAdd),
        OpKind::Sub => compute_elementwise_tensor_tensor(inputs, output_buffer, vdSub),
        OpKind::Mul => compute_elementwise_tensor_tensor(inputs, output_buffer, vdMul),
        OpKind::Div => compute_elementwise_tensor_tensor(inputs, output_buffer, vdDiv),
        OpKind::MatMul(a) => cpu_compute_matmul_sum_scaled(
            inputs,
            *a,
            0.0,
            output_buffer,
            output_layout,
            false,
            cblas_dgemm_batch_strided,
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
                cblas_dgemm_batch_strided,
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
pub(crate) fn cpu_compute_op_f64_inplace(
    op: &OpKind<f64>,
    output_layout: &Layout,
    mut inputs: Vec<TensorData<f64>>,
    output_idx: usize,
) -> TensorData<f64> {
    const BLAS_OPS: CommonBLASOps<f64> = CommonBLASOps {
        add: vdAddI,
        scal: cblas_dscal,
        axby: cblas_daxpy,
        exp: vdExp,
        ln: vdLn,
        log2: vdLog2,
    };

    match op {
        OpKind::ScalarOp(s) => {
            compute_scalar_inplace(&[s.clone()], output_layout, inputs, BLAS_OPS)
        }
        OpKind::FusedScalar(ss) => compute_scalar_inplace(ss, output_layout, inputs, BLAS_OPS),
        OpKind::Add => compute_elementwise_tensor_tensor_inplace(inputs, output_idx, vdAdd),
        OpKind::Sub => compute_elementwise_tensor_tensor_inplace(inputs, output_idx, vdSub),
        OpKind::Mul => compute_elementwise_tensor_tensor_inplace(inputs, output_idx, vdMul),
        OpKind::Div => compute_elementwise_tensor_tensor_inplace(inputs, output_idx, vdDiv),
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

#[cfg(test)]
#[path = "cpu_f64_tests.rs"]
mod tests;
