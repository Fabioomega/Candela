use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::mkl_extension::vdAddI;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::ops::impl_compute::cpu_compute_generic::{
    CommonBLASOps, compute_elementwise_tensor_tensor, compute_elementwise_tensor_tensor_inplace,
    compute_scalar, compute_scalar_inplace,
};
use crate::tensor::storage::{Storage, TensorData};
use crate::tensor::traits::Dimension;
use cblas::daxpy;
use cblas_sys::{cblas_daxpy, cblas_dgemm, cblas_dscal};
use intel_mkl_sys::{vdAdd, vdDiv, vdExp, vdLn, vdLog2, vdMul, vdSub};

// TODO: Add custom kernel for non-contiguous tensors.
// TODO: Add support for matmul
fn cpu_compute_matmul_f64(
    output_layout: &Layout,
    mut inputs: Vec<TensorData<f64>>,
) -> TensorData<f64> {
    let out = vec![0.0; output_layout.len()];

    let raw_a = inputs.pop().unwrap();
    let raw_b = inputs.pop().unwrap();

    let a_stride_len = raw_a.stride().len();
    let b_stride_len = raw_b.stride().len();

    let mut transa = cblas::Transpose::None;
    let mut is_a_trans = false;
    let mut transb = cblas::Transpose::None;
    let mut is_b_trans = false;

    // Check whether the tensor is transposed between the last 2 axis
    // and if it would be contiguous if it was.
    if raw_a.shape().len() >= 2
        && raw_a.stride()[a_stride_len - 2] == 1
        && raw_a.stride()[a_stride_len - 1] as usize == raw_a.shape()[a_stride_len - 1]
    {
        transa = cblas::Transpose::Ordinary;
        is_a_trans = true;
    }

    if raw_b.shape().len() >= 2
        && raw_b.stride()[b_stride_len - 2] == 1
        && raw_b.stride()[b_stride_len - 1] as usize == raw_b.shape()[b_stride_len - 1]
    {
        transb = cblas::Transpose::Ordinary;
        is_b_trans = true;
    }

    let a_tensor = if is_a_trans
        || raw_a.is_contiguous()
        || (raw_a.shape().len() >= 2 && raw_a.is_contiguous_at_axis(a_stride_len - 2))
    {
        raw_a
    } else {
        raw_a.as_contiguous()
    };

    // cblas_dgemm(cblas::Layout::RowMajor, , transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

    let storage = Storage::from_vec(out);
    TensorData::new(storage, output_layout.clone())
}

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        skip(inputs, output_buffer, output_layout),
        fields(op = op.as_str(), out_len = output_layout.len())
    )
)]
pub fn cpu_compute_op_f64(
    op: &OpKind<f64>,
    output_buffer: Vec<f64>,
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
        OpKind::AsContiguous => TensorData::from_iter(inputs[0].copied_iter(), inputs[0].shape()),
        OpKind::Add => compute_elementwise_tensor_tensor(inputs, output_buffer, vdAdd),
        OpKind::Sub => compute_elementwise_tensor_tensor(inputs, output_buffer, vdSub),
        OpKind::Mul => compute_elementwise_tensor_tensor(inputs, output_buffer, vdMul),
        OpKind::Div => compute_elementwise_tensor_tensor(inputs, output_buffer, vdDiv),
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
pub fn cpu_compute_op_f64_inplace(
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
        | OpKind::TransposeAxes(new_layout) => {
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
