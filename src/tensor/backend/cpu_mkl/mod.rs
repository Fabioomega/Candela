extern crate cblas;
extern crate intel_mkl_src;
extern crate intel_mkl_sys;

mod f32;
mod f64;
mod kernels;
mod mkl_extension;

use crate::Layout;
use crate::tensor::backend::{Backend, ComputeFor, Dtype};
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::storage::TensorData;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CpuMkl;

impl Backend for CpuMkl {
    const SUPPORTS_2D_TRANSPOSED_MATMUL: bool = true;
    const SUPPORTS_NON_CONTIGUOUS_MATMUL: bool = false;

    fn compute<T>(
        op: &OpKind<T>,
        output_buffer: &mut [T],
        output_layout: &Layout,
        inputs: &[TensorData<T>],
    ) where
        T: Dtype + ComputeFor<CpuMkl>,
    {
        T::compute(op, output_buffer, output_layout, inputs)
    }

    fn compute_inplace<T>(
        op: &OpKind<T>,
        output_buffer: &mut [T],
        output_layout: &Layout,
        inputs: &[TensorData<T>],
        output_idx: usize,
    ) where
        T: Dtype + ComputeFor<Self>,
    {
        T::compute_inplace(op, output_buffer, output_layout, inputs, output_idx)
    }
}

impl ComputeFor<CpuMkl> for f64 {
    fn compute(
        op: &OpKind<Self>,
        output_buffer: &mut [Self],
        output_layout: &Layout,
        inputs: &[TensorData<Self>],
    ) {
        f64::compute_op(op, output_buffer, output_layout, inputs)
    }

    fn compute_inplace(
        op: &OpKind<Self>,
        output_buffer: &mut [Self],
        output_layout: &Layout,
        inputs: &[TensorData<Self>],
        output_idx: usize,
    ) {
        f64::compute_op_inplace(op, output_buffer, output_layout, inputs, output_idx)
    }
}

impl ComputeFor<CpuMkl> for f32 {
    fn compute(
        op: &OpKind<Self>,
        output_buffer: &mut [Self],
        output_layout: &Layout,
        inputs: &[TensorData<Self>],
    ) {
        f32::compute_op(op, output_buffer, output_layout, inputs)
    }

    fn compute_inplace(
        op: &OpKind<Self>,
        output_buffer: &mut [Self],
        output_layout: &Layout,
        inputs: &[TensorData<Self>],
        output_idx: usize,
    ) {
        f32::compute_op_inplace(op, output_buffer, output_layout, inputs, output_idx)
    }
}
