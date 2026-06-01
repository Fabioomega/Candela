mod f32;
mod f64;
mod kernels;

use crate::Layout;
use crate::tensor::backend::{Backend, ComputeFor, Dtype};
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::storage::TensorData;

#[derive(Debug)]
pub struct CpuPure;

impl Backend for CpuPure {
    const SUPPORTS_2D_TRANSPOSED_MATMUL: bool = true;
    const SUPPORTS_NON_CONTIGUOUS_MATMUL: bool = true;

    fn compute<T>(
        op: &OpKind<T>,
        output_buffer: Vec<T>,
        output_layout: &Layout,
        inputs: &[TensorData<T>],
    ) -> TensorData<T>
    where
        T: Dtype + ComputeFor<CpuPure>,
    {
        T::compute(op, output_buffer, output_layout, inputs)
    }

    fn compute_inplace<T>(
        op: &OpKind<T>,
        output_layout: &Layout,
        inputs: Vec<TensorData<T>>,
        output_idx: usize,
    ) -> TensorData<T>
    where
        T: Dtype + ComputeFor<Self>,
    {
        T::compute_inplace(op, output_layout, inputs, output_idx)
    }
}

impl ComputeFor<CpuPure> for f64 {
    fn compute(
        op: &OpKind<f64>,
        output_buffer: Vec<f64>,
        output_layout: &Layout,
        inputs: &[TensorData<f64>],
    ) -> TensorData<f64> {
        f64::compute_op(op, output_buffer, output_layout, inputs)
    }

    fn compute_inplace(
        op: &OpKind<Self>,
        output_layout: &Layout,
        inputs: Vec<TensorData<Self>>,
        output_idx: usize,
    ) -> TensorData<Self> {
        f64::compute_op_inplace(op, output_layout, inputs, output_idx)
    }
}

impl ComputeFor<CpuPure> for f32 {
    fn compute(
        op: &OpKind<f32>,
        output_buffer: Vec<f32>,
        output_layout: &Layout,
        inputs: &[TensorData<f32>],
    ) -> TensorData<f32> {
        f32::compute_op(op, output_buffer, output_layout, inputs)
    }

    fn compute_inplace(
        op: &OpKind<Self>,
        output_layout: &Layout,
        inputs: Vec<TensorData<Self>>,
        output_idx: usize,
    ) -> TensorData<Self> {
        f32::compute_op_inplace(op, output_layout, inputs, output_idx)
    }
}
