pub mod def_op;
pub mod fusion;
pub mod impl_compute;
mod impl_layout;
pub mod impl_op;

pub use impl_layout::compute_layout;

use crate::tensor::definitions::NumberLike;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::OpKind;
use crate::tensor::ops::impl_compute::{cpu_compute_op_f64, cpu_compute_op_f64_inplace};
use crate::tensor::storage::TensorData;

pub(crate) trait ComputeWrapperSpec
where
    Self: Copy,
{
    const MUL_NEUTRAL: Self;
    const SUM_NEUTRAL: Self;

    fn compute_for_type(
        op: &OpKind<Self>,
        output_buffer: Vec<Self>,
        output_layout: &Layout,
        inputs: &[TensorData<Self>],
    ) -> TensorData<Self>;

    fn compute_for_type_inplace(
        op: &OpKind<Self>,
        output_layout: &Layout,
        inputs: Vec<TensorData<Self>>,
        output_idx: usize,
    ) -> TensorData<Self>;
}

impl ComputeWrapperSpec for f64 {
    const MUL_NEUTRAL: Self = 1.0;
    const SUM_NEUTRAL: Self = 0.0;

    #[inline]
    fn compute_for_type(
        op: &OpKind<f64>,
        output_buffer: Vec<f64>,
        output_layout: &Layout,
        inputs: &[TensorData<f64>],
    ) -> TensorData<f64> {
        cpu_compute_op_f64(op, output_buffer, output_layout, inputs)
    }

    #[inline]
    fn compute_for_type_inplace(
        op: &OpKind<f64>,
        output_layout: &Layout,
        inputs: Vec<TensorData<f64>>,
        output_idx: usize,
    ) -> TensorData<f64> {
        cpu_compute_op_f64_inplace(op, output_layout, inputs, output_idx)
    }
}

pub(crate) trait TensorElement: NumberLike + ComputeWrapperSpec {}
impl<T: NumberLike + ComputeWrapperSpec> TensorElement for T {}

#[inline]
pub(crate) fn cpu_compute<T: ComputeWrapperSpec>(
    op: &OpKind<T>,
    output_buffer: Vec<T>,
    output_layout: &Layout,
    inputs: &[TensorData<T>],
) -> TensorData<T> {
    T::compute_for_type(op, output_buffer, output_layout, inputs)
}

#[inline]
pub(crate) fn cpu_compute_inplace<T: ComputeWrapperSpec>(
    op: &OpKind<T>,
    output_layout: &Layout,
    inputs: Vec<TensorData<T>>,
    output_idx: usize,
) -> TensorData<T> {
    T::compute_for_type_inplace(op, output_layout, inputs, output_idx)
}
