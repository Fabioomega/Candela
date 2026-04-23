use crate::tensor::definitions::{ChunkedIter, NumberLike};
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::{OpKind, OpKindScalar};
use crate::tensor::ops::reusable::{get_reusable_or_alloc, unordered_get_reusable_or_alloc_n};
use crate::tensor::storage::{Storage, TensorData};
use crate::tensor::traits::{Dimension, StreamingIterator};
use cblas_sys::cblas_dgemm;
use intel_mkl_sys::{vdAdd, vdDiv, vdMul, vdSub};

// TODO: Add BLAS support for scalar ops using vdAddl and the like
fn compute_scalar_op<T: NumberLike>(op: &OpKindScalar<T>, mut input: Vec<T>) -> Vec<T> {
    match op {
        OpKindScalar::Sum(scalar) => {
            for el in input.iter_mut() {
                *el = *el + *scalar;
            }
            input
        }
        OpKindScalar::Sub(scalar) => {
            for el in input.iter_mut() {
                *el = *el - *scalar;
            }
            input
        }
        OpKindScalar::Mul(scalar) => {
            for el in input.iter_mut() {
                *el = *el * *scalar;
            }
            input
        }
        OpKindScalar::Div(scalar) => {
            for el in input.iter_mut() {
                *el = *el / *scalar;
            }

            input
        }
    }
}

fn compute_elementwise_tensor_tensor<T: Copy + Default>(
    mut inputs: Vec<TensorData<T>>,
    operation: unsafe extern "C" fn(i32, *const T, *const T, *mut T),
) -> TensorData<T> {
    // TODO: This is a mess. It would be ideal if we can design the operations without having to
    // think about reusability and then plug it on some magic and it starts reusing tensors.
    let mut output_data = unordered_get_reusable_or_alloc_n(&mut inputs, 0);

    // Non-contiguous path
    if !inputs[0].is_contiguous() {
        // TODO: There's no need to pack the input. Maybe we should
        // allocate a full buffer and then operate directly
        let mut packed_iter: ChunkedIter<'_, T> = inputs[0].packed_iter();

        while let Some(chunk) = packed_iter.next() {
            let buffer_size: usize = chunk.packing_buffer.len();

            unsafe {
                operation(
                    buffer_size as i32,
                    output_data
                        .v
                        .as_ptr()
                        .add(chunk.absolute_buffer_position + output_data.offset),
                    chunk.packing_buffer.as_ptr(),
                    output_data
                        .v
                        .as_mut_ptr()
                        .add(chunk.absolute_buffer_position + output_data.offset),
                )
            }
        }
    // Contiguous path
    } else {
        let lhs_buffer = &inputs[0].storage.buffer;

        unsafe {
            operation(
                (output_data.v.len() - output_data.offset) as i32,
                output_data.v.as_ptr().add(output_data.offset),
                lhs_buffer.as_ptr(),
                output_data.v.as_mut_ptr().add(output_data.offset),
            )
        }
    }

    TensorData::from_vec(output_data.v, inputs[0].shape(), output_data.offset).mark_as_reusable()
}

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
        && raw_a.stride()[a_stride_len - 1] == raw_a.shape()[a_stride_len - 1]
    {
        transa = cblas::Transpose::Ordinary;
        is_a_trans = true;
    }

    if raw_b.shape().len() >= 2
        && raw_b.stride()[b_stride_len - 2] == 1
        && raw_b.stride()[b_stride_len - 1] == raw_b.shape()[b_stride_len - 1]
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

fn cpu_compute_elementwise_f64(
    op: &OpKind<f64>,
    output_layout: &Layout,
    mut inputs: Vec<TensorData<f64>>,
) -> TensorData<f64> {
    let buffer = get_reusable_or_alloc(inputs.pop().unwrap());

    match op {
        OpKind::ScalarOp(op) => TensorData::from_vec(
            compute_scalar_op(op, buffer.v),
            output_layout.shape(),
            buffer.offset,
        ),
        OpKind::FusedScalar(ops) => {
            let offset = buffer.offset;
            let mut buffer = buffer.v;

            for op in ops {
                buffer = compute_scalar_op(op, buffer);
            }

            TensorData::from_vec(buffer, output_layout.shape(), offset)
        }
        _ => unreachable!("no other op should appear here"),
    }
    .mark_as_reusable()
}

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        level = "debug",
        skip(inputs, output_layout),
        fields(op = op.as_str(), out_len = output_layout.len())
    )
)]
fn cpu_compute_op_f64(
    op: &OpKind<f64>,
    output_layout: &Layout,
    mut inputs: Vec<TensorData<f64>>,
) -> TensorData<f64> {
    match op {
        OpKind::ScalarOp(_) | OpKind::FusedScalar(_) => {
            cpu_compute_elementwise_f64(op, output_layout, inputs)
        }
        OpKind::Slice(new_layout)
        | OpKind::View(new_layout)
        | OpKind::TransposeAxes(new_layout) => inputs[0].as_layout(new_layout.clone()),
        OpKind::AsContiguous => {
            if inputs[0].is_contiguous() {
                inputs[0].clone_reference()
            } else {
                TensorData::from_iter(inputs[0].copied_iter(), inputs[0].shape()).mark_as_reusable()
            }
        }
        OpKind::Transpose => {
            let layout = inputs[0].layout();
            inputs[0].as_layout(layout.transpose())
        }
        OpKind::Add => compute_elementwise_tensor_tensor(inputs, vdAdd),
        OpKind::Sub => compute_elementwise_tensor_tensor(inputs, vdSub),
        OpKind::Mul => compute_elementwise_tensor_tensor(inputs, vdMul),
        OpKind::Div => compute_elementwise_tensor_tensor(inputs, vdDiv),
        OpKind::NoOp => unsafe { inputs.pop().unwrap_unchecked() },
        _ => todo!("not implemented"),
    }
}

pub trait ComputeWrapperSpec
where
    Self: Copy,
{
    fn compute_for_type(
        op: &OpKind<Self>,
        output_layout: &Layout,
        inputs: Vec<TensorData<Self>>,
    ) -> TensorData<Self>;
}

impl ComputeWrapperSpec for f64 {
    #[inline]
    fn compute_for_type(
        op: &OpKind<f64>,
        output_layout: &Layout,
        inputs: Vec<TensorData<f64>>,
    ) -> TensorData<f64> {
        cpu_compute_op_f64(op, output_layout, inputs)
    }
}

#[inline]
pub fn cpu_compute<T: ComputeWrapperSpec>(
    op: &OpKind<T>,
    output_layout: &Layout,
    inputs: Vec<TensorData<T>>,
) -> TensorData<T> {
    T::compute_for_type(op, output_layout, inputs)
}
