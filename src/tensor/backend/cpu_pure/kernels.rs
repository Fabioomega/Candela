use std::iter::zip;

use crate::tensor::backend::common::clone_to_buffer;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::OpKindScalar;
use crate::tensor::storage::{Storage, TensorData};
use crate::tensor::traits::{Numeric, StreamingIterator};
use crate::{Dimension, PACKING_BUFFER_SIZE, branch_duo_fast_iter, branch_fast_iter};

pub(crate) struct CommonBLASOps<T> {
    pub fma: fn(T, T, T) -> T,
    pub exp: fn(T) -> T,
    pub ln: fn(T) -> T,
    pub log2: fn(T) -> T,
    pub max: fn(T, T) -> T,
    pub tanh: fn(T) -> T,
}

pub fn compute_scalar<T: Numeric>(
    ops: &[OpKindScalar<T>],
    inputs: &[TensorData<T>],
    mut output_buffer: Vec<T>,
    output_layout: &Layout,
    blas: CommonBLASOps<T>,
) -> TensorData<T> {
    let input = &inputs[0];
    output_buffer.fill(T::ZERO);

    branch_fast_iter!(input.fast_packed_iter(PACKING_BUFFER_SIZE) => it, {
        let mut it = it;
        while let Some(chunk) = it.next_stream() {
        let start = chunk.absolute_buffer_position;
        for (i_el, o_el) in zip(
            chunk.packing_buffer.iter(),
            output_buffer[start..].iter_mut(),
        ) {
            for op in ops {
                match op {
                    OpKindScalar::AxBy(a, b) => *o_el = (blas.fma)(*o_el, a.clone(), b.clone()),
                    OpKindScalar::Exp => *o_el = (blas.exp)(i_el.clone()),
                    OpKindScalar::Ln => *o_el = (blas.ln)(i_el.clone()),
                    OpKindScalar::Log2 => *o_el = (blas.log2)(i_el.clone()),
                    OpKindScalar::Inv => *o_el = T::ONE / i_el.clone(),
                    OpKindScalar::ReLU => *o_el = (blas.max)(i_el.clone(), T::ZERO),
                    OpKindScalar::Tanh => *o_el = (blas.tanh)(i_el.clone()),
                }
            }
        }
    }
    });

    TensorData::new(Storage::from_vec(output_buffer), output_layout.clone())
}

pub fn compute_scalar_inplace<T: Numeric>(
    ops: &[OpKindScalar<T>],
    mut inputs: Vec<TensorData<T>>,
    output_layout: &Layout,
    blas: CommonBLASOps<T>,
) -> TensorData<T> {
    let mut input = inputs.pop().unwrap();
    for o_el in input.iter_mut().unwrap() {
        for op in ops {
            match op {
                OpKindScalar::AxBy(a, b) => *o_el = (blas.fma)(*o_el, a.clone(), b.clone()),
                OpKindScalar::Exp => *o_el = (blas.exp)(o_el.clone()),
                OpKindScalar::Ln => *o_el = (blas.ln)(o_el.clone()),
                OpKindScalar::Log2 => *o_el = (blas.log2)(o_el.clone()),
                OpKindScalar::Inv => *o_el = T::ONE / o_el.clone(),
                OpKindScalar::ReLU => *o_el = (blas.max)(o_el.clone(), T::ZERO),
                OpKindScalar::Tanh => *o_el = (blas.tanh)(o_el.clone()),
            }
        }
    }

    input.into_layout(output_layout.clone())
}

pub fn compute_elementwise_tensor_tensor<T: Numeric, F: Fn(T, T) -> T>(
    inputs: &[TensorData<T>],
    mut output_buffer: Vec<T>,
    op: F,
) -> TensorData<T> {
    let a = &inputs[0];
    let b = &inputs[1];

    branch_duo_fast_iter!(a.fast_iter() => a_it, b.fast_iter() => b_it, {
        for ((a_el, b_el), o_el) in zip(zip(a_it, b_it), output_buffer.iter_mut()) {
            *o_el = op(a_el.clone(), b_el.clone());
        }
    });

    TensorData::new(Storage::from_vec(output_buffer), a.layout().clone())
}

pub fn compute_elementwise_tensor_tensor_inplace<T: Numeric, F: Fn(T, T) -> T>(
    mut output: TensorData<T>,
    other: TensorData<T>,
    op: F,
) -> TensorData<T> {
    for (o_el, x_el) in zip(output.iter_mut().unwrap(), other.iter()) {
        *o_el = op(o_el.clone(), x_el.clone());
    }

    output
}

pub fn compute_matmul_sum(
    inputs: &[TensorData<f64>],
    alpha: f64,
    beta: f64,
    mut output_buffer: Vec<f64>,
    output_layout: &Layout,
    fill_output_with_c: bool,
) {
    let a = &inputs[0];
    let b = &inputs[1];

    let a_shape = a.layout().shape_as_3d();
    let a_stride_len = a.stride().len();

    let b_shape = b.layout().shape_as_3d();
    let b_stride_len = b.stride().len();

    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[2];

    let a_rs = a.stride()[a_stride_len - 2] as isize;
    let a_cs = a.stride()[a_stride_len - 1] as isize;
    let b_rs = b.stride()[b_stride_len - 2] as isize;
    let b_cs = b.stride()[b_stride_len - 1] as isize;

    // Output buffer is a fresh contiguous (m, n) allocation owned by us.
    let c_rs = n as isize;
    let c_cs = 1isize;

    if fill_output_with_c {
        let c = &inputs[2];
        output_buffer = clone_to_buffer(c, output_buffer);
    }

    unsafe {
        matrixmultiply::dgemm(
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            a_rs,
            a_cs,
            b.as_ptr(),
            b_rs,
            b_cs,
            beta,
            output_buffer.as_mut_ptr(),
            c_rs,
            c_cs,
        )
    };
}
