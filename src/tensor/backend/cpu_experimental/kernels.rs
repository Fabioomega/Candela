use std::iter::zip;

use crate::tensor::backend::common::clone_to_buffer;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::OpKindScalar;
use crate::tensor::storage::TensorData;
use crate::tensor::traits::Numeric;
use crate::{Dimension, branch_duo_fast_iter};

type MatMulFn<T> = unsafe fn(
    m: usize,
    k: usize,
    n: usize,
    alpha: T,
    a: *const T,
    rsa: isize,
    csa: isize,
    b: *const T,
    rsb: isize,
    csb: isize,
    beta: T,
    c: *mut T,
    rsc: isize,
    csc: isize,
);

pub(crate) struct CommonBLASOps<T> {
    pub fma: fn(T, T, T) -> T,
    pub exp: fn(T) -> T,
    pub ln: fn(T) -> T,
    pub log2: fn(T) -> T,
    pub max: fn(T, T) -> T,
    pub tanh: fn(T) -> T,
    pub matmul: MatMulFn<T>,
}

//
#[inline]
pub fn compute_scalar<T: Numeric, F: Fn(&[T], &mut [T], &Layout, &[OpKindScalar<T>])>(
    ops: &[OpKindScalar<T>],
    inputs: &[TensorData<T>],
    output_buffer: &mut [T],
    compute_inline: F,
) {
    let input = &inputs[0];

    compute_inline(input.data(), output_buffer, input.layout(), ops);
}

pub fn compute_scalar_inplace<T: Numeric, F: Fn(&mut [T], &Layout, &[OpKindScalar<T>])>(
    ops: &[OpKindScalar<T>],
    mut inputs: Vec<TensorData<T>>,
    output_layout: &Layout,
    apply: F,
) -> TensorData<T> {
    let mut input = inputs.pop().unwrap();
    let layout = input.layout().clone();

    apply(input.storage.mut_data().unwrap(), &layout, ops);

    input.into_layout(output_layout.clone())
}

pub fn compute_elementwise_tensor_tensor<T: Numeric, F: Fn(T, T) -> T>(
    inputs: &[TensorData<T>],
    output_buffer: &mut [T],
    op: F,
) {
    let a = &inputs[0];
    let b = &inputs[1];

    branch_duo_fast_iter!(a.fast_iter() => a_it, b.fast_iter() => b_it, {
        for ((a_el, b_el), o_el) in zip(zip(a_it, b_it), output_buffer.iter_mut()) {
            *o_el = op(*a_el, *b_el);
        }
    });
}

pub fn compute_elementwise_tensor_tensor_inplace<T: Numeric, F: Fn(T, T) -> T>(
    mut output: TensorData<T>,
    other: TensorData<T>,
    op: F,
) -> TensorData<T> {
    for (o_el, x_el) in zip(output.iter_slice_mut().unwrap(), other.iter()) {
        *o_el = op(*o_el, *x_el);
    }

    output
}

pub fn compute_matmul_sum<T: Clone>(
    inputs: &[TensorData<T>],
    alpha: T,
    beta: T,
    output_buffer: &mut [T],
    fill_output_with_c: bool,
    blas: CommonBLASOps<T>,
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
        clone_to_buffer(c, output_buffer);
    }

    // TODO: The unwrapping here should be sound. We can remove it later.
    let a_dim = a.shape().len();
    let a_3d_layout = if a_dim > 2 {
        a.layout().clone().rotate_axis_innermost(a_dim - 3).unwrap()
    } else {
        a.layout()
            .broadcast(a_shape)
            .unwrap()
            .rotate_axis_innermost(0)
            .unwrap()
    };

    let b_dim = b.shape().len();
    let b_3d_layout = if b_dim > 2 {
        b.layout().rotate_axis_innermost(b_dim - 3).unwrap()
    } else {
        b.layout()
            .broadcast(b_shape)
            .unwrap()
            .rotate_axis_innermost(0)
            .unwrap()
    };

    let batch_dimension_size = a_shape[0];
    debug_assert!(
        a_shape[0] == b_shape[0],
        "one of the tensors in matmul is not correctly broadcasted"
    );

    let a_iter = unsafe { a.iter_as_layout(&a_3d_layout) };
    let b_iter = unsafe { b.iter_as_layout(&b_3d_layout) };

    // TODO: Maybe add a contiguous variant of this iterator if profiling ask for it, but I doubt it will.
    for (batch_idx, (a_ref, b_ref)) in zip(a_iter, b_iter).take(batch_dimension_size).enumerate() {
        unsafe {
            (blas.matmul)(
                m,
                k,
                n,
                alpha.clone(),
                a_ref as *const T,
                a_rs,
                a_cs,
                b_ref as *const T,
                b_rs,
                b_cs,
                beta.clone(),
                output_buffer.as_mut_ptr().add(batch_idx * m * n),
                c_rs,
                c_cs,
            )
        }
    }
}
