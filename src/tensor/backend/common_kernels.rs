use std::iter::zip;

use crate::tensor::ops::def_op::OpKindScalar;
use crate::tensor::storage::TensorData;
use crate::tensor::traits::Numeric;
use crate::tensor::walker::{DimWalker, fold_chunk, map_chunk, map_chunk_inplace};
use crate::tensor::walker::{map2, map2_inplace};
use crate::tensor::{FromIndex, MAX_DIMS};
use crate::{Dimension, Layout};

#[inline]
pub fn compute_scalar<const TILE: usize, T, F, E, R>(
    src: &[T],
    dst: &mut [T],
    layout: &Layout,
    ops: &[OpKindScalar<T>],
    op_out: F,
    op_element: E,
    op_inplace: R,
) where
    T: Clone,
    F: Fn(&OpKindScalar<T>, &[T], &mut [T]),
    E: Fn(&[OpKindScalar<T>], T) -> T,
    R: Fn(&OpKindScalar<T>, &mut [T]),
{
    let chain = |src: &[T], dst: &mut [T]| {
        // SAFETY: A ops always has at least a single operation otherwise it
        // cannot exist.
        let (first, rest) = unsafe { ops.split_first().unwrap_unchecked() };

        op_out(first, src, dst);
        for op in rest {
            op_inplace(op, dst);
        }
    };

    let ch = |src: &[T], dst: &mut [T]| {
        if TILE == 0 {
            chain(src, dst);
        } else {
            let (i_tile, i_remainder) = src.as_chunks::<TILE>();
            let (o_tile, o_remainder) = dst.as_chunks_mut::<TILE>();

            for (i, o) in zip(i_tile, o_tile) {
                chain(i, o);
            }

            chain(i_remainder, o_remainder);
        }
    };

    let elem = |src| op_element(ops, src);

    map_chunk(src, layout, dst, ch, elem);
}

#[inline]
fn compute_scalar_inplace_internal<
    const TILE: usize,
    T: Clone,
    E: Fn(&[OpKindScalar<T>], T) -> T,
    R: Fn(&OpKindScalar<T>, &mut [T]),
>(
    dst: &mut [T],
    layout: &Layout,
    ops: &[OpKindScalar<T>],
    op_element: E,
    op_inplace: R,
) {
    let ch = |dst: &mut [T]| {
        if TILE == 0 {
            for op in ops {
                op_inplace(op, dst);
            }
        } else {
            let (o_tile, o_remainder) = dst.as_chunks_mut::<TILE>();

            for tile in o_tile {
                for op in ops {
                    op_inplace(op, tile);
                }
            }

            for op in ops {
                op_inplace(op, o_remainder);
            }
        }
    };

    let elem = |src| op_element(ops, src);

    map_chunk_inplace(dst, layout, ch, elem);
}

#[inline]
pub fn compute_scalar_inplace<const TILE: usize, T, E, R>(
    ops: &[OpKindScalar<T>],
    mut inputs: Vec<TensorData<T>>,
    output_layout: &Layout,
    op_element: E,
    op_inplace: R,
) -> TensorData<T>
where
    T: Clone,
    E: Fn(&[OpKindScalar<T>], T) -> T,
    R: Fn(&OpKindScalar<T>, &mut [T]),
{
    let mut input = inputs.pop().unwrap();
    // The output must be contiguous
    debug_assert!(input.is_contiguous());
    let layout = input.layout().clone();

    compute_scalar_inplace_internal::<TILE, T, E, R>(
        input.storage.mut_data().unwrap(),
        &layout,
        ops,
        op_element,
        op_inplace,
    );

    let lay = output_layout
        .clone()
        .with_offset(input.offset() + output_layout.offset());

    input.into_layout(lay)
}

#[inline]
pub fn compute_elementwise<T: Numeric, F: Fn(T, T) -> T>(
    inputs: &[TensorData<T>],
    output_buffer: &mut [T],
    op: F,
) {
    let a = &inputs[0];
    let b = &inputs[1];

    map2(
        a.data(),
        a.layout(),
        b.data(),
        b.layout(),
        output_buffer,
        op,
    );
}

#[inline]
fn compute_elementwise_inplace_internal<T: Numeric, F: Fn(T, T) -> T>(
    mut output: TensorData<T>,
    other: TensorData<T>,
    op: F,
) -> TensorData<T> {
    // The output must be contiguous
    debug_assert!(output.is_contiguous());
    map2_inplace(output.mut_data().unwrap(), other.data(), other.layout(), op);

    output
}

#[inline]
pub fn compute_elementwise_inplace<T: Numeric, F: Fn(T, T) -> T>(
    mut inputs: Vec<TensorData<T>>,
    output_idx: usize,
    f: F,
) -> TensorData<T> {
    let b = inputs.pop().unwrap();
    let a = inputs.pop().unwrap();

    if output_idx == 0 {
        compute_elementwise_inplace_internal(a, b, f)
    } else {
        compute_elementwise_inplace_internal(b, a, |b, a| f(a, b))
    }
}

////////////////////////////////////////////////////

fn remove_idx(l: &Layout, idx: usize) -> Layout {
    let mut shape: [usize; MAX_DIMS] = l.shape.clone();
    let mut stride: [i32; MAX_DIMS] = l.stride.clone();
    let rank = l.rank;

    for i in (idx + 1)..rank {
        shape[i - 1] = shape[i];
        stride[i - 1] = stride[i];
    }

    let len = l.len() / l.shape()[idx];

    Layout::from_raw_parts(&shape[..rank - 1], &stride[..rank - 1], l.offset(), len)
}

#[inline]
pub(crate) fn compute_reduction<T: Copy, F: Fn(T, T) -> T>(
    inputs: &[TensorData<T>],
    output_buffer: &mut [T],
    start_value: T,
    op: F,
) {
    const N_ACC: usize = 8;
    const LANE: usize = 16;

    let input = &inputs[0];

    let ch = |acc, chunk: &[T]| {
        let mut internal_acc: [[T; LANE]; N_ACC] = [[start_value.clone(); LANE]; N_ACC];

        let (chunks, remainder) = chunk.as_chunks::<LANE>();
        let (groups, groups_r) = chunks.as_chunks::<N_ACC>();

        for group in groups {
            for (i, c) in group.iter().enumerate() {
                for k in 0..LANE {
                    internal_acc[i][k] = op(internal_acc[i][k], c[k]);
                }
            }
        }

        for r in groups_r {
            for k in 0..LANE {
                internal_acc[0][k] = op(internal_acc[0][k], r[k]);
            }
        }

        let mut collapsed = internal_acc[0];
        for x in &internal_acc[1..] {
            for k in 0..LANE {
                collapsed[k] = op(collapsed[k], x[k]);
            }
        }

        let mut result = collapsed.iter().fold(acc, |acc, x| op(acc, *x));

        for r in remainder {
            result = op(result, *r);
        }

        result
    };

    let elem = |acc, x| op(acc, x);
    output_buffer[0] = fold_chunk(input.data(), input.layout(), start_value, ch, elem);
}

#[inline]
pub fn compute_reduction_axis<'a, T: Copy, F: Fn(T, T) -> T>(
    inputs: &[TensorData<T>],
    axis: usize,
    output_buffer: &'a mut [T],
    output_layout: &Layout,
    start_value: T,
    op: F,
) {
    const N_ACC: usize = 4;
    const LANE: usize = 16;

    let t = &inputs[0];
    let input = t.data();
    let n_size: usize = t.shape()[axis];

    let axis_stride: isize = t.stride()[axis] as isize;

    // Layout does not support 0D tensors
    if t.shape().len() == 1 {
        let mut acc = start_value;
        let mut pos = t.offset();

        for _ in 0..n_size {
            acc = op(acc, input[pos]);
            pos = pos.wrapping_add_signed(axis_stride);
        }

        output_buffer[0] = acc;
        return;
    }

    let input_layout = &remove_idx(t.layout(), axis);
    let output_layout = if t.layout().shape().len() == output_layout.shape().len() {
        let offset = output_layout.offset();
        &Layout::new(input_layout.shape()).with_offset(offset)
    } else {
        output_layout
    };

    let walker = DimWalker::new([output_layout, input_layout]);
    let (len, strides) = walker.strides();

    let chunk_len = len / LANE;

    match strides {
        [1, 1] => {
            let mut process = |offsets: [usize; 2]| {
                let o_offset = offsets[0];
                let i_offset = offsets[1];

                let mut lane_start = 0;

                while lane_start + N_ACC <= chunk_len {
                    let mut pos = i_offset;
                    let mut acc = [[start_value; LANE]; N_ACC];

                    for _ in 0..n_size {
                        let (input_chunks, _) = input[pos..pos + len].as_chunks::<LANE>();

                        for k in 0..N_ACC {
                            for i in 0..LANE {
                                acc[k][i] = op(acc[k][i], input_chunks[lane_start + k][i]);
                            }
                        }

                        pos = pos.wrapping_add_signed(axis_stride);
                    }

                    let (output_chunks, _) =
                        output_buffer[o_offset..o_offset + len].as_chunks_mut::<LANE>();

                    for k in 0..N_ACC {
                        output_chunks[k + lane_start].copy_from_slice(&acc[k]);
                    }

                    lane_start += N_ACC;
                }

                while lane_start < chunk_len {
                    let mut pos = i_offset;
                    let mut acc = [start_value; LANE];

                    for _ in 0..n_size {
                        let (input_chunks, _) = input[pos..pos + len].as_chunks::<LANE>();

                        for i in 0..LANE {
                            acc[i] = op(acc[i], input_chunks[lane_start][i]);
                        }

                        pos = pos.wrapping_add_signed(axis_stride);
                    }

                    let (output_chunks, _) =
                        output_buffer[o_offset..o_offset + len].as_chunks_mut::<LANE>();

                    output_chunks[lane_start].copy_from_slice(&acc);

                    lane_start += 1;
                }

                for lane_start in (chunk_len * LANE)..len {
                    let mut acc = start_value;
                    let mut pos = i_offset;

                    for _ in 0..n_size {
                        acc = op(acc, input[pos + lane_start]);

                        pos = pos.wrapping_add_signed(axis_stride);
                    }

                    output_buffer[o_offset + lane_start] = acc;
                }
            };

            if walker.is_fully_contiguous() {
                process([output_layout.offset(), input_layout.offset()]);
            } else {
                walker.for_each(process);
            }
        }
        [1, 0] => {
            walker.for_each(|offsets| {
                let o_offset = offsets[0];
                let i_offset = offsets[1];

                let mut start = i_offset;
                let mut acc = start_value;
                for _ in 0..n_size {
                    acc = op(acc, input[start]);

                    start = start.wrapping_add_signed(axis_stride);
                }

                output_buffer[o_offset..o_offset + len].fill(acc);
            });
        }
        [1, s] => {
            walker.for_each(|offsets| {
                let o_offset = offsets[0];
                let i_offset = offsets[1];

                let mut lane_start = 0;

                for i in 0..len {
                    let mut acc = start_value;
                    let mut pos = i_offset;

                    for _ in 0..n_size {
                        acc = op(acc, input[pos + lane_start]);

                        pos = pos.wrapping_add_signed(axis_stride);
                    }

                    output_buffer[o_offset + i] = acc;

                    lane_start = lane_start.wrapping_add_signed(s);
                }
            });
        }
        _ => unreachable!(),
    }
}

////////////////////////////////////////////////////

#[inline]
pub(crate) fn compute_mean<T: Numeric + FromIndex>(
    inputs: &[TensorData<T>],
    output_buffer: &mut [T],
) {
    compute_reduction(inputs, output_buffer, T::ZERO, |x, y| x + y);

    output_buffer[0] = output_buffer[0] / T::from_index(inputs[0].len());
}

#[inline]
pub fn compute_mean_axis<'a, T: Numeric + FromIndex>(
    inputs: &[TensorData<T>],
    axis: usize,
    output_buffer: &'a mut [T],
    output_layout: &Layout,
) {
    compute_reduction_axis(
        inputs,
        axis,
        output_buffer,
        output_layout,
        T::ZERO,
        |x, y| x + y,
    );

    let axis_shape = inputs[0].shape()[axis];
    for o in output_buffer {
        *o = *o / T::from_index(axis_shape);
    }
}
