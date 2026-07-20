use std::iter::zip;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::tensor::backend::Backend;
use crate::tensor::errors::OpError;
use crate::tensor::graph::NodeKind;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::mem_formats::slice::SliceRange;
use crate::tensor::ops::capabilities::{CanMatMul, FloatLike, NumericOp};
use crate::tensor::ops::compute_layout;
use crate::tensor::ops::def_op::{OpKind, OpKindScalar};
use crate::tensor::shape::IntoShape;
use crate::tensor::skeleton::{
    BakedPromise, BinaryResult, Clean, SkeletonPromise, SkeletonSlot, Tainting, UnaryResult,
};
use crate::tensor::traits::{Dimension, Numeric, Operand};
use crate::tensor::{CachedTensorPromise, Tensor, TensorPromise};

//////////////////////////////////////////////////////////////

struct NodeWithLayout<T: Numeric, B: Backend> {
    node: NodeKind<T, B>,
    layout: Layout,
}

impl<T: Numeric, B: Backend> Dimension for NodeWithLayout<T, B> {
    fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl<T: Numeric, B: Backend> Operand<T, B> for NodeWithLayout<T, B> {
    fn to_node(&self) -> NodeKind<T, B> {
        self.node.clone()
    }
}

impl<T: Numeric, B: Backend> Tainting for NodeWithLayout<T, B> {
    type Mark = Clean;
}

//////////////////////////////////////////////////////////////

// This works for checking broadcasting in a broad sense.
// More specific check are done after the broadcast tries to happen.
#[inline]
fn find_broadcast_target(l1: &Layout, l2: &Layout) -> Vec<usize> {
    let (largest, smallest) = if l1.shape().len() >= l2.shape().len() {
        (l1, l2)
    } else {
        (l2, l1)
    };
    let largest_size = largest.shape().len();
    debug_assert!(
        largest.shape().len() >= smallest.shape().len(),
        "broadcast helper precondition violated"
    );
    let diff = largest.shape().len() - smallest.shape().len();

    let mut new_shape = vec![0_usize; largest_size];

    for (i, (&dim1, &dim2)) in zip(l1.shape().iter().rev(), l2.shape().iter().rev()).enumerate() {
        new_shape[largest_size - i - 1] = dim1.max(dim2);
    }

    new_shape[..diff].copy_from_slice(&largest.shape()[..diff]);

    new_shape
}

#[inline]
fn find_broadcast_target_until_batch(l1: &Layout, l2: &Layout) -> Option<(Vec<usize>, Vec<usize>)> {
    let (largest, smallest) = if l1.shape().len() >= l2.shape().len() {
        (l1, l2)
    } else {
        (l2, l1)
    };
    let largest_size = largest.shape().len();
    debug_assert!(
        largest.shape().len() >= smallest.shape().len(),
        "matmul-batch broadcast helper precondition violated"
    );
    let diff = largest.shape().len() - smallest.shape().len();
    let smallest_diff: usize = 2.min(smallest.shape().len());

    if largest_size <= 2 {
        return None;
    }

    let mut new_l1_shape = vec![0_usize; largest_size];
    let mut new_l2_shape = vec![0_usize; largest_size];

    for dim in 0..smallest_diff {
        new_l1_shape[largest_size - dim - 1] = l1.shape()[l1.shape().len() - dim - 1];
        new_l2_shape[largest_size - dim - 1] = l2.shape()[l2.shape().len() - dim - 1];
    }

    for (i, (&dim1, &dim2)) in zip(l1.shape().iter().rev(), l2.shape().iter().rev())
        .enumerate()
        .skip(smallest_diff)
    {
        let max = dim1.max(dim2);
        new_l1_shape[largest_size - i - 1] = max;
        new_l2_shape[largest_size - i - 1] = max;
    }

    for dim in 0..diff {
        let n = largest.shape()[dim];
        new_l1_shape[dim] = n;
        new_l2_shape[dim] = n;
    }

    Some((new_l1_shape, new_l2_shape))
}

#[inline]
fn is_blas_ready<T, B, D>(source: &D) -> bool
where
    B: Backend,
    D: Operand<T, B>,
{
    if B::SUPPORTS_NON_CONTIGUOUS_MATMUL {
        return true;
    }

    // We don't need to check if it's contiguous because it will become a zero-copy or removed if not necessary
    // by either the planner or the fusion system.
    // layout.is_contiguous() ||
    B::SUPPORTS_2D_TRANSPOSED_MATMUL && source.layout().is_last_axes_transposed()
}

type NodeTransform<Output, Backend> = Result<
    (
        NodeWithLayout<Output, Backend>,
        NodeWithLayout<Output, Backend>,
        Layout,
    ),
    OpError,
>;

#[inline]
fn apply_transform_to_pair<T, B, D1, D2, F, N1, N2, L>(
    lhs: &D1,
    rhs: &D2,
    filter: F,
    transform_l: N1,
    transform_r: N2,
    compute_output_layout: L,
) -> NodeTransform<T, B>
where
    T: Numeric,
    B: Backend,
    D1: Operand<T, B>,
    D2: Operand<T, B>,
    F: FnOnce(&D1, &D2) -> (bool, bool),
    N1: FnOnce(&D1) -> Result<TensorPromise<T, B>, OpError>,
    N2: FnOnce(&D2) -> Result<TensorPromise<T, B>, OpError>,
    L: FnOnce(&Layout, &Layout) -> Result<Layout, OpError>,
{
    let (apply_l, apply_r) = filter(lhs, rhs);

    let (node1, layout1, node2, layout2) = match (apply_l, apply_r) {
        (false, false) => (
            lhs.to_node(),
            lhs.layout().clone(),
            rhs.to_node(),
            rhs.layout().clone(),
        ),
        (true, false) => {
            let temp = transform_l(lhs)?;
            let layout = temp.layout().clone();
            (
                NodeKind::Node(temp.graph),
                layout,
                rhs.to_node(),
                rhs.layout().clone(),
            )
        }
        (false, true) => {
            let temp = transform_r(rhs)?;
            let layout = temp.layout().clone();
            (
                lhs.to_node(),
                lhs.layout().clone(),
                NodeKind::Node(temp.graph),
                layout,
            )
        }
        (true, true) => {
            let temp1 = transform_l(lhs)?;
            let layout1 = temp1.layout().clone();
            let temp2 = transform_r(rhs)?;
            let layout2 = temp2.layout().clone();
            (
                NodeKind::Node(temp1.graph),
                layout1,
                NodeKind::Node(temp2.graph),
                layout2,
            )
        }
    };

    let layout = compute_output_layout(&layout1, &layout2)?;
    Ok((
        NodeWithLayout {
            node: node1,
            layout: layout1,
        },
        NodeWithLayout {
            node: node2,
            layout: layout2,
        },
        layout,
    ))
}

//////////////////////////////////////////////////////////////

fn view_impl<T, B, D>(source: &D, shape: &[usize]) -> Result<TensorPromise<T, B>, OpError>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input: Box<[NodeKind<T, B>; 1]> = Box::new([source.to_node()]);
    let layout = source.layout().view(shape)?.with_offset(0);

    Ok(TensorPromise::with_layout(OpKind::View, input, layout))
}

fn broadcast_impl<T, B, D>(source: &D, shape: &[usize]) -> Result<TensorPromise<T, B>, OpError>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);
    let layout = source.layout().broadcast(shape)?.with_offset(0);

    Ok(TensorPromise::with_layout(OpKind::Broadcast, input, layout))
}

fn reshape_impl<T, B, D>(source: &D, shape: &[usize]) -> Result<TensorPromise<T, B>, OpError>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let cont: TensorPromise<T, B> = as_contiguous_impl(source);
    let layout = cont.graph.layout.view(shape)?.with_offset(0);
    let input = Box::new([NodeKind::Node(cont.graph)]);

    Ok(TensorPromise::with_layout(OpKind::View, input, layout))
}

fn slice_impl<T, B, D>(source: &D, range: &[SliceRange]) -> Result<TensorPromise<T, B>, OpError>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);
    let layout = source.layout().clone().with_offset(0).slice(range)?;

    Ok(TensorPromise::with_layout(OpKind::Slice, input, layout))
}

fn transpose_impl<T, B, D>(source: &D) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);
    let layout = source.layout().transpose().with_offset(0);

    TensorPromise::with_layout(OpKind::Transpose, input, layout)
}

fn transpose_axes_impl<T, B, D>(source: &D, axes: &[usize]) -> Result<TensorPromise<T, B>, OpError>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);
    let layout = source.layout().transpose_axes(axes)?.with_offset(0);

    Ok(TensorPromise::with_layout(
        OpKind::TransposeAxes,
        input,
        layout,
    ))
}

fn as_contiguous_impl<T, B, D>(source: &D) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let node = source.to_node();
    unsafe { TensorPromise::new(OpKind::AsContiguous, Box::new([node])).unwrap_unchecked() }
}

//////////////////////////////////////////////////////////////

fn add_scalar_impl<T, B, D>(lhs: &D, rhs: T) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    unsafe {
        TensorPromise::new(
            OpKind::ScalarOp(OpKindScalar::AxBy(T::MUL_NEUTRAL, rhs)),
            Box::new([lhs.to_node()]),
        )
        .unwrap_unchecked()
    }
}

fn sub_scalar_impl<T, B, D>(lhs: &D, rhs: T) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
    T: Numeric + Neg<Output = T>,
{
    unsafe {
        TensorPromise::new(
            OpKind::ScalarOp(OpKindScalar::AxBy(T::MUL_NEUTRAL, -rhs)),
            Box::new([lhs.to_node()]),
        )
        .unwrap_unchecked()
    }
}

fn mul_scalar_impl<T, B, D>(lhs: &D, rhs: T) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    unsafe {
        TensorPromise::new(
            OpKind::ScalarOp(OpKindScalar::AxBy(rhs, T::SUM_NEUTRAL)),
            Box::new([lhs.to_node()]),
        )
        .unwrap_unchecked()
    }
}

fn div_scalar_impl<T, B, D>(lhs: &D, rhs: T) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    if rhs == T::SUM_NEUTRAL {
        panic!("cannot divide by zero. stop.")
    }

    unsafe {
        TensorPromise::new(
            OpKind::ScalarOp(OpKindScalar::AxBy(T::MUL_NEUTRAL / rhs, T::SUM_NEUTRAL)),
            Box::new([lhs.to_node()]),
        )
        .unwrap_unchecked()
    }
}

fn exp_impl<T, B, D>(source: &D) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);

    unsafe { TensorPromise::new(OpKind::ScalarOp(OpKindScalar::Exp), input).unwrap_unchecked() }
}

fn ln_impl<T, B, D>(source: &D) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);

    unsafe { TensorPromise::new(OpKind::ScalarOp(OpKindScalar::Ln), input).unwrap_unchecked() }
}

fn log2_impl<T, B, D>(source: &D) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);

    unsafe { TensorPromise::new(OpKind::ScalarOp(OpKindScalar::Log2), input).unwrap_unchecked() }
}

fn relu_impl<T, B, D>(source: &D) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);

    unsafe { TensorPromise::new(OpKind::ScalarOp(OpKindScalar::ReLU), input).unwrap_unchecked() }
}

fn tanh_impl<T, B, D>(source: &D) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);

    unsafe { TensorPromise::new(OpKind::ScalarOp(OpKindScalar::Tanh), input).unwrap_unchecked() }
}

//////////////////////////////////////////////////////////////

fn add_tensor_impl<T, B, D1, D2>(lhs: &D1, rhs: &D2) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D1: Operand<T, B>,
    D2: Operand<T, B>,
{
    let target = find_broadcast_target(lhs.layout(), rhs.layout());

    let result = apply_transform_to_pair(
        lhs,
        rhs,
        |l, r| (l.layout().shape() != target, r.layout().shape() != target),
        |x| broadcast_impl(x, &target),
        |x| broadcast_impl(x, &target),
        |l1, l2| compute_layout(&OpKind::<T>::Add, &[l1, l2]),
    );

    if let Err(err) = result {
        panic!("{}", err);
    }

    let (lhs_b, rhs_b, layout) = unsafe { result.unwrap_unchecked() };
    TensorPromise::with_layout(
        OpKind::Add,
        [lhs_b.to_node(), rhs_b.to_node()].into(),
        layout,
    )
}

fn sub_tensor_impl<T, B, D1, D2>(lhs: &D1, rhs: &D2) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D1: Operand<T, B>,
    D2: Operand<T, B>,
{
    let target = find_broadcast_target(lhs.layout(), rhs.layout());

    let result = apply_transform_to_pair(
        lhs,
        rhs,
        |l, r| (l.layout().shape() != target, r.layout().shape() != target),
        |x| broadcast_impl(x, &target),
        |x| broadcast_impl(x, &target),
        |l1, l2| compute_layout(&OpKind::<T>::Sub, &[l1, l2]),
    );

    if let Err(err) = result {
        panic!("{}", err);
    }

    let (lhs_b, rhs_b, layout) = unsafe { result.unwrap_unchecked() };
    TensorPromise::with_layout(
        OpKind::Sub,
        [lhs_b.to_node(), rhs_b.to_node()].into(),
        layout,
    )
}

fn mul_tensor_impl<T, B, D1, D2>(lhs: &D1, rhs: &D2) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D1: Operand<T, B>,
    D2: Operand<T, B>,
{
    let target = find_broadcast_target(lhs.layout(), rhs.layout());

    let result = apply_transform_to_pair(
        lhs,
        rhs,
        |l, r| (l.layout().shape() != target, r.layout().shape() != target),
        |x| broadcast_impl(x, &target),
        |x| broadcast_impl(x, &target),
        |l1, l2| compute_layout(&OpKind::<T>::Mul, &[l1, l2]),
    );

    if let Err(err) = result {
        panic!("{}", err);
    }

    let (lhs_b, rhs_b, layout) = unsafe { result.unwrap_unchecked() };
    TensorPromise::with_layout(
        OpKind::Mul,
        [lhs_b.to_node(), rhs_b.to_node()].into(),
        layout,
    )
}

fn div_tensor_impl<T, B, D1, D2>(lhs: &D1, rhs: &D2) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D1: Operand<T, B>,
    D2: Operand<T, B>,
{
    let target = find_broadcast_target(lhs.layout(), rhs.layout());

    let result = apply_transform_to_pair(
        lhs,
        rhs,
        |l, r| (l.layout().shape() != target, r.layout().shape() != target),
        |x| broadcast_impl(x, &target),
        |x| broadcast_impl(x, &target),
        |l1, l2| compute_layout(&OpKind::<T>::Div, &[l1, l2]),
    );

    if let Err(err) = result {
        panic!("{}", err);
    }

    let (lhs_b, rhs_b, layout) = unsafe { result.unwrap_unchecked() };
    TensorPromise::with_layout(
        OpKind::Div,
        [lhs_b.to_node(), rhs_b.to_node()].into(),
        layout,
    )
}

//////////////////////////////////////////////////////////////

fn matmul_core<T, B, D1, D2>(lhs: &D1, rhs: &D2) -> Result<TensorPromise<T, B>, OpError>
where
    T: Numeric,
    B: Backend,
    D1: Operand<T, B>,
    D2: Operand<T, B>,
{
    let (lhs_c, rhs_c, _) = apply_transform_to_pair(
        lhs,
        rhs,
        |l, r| (!is_blas_ready(l), !is_blas_ready(r)),
        |x| Ok(as_contiguous_impl(x)),
        |x| Ok(as_contiguous_impl(x)),
        |l1, _| Ok(l1.clone()),
    )?;

    let target = find_broadcast_target_until_batch(lhs_c.layout(), rhs_c.layout());

    let (lhs_b, rhs_b, layout) = apply_transform_to_pair(
        &lhs_c,
        &rhs_c,
        |l, r| {
            (
                target
                    .as_ref()
                    .is_some_and(|target| l.layout().shape() != target.0),
                target
                    .as_ref()
                    .is_some_and(|target| r.layout().shape() != target.1),
            )
        },
        |x| broadcast_impl(x, unsafe { &target.as_ref().unwrap_unchecked().0 }),
        |x| broadcast_impl(x, unsafe { &target.as_ref().unwrap_unchecked().1 }),
        |l1, l2| compute_layout(&OpKind::<T>::MatMul(T::MUL_NEUTRAL), &[l1, l2]),
    )?;

    Ok(TensorPromise::with_layout(
        OpKind::MatMul(T::MUL_NEUTRAL),
        [lhs_b.to_node(), rhs_b.to_node()].into(),
        layout,
    ))
}

// Drop the dim at position `len - 1 - from_end` via a metadata-only View.
// Used by matmul's 1-D promotion to strip the size-1 dim introduced by
// promoting a vector operand to a matrix.
fn drop_dim_from_end<T, B, D>(source: &D, from_end: usize) -> Result<TensorPromise<T, B>, OpError>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let mut new_shape: Vec<usize> = source.layout().shape().to_vec();
    new_shape.remove(new_shape.len() - 1 - from_end);
    view_impl(source, &new_shape)
}

fn matmul_tensor_impl<T, B, D1, D2>(lhs: &D1, rhs: &D2) -> Result<TensorPromise<T, B>, OpError>
where
    T: Numeric,
    B: Backend,
    D1: Operand<T, B>,
    D2: Operand<T, B>,
{
    match (lhs.layout().shape().len(), rhs.layout().shape().len()) {
        // [K] @ [K] -> [1, K] @ [K, 1] = [1, 1], strip to [1].
        (1, 1) => {
            let lhs_p = reshape_impl(lhs, &[1, lhs.layout().shape()[0]])?;
            let rhs_p = reshape_impl(rhs, &[rhs.layout().shape()[0], 1])?;
            let result = matmul_core(&lhs_p, &rhs_p)?;
            drop_dim_from_end(&result, 0)
        }
        // [K] @ [..., K, N] -> [1, K] @ [..., K, N] = [..., 1, N], drop the prepended 1.
        (1, _) => {
            let lhs_p = reshape_impl(lhs, &[1, lhs.layout().shape()[0]])?;
            let result = matmul_core(&lhs_p, rhs)?;
            drop_dim_from_end(&result, 1)
        }
        // [..., M, K] @ [K] -> [..., M, K] @ [K, 1] = [..., M, 1], drop the appended 1.
        (_, 1) => {
            let rhs_p = reshape_impl(rhs, &[rhs.layout().shape()[0], 1])?;
            let result = matmul_core(lhs, &rhs_p)?;
            drop_dim_from_end(&result, 0)
        }
        // Both already >= 2-D (construction gate rules out 0-D): straight matmul.
        _ => matmul_core(lhs, rhs),
    }
}

//////////////////////////////////////////////////////////////

fn sum_impl<T, B, D>(source: &D) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);

    unsafe { TensorPromise::new(OpKind::Sum, input).unwrap_unchecked() }
}

fn sum_axis_impl<T, B, D>(
    source: &D,
    axis: isize,
    keep_dims: bool,
) -> Result<TensorPromise<T, B>, OpError>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);
    let op = OpKind::<T>::SumAxis(axis, keep_dims);
    let layout = compute_layout(&op, &[source.layout()])?;

    Ok(TensorPromise::with_layout(op, input, layout))
}

fn max_impl<T, B, D>(source: &D) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);

    unsafe { TensorPromise::new(OpKind::Max, input).unwrap_unchecked() }
}

fn max_axis_impl<T, B, D>(
    source: &D,
    axis: isize,
    keep_dims: bool,
) -> Result<TensorPromise<T, B>, OpError>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);
    let op = OpKind::<T>::MaxAxis(axis, keep_dims);
    let layout = compute_layout(&op, &[source.layout()])?;

    Ok(TensorPromise::with_layout(op, input, layout))
}

fn mean_impl<T, B, D>(source: &D) -> TensorPromise<T, B>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);

    unsafe { TensorPromise::new(OpKind::Mean, input).unwrap_unchecked() }
}

fn mean_axis_impl<T, B, D>(
    source: &D,
    axis: isize,
    keep_dims: bool,
) -> Result<TensorPromise<T, B>, OpError>
where
    T: Numeric,
    B: Backend,
    D: Operand<T, B>,
{
    let input = Box::new([source.to_node()]);
    let op = OpKind::<T>::MeanAxis(axis, keep_dims);
    let layout = compute_layout(&op, &[source.layout()])?;

    Ok(TensorPromise::with_layout(op, input, layout))
}

//////////////////////////////////////////////////////////////

macro_rules! impl_view {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: Numeric,
            B: Backend,
        {
            /// Reinterprets the tensor's data as having shape `shape` without allocating.
            ///
            /// The tensor must be contiguous and must have the same length as the original tensor.
            /// Use [`.reshape()`][Self::reshape] if the tensor may not be contiguous.
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::{Tensor, Dimension};
            ///
            /// let t = Tensor::from_slice(&[4.0, 3.0, 2.0, 1.0], &[4]);
            /// // Shares the same underlying data as t
            /// let v = t.view(&[2, 2]).unwrap().materialize();
            ///
            /// assert_eq!(v.shape(), &[2, 2]);
            /// ```
            ///
            /// # Errors
            ///
            /// Returns [`OpError::NonContiguousView`] if the tensor is not contiguous or [`OpError::InvalidViewShape`] if the shape is invalid.
            #[inline]
            pub fn view(
                &self,
                shape: impl IntoShape,
            ) -> Result<<$ty<T, B> as UnaryResult<T, B>>::Output, OpError> {
                let (rank, shape) = shape.into_shape();
                view_impl(self, &shape[..rank]).map(<$ty<T, B> as UnaryResult<T, B>>::wrap)
            }

            /// Reinterprets the tensor's data as having shape `shape`,
            /// allocating if the tensor is not contiguous.
            ///
            /// Unlike [`.view()`][Self::view], this never fails on a non-contiguous
            /// tensor — it only requires that the new shape has the same total number
            /// of elements as the original.
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::{Tensor, Dimension};
            ///
            /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
            /// let r = t.transpose().reshape(&[4]).unwrap().materialize();
            ///
            /// assert_eq!(r.shape(), &[4]);
            /// // r is contiguous as it was allocated in a new buffer.
            /// assert!(r.is_contiguous());
            /// ```
            ///
            /// # Errors
            ///
            /// Returns [`OpError::InvalidViewShape`] if `shape` does not have the same
            /// total number of elements as the original.
            #[inline]
            pub fn reshape(
                &self,
                shape: impl IntoShape,
            ) -> Result<<$ty<T, B> as UnaryResult<T, B>>::Output, OpError> {
                let (rank, shape) = shape.into_shape();
                reshape_impl(self, &shape[..rank]).map(<$ty<T, B> as UnaryResult<T, B>>::wrap)
            }
        }
    };
}

macro_rules! impl_slice {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: Numeric,
            B: Backend,
        {
            /// Selects a rectangular subregion of the tensor, without allocating.
            ///
            /// Each [`SliceRange`] picks a range along one axis, applied from the
            /// outermost axis inward; axes you leave out are kept whole. Build the
            /// ranges with the [`s!`] macro using ordinary range syntax — negative
            /// bounds count from the end. The result is a view into the original buffer.
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::{Tensor, Dimension, s};
            ///
            /// let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[2, 3]);
            /// let sub = t.slice(s![1..2, 0..2]).unwrap().materialize(); // row 1, cols 0..2
            ///
            /// assert_eq!(sub.shape(), &[1, 2]);
            /// ```
            ///
            /// # Errors
            ///
            /// Returns [`OpError::AxesOutOfBounds`] if more ranges are given than the
            /// tensor has axes, or [`OpError::SliceOutOfBounds`] if a range is empty.
            #[inline]
            pub fn slice(
                &self,
                shape: &[SliceRange],
            ) -> Result<<$ty<T, B> as UnaryResult<T, B>>::Output, OpError> {
                slice_impl(self, shape).map(<$ty<T, B> as UnaryResult<T, B>>::wrap)
            }
        }
    };
}

macro_rules! impl_transpose {
    ($ty: ident) => {
        impl<T, B> $ty<T, B>
        where
            T: Numeric,
            B: Backend,
        {
            /// Reverses the order of every axis, without allocating.
            ///
            /// For a 2-D tensor this is the familiar matrix transpose; for higher
            /// ranks it flips all axes at once (axis `i` becomes axis `rank - 1 - i`).
            /// Only the layout changes — the data stays put until something forces a
            /// copy, so reach for [`.as_contiguous()`][Self::as_contiguous] when you
            /// need the transposed values in their own buffer. For an arbitrary
            /// permutation, see [`.transpose_axes()`][Self::transpose_axes].
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::{Tensor, Dimension};
            ///
            /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
            /// let tt = t.transpose().materialize();
            ///
            /// assert_eq!(tt.shape(), &[3, 2]);
            /// ```
            #[inline]
            pub fn transpose(&self) -> <$ty<T, B> as UnaryResult<T, B>>::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(transpose_impl(self))
            }
        }
    };
}

macro_rules! impl_transpose_axes {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: Numeric,
            B: Backend,
        {
            /// Reorders the axes by an explicit permutation, without allocating.
            ///
            /// `axes` must list every axis index exactly once: `transpose_axes(&[1, 0])`
            /// is the plain 2-D [`.transpose()`][Self::transpose], while `&[0, 2, 1]`
            /// swaps only the last two axes of a rank-3 tensor and leaves the first
            /// alone. Like [`.transpose()`][Self::transpose] it only relabels the
            /// layout — see [`.as_contiguous()`][Self::as_contiguous] to materialize
            /// the reordered values.
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::{Tensor, Dimension};
            ///
            /// let t = Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &[1, 2, 3]);
            /// let s = t.transpose_axes(&[0, 2, 1]).unwrap().materialize();
            ///
            /// assert_eq!(s.shape(), &[1, 3, 2]);
            /// ```
            ///
            /// # Errors
            ///
            /// Returns [`OpError::NotEnoughAxes`] if `axes` doesn't have one entry per
            /// axis, or [`OpError::AxesOutOfBounds`] if an index is out of range or
            /// repeated (so the list isn't a valid permutation).
            #[inline]
            pub fn transpose_axes(
                &self,
                axes: impl IntoShape,
            ) -> Result<<$ty<T, B> as UnaryResult<T, B>>::Output, OpError> {
                let (rank, axes) = axes.into_shape();
                transpose_axes_impl(self, &axes[..rank])
                    .map(<$ty<T, B> as UnaryResult<T, B>>::wrap)
            }
        }
    };
}

macro_rules! impl_as_contiguous {
    ($ty: ident) => {
        impl<T, B> $ty<T, B>
        where
            T: Numeric,
            B: Backend,
        {
            /// Packs the tensor into a fresh contiguous buffer in row-major order.
            ///
            /// Layout-only ops like [`.transpose()`][Self::transpose] and
            /// [`.slice()`][Self::slice] leave the data where it is and only change
            /// how it's addressed. `as_contiguous` turns such a view back into a
            /// densely laid-out tensor. If the input is already contiguous it costs
            /// nothing — the call collapses to a no-op. Candela also inserts it
            /// automatically wherever an op needs contiguous memory (a BLAS matmul, a
            /// [`.reshape()`][Self::reshape]), so you rarely call it by hand.
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::{Tensor, Dimension};
            ///
            /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
            /// // transpose is a view; as_contiguous lays the transposed values out for real.
            /// let c = t.transpose().as_contiguous().materialize();
            ///
            /// assert!(c.is_contiguous());
            /// assert_eq!(c.data(), &[1.0, 3.0, 2.0, 4.0]);
            /// ```
            #[inline]
            pub fn as_contiguous(&self) -> <$ty<T, B> as UnaryResult<T, B>>::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(as_contiguous_impl(self))
            }
        }
    };
}

macro_rules! impl_broadcast {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: Numeric,
            B: Backend,
        {
            /// Expands the tensor to a larger shape by repeating elements along new
            /// or size-1 axes, without allocating.
            ///
            /// Broadcasting follows NumPy's right-aligned rules: a target axis must
            /// either match the source or expand from size 1, and extra leading axes
            /// are added on the left. No data is copied — the repeated axes are faked
            /// with zero strides. The arithmetic operators broadcast on their own, so
            /// you mostly need this only to force a specific shape up front.
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::{Tensor, Dimension};
            ///
            /// let row = Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]);
            /// let b = row.broadcast(&[2, 3]).unwrap().materialize();
            ///
            /// assert_eq!(b.shape(), &[2, 3]);
            /// ```
            ///
            /// # Errors
            ///
            /// Returns [`OpError::CannotBroadcast`] if the target shape has fewer axes
            /// than the source, or an axis is neither equal to the source nor
            /// expandable from 1.
            #[inline]
            pub fn broadcast(
                &self,
                shape: impl IntoShape,
            ) -> Result<<$ty<T, B> as UnaryResult<T, B>>::Output, OpError> {
                let (rank, shape) = shape.into_shape();
                broadcast_impl(self, &shape[..rank])
                    .map(<$ty<T, B> as UnaryResult<T, B>>::wrap)
            }
        }
    };
}

macro_rules! impl_reshape_like {
    ($ty:ident) => {
        impl_view!($ty);
        impl_slice!($ty);
        impl_transpose!($ty);
        impl_transpose_axes!($ty);
        impl_as_contiguous!($ty);
        impl_broadcast!($ty);
    };
}
//////////////////////////////////////////////////////////////

macro_rules! impl_add_scalar {
    ($ty:ident) => {
        impl<T, B> Add<T> for &$ty<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            type Output = <$ty<T, B> as UnaryResult<T, B>>::Output;

            #[inline]
            fn add(self, rhs: T) -> Self::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(add_scalar_impl(self, rhs))
            }
        }

        impl<T, B> Add<T> for $ty<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            type Output = <$ty<T, B> as UnaryResult<T, B>>::Output;

            #[inline]
            fn add(self, rhs: T) -> Self::Output {
                (&self).add(rhs)
            }
        }
    };
}

macro_rules! impl_sub_scalar {
    ($ty:ident) => {
        impl<T, B> Sub<T> for &$ty<T, B>
        where
            T: NumericOp + Neg<Output = T>,
            B: Backend,
        {
            type Output = <$ty<T, B> as UnaryResult<T, B>>::Output;

            #[inline]
            fn sub(self, rhs: T) -> Self::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(sub_scalar_impl(self, rhs))
            }
        }

        impl<T, B> Sub<T> for $ty<T, B>
        where
            T: NumericOp + Neg<Output = T>,
            B: Backend,
        {
            type Output = <$ty<T, B> as UnaryResult<T, B>>::Output;

            #[inline]
            fn sub(self, rhs: T) -> Self::Output {
                (&self).sub(rhs)
            }
        }
    };
}

macro_rules! impl_mul_scalar {
    ($ty:ident) => {
        impl<T, B> Mul<T> for &$ty<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            type Output = <$ty<T, B> as UnaryResult<T, B>>::Output;

            #[inline]
            fn mul(self, rhs: T) -> Self::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(mul_scalar_impl(self, rhs))
            }
        }

        impl<T, B> Mul<T> for $ty<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            type Output = <$ty<T, B> as UnaryResult<T, B>>::Output;

            #[inline]
            fn mul(self, rhs: T) -> Self::Output {
                (&self).mul(rhs)
            }
        }
    };
}

macro_rules! impl_div_scalar {
    ($ty:ident) => {
        impl<T, B> Div<T> for &$ty<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            type Output = <$ty<T, B> as UnaryResult<T, B>>::Output;

            /// # Panics
            ///
            /// Panics when the operator is applied — not at `.materialize()` — if
            /// `rhs` is zero.
            #[inline]
            fn div(self, rhs: T) -> Self::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(div_scalar_impl(self, rhs))
            }
        }

        impl<T, B> Div<T> for $ty<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            type Output = <$ty<T, B> as UnaryResult<T, B>>::Output;

            /// # Panics
            ///
            /// Panics when the operator is applied — not at `.materialize()` — if
            /// `rhs` is zero.
            #[inline]
            fn div(self, rhs: T) -> Self::Output {
                (&self).div(rhs)
            }
        }
    };
}

macro_rules! impl_exp {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: FloatLike,
            B: Backend,
        {
            /// Computes `e^x` for each element.
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::Tensor;
            ///
            /// let t = Tensor::from_slice(&[0.0_f64], &[1]);
            /// assert_eq!(t.exp().materialize().data(), &[1.0]); // e^0 == 1
            /// ```
            #[inline]
            pub fn exp(&self) -> <$ty<T, B> as UnaryResult<T, B>>::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(exp_impl(self))
            }
        }
    };
}

macro_rules! impl_ln {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: FloatLike,
            B: Backend,
        {
            /// Computes the natural logarithm of each element.
            ///
            /// Elements `<= 0` follow the platform `ln` behavior: `-inf` at zero, `NaN` below.
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::Tensor;
            ///
            /// let t = Tensor::from_slice(&[1.0_f64], &[1]);
            /// assert_eq!(t.ln().materialize().data(), &[0.0]); // ln(1) == 0
            /// ```
            #[inline]
            pub fn ln(&self) -> <$ty<T, B> as UnaryResult<T, B>>::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(ln_impl(self))
            }
        }
    };
}

macro_rules! impl_log2 {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: FloatLike,
            B: Backend,
        {
            /// Computes the base-2 logarithm of each element.
            ///
            /// Elements `<= 0` follow the platform `log2` behavior: `-inf` at zero, `NaN` below.
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::Tensor;
            ///
            /// let t = Tensor::from_slice(&[8.0_f64], &[1]);
            /// assert_eq!(t.log2().materialize().data(), &[3.0]); // log2(8) == 3
            /// ```
            #[inline]
            pub fn log2(&self) -> <$ty<T, B> as UnaryResult<T, B>>::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(log2_impl(self))
            }
        }
    };
}

macro_rules! impl_relu {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: FloatLike,
            B: Backend,
        {
            /// Applies the rectified linear unit (`relu`): `max(x, 0.0)` for each element.
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::Tensor;
            ///
            /// // Negative values clamp to zero; non-negative values pass through.
            /// let t = Tensor::from_slice(&[-2.0_f64, -0.5, 0.0, 1.5], &[4]);
            /// assert_eq!(t.relu().materialize().data(), &[0.0, 0.0, 0.0, 1.5]);
            /// ```
            #[inline]
            pub fn relu(&self) -> <$ty<T, B> as UnaryResult<T, B>>::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(relu_impl(self))
            }
        }
    };
}

macro_rules! impl_tanh {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: FloatLike,
            B: Backend,
        {
            /// Applies the hyperbolic tangent to each element, mapping values into `(-1, 1)`.
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::Tensor;
            ///
            /// let t = Tensor::from_scalar(0.0_f64, &[1]);
            /// assert_eq!(*t.tanh().materialize().item(), 0.0); // tanh(0) == 0
            /// ```
            #[inline]
            pub fn tanh(&self) -> <$ty<T, B> as UnaryResult<T, B>>::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(tanh_impl(self))
            }
        }
    };
}

macro_rules! impl_unary_scalar_ops {
    ($ty:ident) => {
        impl_exp!($ty);
        impl_ln!($ty);
        impl_log2!($ty);
        impl_relu!($ty);
        impl_tanh!($ty);
    };
}

macro_rules! impl_op_scalar {
    ($ty:ident) => {
        impl_add_scalar!($ty);
        impl_sub_scalar!($ty);
        impl_div_scalar!($ty);
        impl_mul_scalar!($ty);
    };
}

//////////////////////////////////////////////////////////////

macro_rules! impl_add_assign_scalar {
    ($ty:ident) => {
        impl<T, B> AddAssign<T> for $ty<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            #[inline]
            fn add_assign(&mut self, rhs: T) {
                *self = add_scalar_impl(&*self, rhs);
            }
        }
    };
}

macro_rules! impl_sub_assign_scalar {
    ($ty:ident) => {
        impl<T, B> SubAssign<T> for $ty<T, B>
        where
            T: NumericOp + Neg<Output = T>,
            B: Backend,
        {
            #[inline]
            fn sub_assign(&mut self, rhs: T) {
                *self = sub_scalar_impl(&*self, rhs);
            }
        }
    };
}

macro_rules! impl_mul_assign_scalar {
    ($ty:ident) => {
        impl<T, B> MulAssign<T> for $ty<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            #[inline]
            fn mul_assign(&mut self, rhs: T) {
                *self = mul_scalar_impl(&*self, rhs);
            }
        }
    };
}

macro_rules! impl_div_assign_scalar {
    ($ty:ident) => {
        impl<T, B> DivAssign<T> for $ty<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            #[inline]
            fn div_assign(&mut self, rhs: T) {
                *self = div_scalar_impl(&*self, rhs);
            }
        }
    };
}

macro_rules! impl_op_assign_scalar {
    ($ty:ident) => {
        impl_add_assign_scalar!($ty);
        impl_sub_assign_scalar!($ty);
        impl_mul_assign_scalar!($ty);
        impl_div_assign_scalar!($ty);
    };
}

//////////////////////////////////////////////////////////////

macro_rules! impl_tensor_binop {
    ($trait:ident, $method:ident, $impl_fn:ident, $lhs:ident, $rhs:ident) => {
        impl<T, B> $trait<&$rhs<T, B>> for &$lhs<T, B>
        where
            T: NumericOp,
            B: Backend,
            $lhs<T, B>: BinaryResult<$rhs<T, B>, T, B>,
        {
            type Output = <$lhs<T, B> as BinaryResult<$rhs<T, B>, T, B>>::Output;

            /// Applies the operation element-wise, broadcasting if the shapes are compatible.
            ///
            /// A `SkeletonSlot` anywhere in either operand makes the result a
            /// [`SkeletonPromise`] instead of a [`TensorPromise`].
            ///
            /// # Panics
            ///
            /// Panics when the operator is applied — not at `.materialize()` — if the
            /// shapes are not broadcast-compatible.
            #[inline]
            fn $method(self, rhs: &$rhs<T, B>) -> Self::Output {
                <$lhs<T, B> as BinaryResult<$rhs<T, B>, T, B>>::wrap($impl_fn(self, rhs))
            }
        }

        impl<T, B> $trait<$rhs<T, B>> for &$lhs<T, B>
        where
            T: NumericOp,
            B: Backend,
            $lhs<T, B>: BinaryResult<$rhs<T, B>, T, B>,
        {
            type Output = <$lhs<T, B> as BinaryResult<$rhs<T, B>, T, B>>::Output;

            /// Applies the operation element-wise, broadcasting if the shapes are compatible.
            ///
            /// # Panics
            ///
            /// Panics when the operator is applied — not at `.materialize()` — if the
            /// shapes are not broadcast-compatible.
            #[inline]
            fn $method(self, rhs: $rhs<T, B>) -> Self::Output {
                <$lhs<T, B> as BinaryResult<$rhs<T, B>, T, B>>::wrap($impl_fn(self, &rhs))
            }
        }

        impl<T, B> $trait<&$rhs<T, B>> for $lhs<T, B>
        where
            T: NumericOp,
            B: Backend,
            $lhs<T, B>: BinaryResult<$rhs<T, B>, T, B>,
        {
            type Output = <$lhs<T, B> as BinaryResult<$rhs<T, B>, T, B>>::Output;

            /// Applies the operation element-wise, broadcasting if the shapes are compatible.
            ///
            /// # Panics
            ///
            /// Panics when the operator is applied — not at `.materialize()` — if the
            /// shapes are not broadcast-compatible.
            #[inline]
            fn $method(self, rhs: &$rhs<T, B>) -> Self::Output {
                <$lhs<T, B> as BinaryResult<$rhs<T, B>, T, B>>::wrap($impl_fn(&self, rhs))
            }
        }

        impl<T, B> $trait<$rhs<T, B>> for $lhs<T, B>
        where
            T: NumericOp,
            B: Backend,
            $lhs<T, B>: BinaryResult<$rhs<T, B>, T, B>,
        {
            type Output = <$lhs<T, B> as BinaryResult<$rhs<T, B>, T, B>>::Output;

            /// Applies the operation element-wise, broadcasting if the shapes are compatible.
            ///
            /// # Panics
            ///
            /// Panics when the operator is applied — not at `.materialize()` — if the
            /// shapes are not broadcast-compatible.
            #[inline]
            fn $method(self, rhs: $rhs<T, B>) -> Self::Output {
                <$lhs<T, B> as BinaryResult<$rhs<T, B>, T, B>>::wrap($impl_fn(&self, &rhs))
            }
        }
    };
}

macro_rules! impl_tensor_ops {
    ($lhs:ident, $rhs:ident) => {
        impl_tensor_binop!(Add, add, add_tensor_impl, $lhs, $rhs);
        impl_tensor_binop!(Sub, sub, sub_tensor_impl, $lhs, $rhs);
        impl_tensor_binop!(Mul, mul, mul_tensor_impl, $lhs, $rhs);
        impl_tensor_binop!(Div, div, div_tensor_impl, $lhs, $rhs);
    };
}

// Cross product: every operand against every operand. The taint algebra
// (`BinaryResult`) decides each cell's output type, so a `SkeletonSlot` on
// either side yields a `SkeletonPromise`.
macro_rules! impl_tensor_ops_cross {
    ([$($ty:ident),+ $(,)?]) => {
        impl_tensor_ops_cross!(@rows [$($ty),+] [$($ty),+]);
    };
    (@rows [$($lhs:ident),+] $rhs:tt) => {
        $( impl_tensor_ops_cross!(@row $lhs $rhs); )+
    };
    (@row $lhs:ident [$($rhs:ident),+]) => {
        $( impl_tensor_ops!($lhs, $rhs); )+
    };
}

//////////////////////////////////////////////////////////////

macro_rules! impl_matmul {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: CanMatMul,
            B: Backend,
        {
            /// Matrix-multiplies, following NumPy's `matmul` conventions.
            ///
            /// For two 2-D tensors, it's as one would expect: `[m, k] @ [k, n]`
            /// gives `[m, n]`, and the inner dimension `k` of both must agree.
            ///
            /// Higher ranks (3-D, 4-D, and so on) are treated as batches of 2-D
            /// matrices, broadcasting the leading axes where needed. So `[b, m, k] @ [k, n]`
            /// gives `[b, m, n]`, because it's the same as doing `[b, m, k] @ [b, k, n]`.
            ///
            /// A 1-D operand is promoted to 2-D for the operation, then the added
            /// axis is dropped from the result:
            /// - `[k] @ [k]` contracts to a one-element tensor (a dot product).
            /// - `[k] @ [.., k, n]` gives `[.., n]` (vector times matrix).
            /// - `[.., m, k] @ [k]` gives `[.., m]` (matrix times vector).
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::{Tensor, Dimension};
            ///
            /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
            /// let b = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0], &[3, 2]);
            /// let c = a.matmul(&b).unwrap().materialize();
            ///
            /// assert_eq!(c.shape(), &[2, 2]);
            /// assert_eq!(c.data(), &[4.0, 2.0, 10.0, 5.0]);
            /// ```
            ///
            /// A 1-D right-hand side contracts the last axis away:
            ///
            /// ```
            /// use candela::{Tensor, Dimension};
            ///
            /// let m = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
            /// let v = Tensor::from_slice(&[1.0, 1.0], &[2]);
            /// let r = m.matmul(&v).unwrap().materialize();
            ///
            /// assert_eq!(r.shape(), &[2]);
            /// assert_eq!(r.data(), &[3.0, 7.0]);
            /// ```
            ///
            /// # Errors
            ///
            /// Returns [`OpError::CannotMatMul`] if the inner dimensions don't agree, or
            /// [`OpError::CannotBroadcast`] if the batch axes aren't broadcast-compatible.
            #[inline]
            pub fn matmul<D>(
                &self,
                rhs: &D,
            ) -> Result<<$ty<T, B> as BinaryResult<D, T, B>>::Output, OpError>
            where
                D: Operand<T, B>,
                $ty<T, B>: BinaryResult<D, T, B>,
            {
                matmul_tensor_impl(self, rhs).map(<$ty<T, B> as BinaryResult<D, T, B>>::wrap)
            }
        }
    };
}

//////////////////////////////////////////////////////////////

macro_rules! impl_sum {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            /// Sums every element, producing a one-element tensor.
            ///
            /// To reduce along a single axis instead, see
            /// [`.sum_axis()`][Self::sum_axis].
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::Tensor;
            ///
            /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
            /// assert_eq!(t.sum().materialize().data(), &[10.0]);
            /// ```
            #[inline]
            pub fn sum(&self) -> <$ty<T, B> as UnaryResult<T, B>>::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(sum_impl(self))
            }
        }
    };
}

macro_rules! impl_sum_axis {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            /// Sums along a single axis.
            ///
            /// `axis` selects the axis to collapse and may be negative to count from
            /// the end. With `keep_dims = false` that axis is removed from the shape;
            /// with `keep_dims = true` it is kept as a size-1 axis, which leaves the
            /// result broadcastable against the input. To reduce the whole tensor,
            /// see [`.sum()`][Self::sum].
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::{Tensor, Dimension};
            ///
            /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
            ///
            /// let dropped = t.sum_axis(0, false).unwrap().materialize();
            /// assert_eq!(dropped.shape(), &[3]);
            /// assert_eq!(dropped.data(), &[5.0, 7.0, 9.0]);
            ///
            /// // keep_dims = true leaves a size-1 axis in place.
            /// let kept = t.sum_axis(0, true).unwrap().materialize();
            /// assert_eq!(kept.shape(), &[1, 3]);
            /// ```
            ///
            /// # Errors
            ///
            /// Returns [`OpError::AxesOutOfBounds`] if `axis` is outside the tensor's rank.
            #[inline]
            pub fn sum_axis(
                &self,
                axis: isize,
                keep_dims: bool,
            ) -> Result<<$ty<T, B> as UnaryResult<T, B>>::Output, OpError> {
                sum_axis_impl(self, axis, keep_dims).map(<$ty<T, B> as UnaryResult<T, B>>::wrap)
            }
        }
    };
}

macro_rules! impl_max {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            /// Returns the largest element, as a one-element tensor.
            ///
            /// To take the maximum along a single axis instead, see
            /// [`.max_axis()`][Self::max_axis].
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::Tensor;
            ///
            /// let t = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 2.0], &[2, 3]);
            /// assert_eq!(t.max().materialize().data(), &[5.0]);
            /// ```
            #[inline]
            pub fn max(&self) -> <$ty<T, B> as UnaryResult<T, B>>::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(max_impl(self))
            }
        }
    };
}

macro_rules! impl_max_axis {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            /// Takes the maximum along a single axis.
            ///
            /// `axis` and `keep_dims` behave exactly as in
            /// [`.sum_axis()`][Self::sum_axis]. To reduce the whole tensor, see
            /// [`.max()`][Self::max].
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::{Tensor, Dimension};
            ///
            /// let t = Tensor::from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 2.0], &[2, 3]);
            /// let m = t.max_axis(1, false).unwrap().materialize();
            ///
            /// assert_eq!(m.shape(), &[2]);
            /// assert_eq!(m.data(), &[4.0, 5.0]);
            /// ```
            ///
            /// # Errors
            ///
            /// Returns [`OpError::AxesOutOfBounds`] if `axis` is outside the tensor's rank.
            #[inline]
            pub fn max_axis(
                &self,
                axis: isize,
                keep_dims: bool,
            ) -> Result<<$ty<T, B> as UnaryResult<T, B>>::Output, OpError> {
                max_axis_impl(self, axis, keep_dims).map(<$ty<T, B> as UnaryResult<T, B>>::wrap)
            }
        }
    };
}

macro_rules! impl_mean {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: FloatLike,
            B: Backend,
        {
            /// Averages every element, producing a one-element tensor.
            ///
            /// To average along a single axis instead, see
            /// [`.mean_axis()`][Self::mean_axis].
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::Tensor;
            ///
            /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
            /// assert_eq!(t.mean().materialize().data(), &[2.5]);
            /// ```
            #[inline]
            pub fn mean(&self) -> <$ty<T, B> as UnaryResult<T, B>>::Output {
                <$ty<T, B> as UnaryResult<T, B>>::wrap(mean_impl(self))
            }
        }
    };
}

macro_rules! impl_mean_axis {
    ($ty:ident) => {
        impl<T, B> $ty<T, B>
        where
            T: FloatLike,
            B: Backend,
        {
            /// Averages along a single axis.
            ///
            /// `axis` and `keep_dims` behave exactly as in
            /// [`.sum_axis()`][Self::sum_axis]. To average the whole tensor, see
            /// [`.mean()`][Self::mean].
            ///
            /// # Examples
            ///
            /// ```
            /// use candela::{Tensor, Dimension};
            ///
            /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
            /// let m = t.mean_axis(1, false).unwrap().materialize();
            ///
            /// assert_eq!(m.shape(), &[2]);
            /// assert_eq!(m.data(), &[2.0, 5.0]);
            /// ```
            ///
            /// # Errors
            ///
            /// Returns [`OpError::AxesOutOfBounds`] if `axis` is outside the tensor's rank.
            #[inline]
            pub fn mean_axis(
                &self,
                axis: isize,
                keep_dims: bool,
            ) -> Result<<$ty<T, B> as UnaryResult<T, B>>::Output, OpError> {
                mean_axis_impl(self, axis, keep_dims).map(<$ty<T, B> as UnaryResult<T, B>>::wrap)
            }
        }
    };
}

//////////////////////////////////////////////////////////////

macro_rules! impl_tensor_assign_binop {
    ($trait:ident, $method:ident, $impl_fn:ident, $rhs:ident) => {
        impl<T, B> $trait<$rhs<T, B>> for TensorPromise<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            #[inline]
            fn $method(&mut self, rhs: $rhs<T, B>) {
                *self = $impl_fn(&*self, &rhs);
            }
        }

        impl<T, B> $trait<&$rhs<T, B>> for TensorPromise<T, B>
        where
            T: NumericOp,
            B: Backend,
        {
            #[inline]
            fn $method(&mut self, rhs: &$rhs<T, B>) {
                *self = $impl_fn(&*self, rhs);
            }
        }
    };
}

macro_rules! impl_tensor_assign_ops {
    ($rhs:ident) => {
        impl_tensor_assign_binop!(AddAssign, add_assign, add_tensor_impl, $rhs);
        impl_tensor_assign_binop!(SubAssign, sub_assign, sub_tensor_impl, $rhs);
        impl_tensor_assign_binop!(MulAssign, mul_assign, mul_tensor_impl, $rhs);
        impl_tensor_assign_binop!(DivAssign, div_assign, div_tensor_impl, $rhs);
    };
}

//////////////////////////////////////////////////////////////

macro_rules! impl_all_ops {
    ($ty:ident) => {
        impl_reshape_like!($ty);
        impl_unary_scalar_ops!($ty);
        impl_op_scalar!($ty);
        impl_matmul!($ty);
        impl_sum!($ty);
        impl_sum_axis!($ty);
        impl_max!($ty);
        impl_max_axis!($ty);
        impl_mean!($ty);
        impl_mean_axis!($ty);
    };
}

impl_all_ops!(Tensor);
impl_all_ops!(TensorPromise);
impl_all_ops!(CachedTensorPromise);
impl_all_ops!(BakedPromise);
impl_all_ops!(SkeletonSlot);
impl_all_ops!(SkeletonPromise);

impl_tensor_ops_cross!([
    Tensor,
    TensorPromise,
    CachedTensorPromise,
    BakedPromise,
    SkeletonSlot,
    SkeletonPromise,
]);

impl_op_assign_scalar!(TensorPromise);

impl_tensor_assign_ops!(Tensor);
impl_tensor_assign_ops!(TensorPromise);
impl_tensor_assign_ops!(CachedTensorPromise);
impl_tensor_assign_ops!(BakedPromise);
