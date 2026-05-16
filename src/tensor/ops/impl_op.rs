#![allow(private_bounds)]
use std::iter::zip;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::tensor::definitions::NumberLike;
use crate::tensor::errors::OpError;
use crate::tensor::graph::NodeKind;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::mem_formats::slice::SliceRange;
use crate::tensor::ops::ComputeWrapperSpec;
use crate::tensor::ops::TensorElement;
use crate::tensor::ops::compute_layout;
use crate::tensor::ops::def_op::{OpKind, OpKindScalar};
use crate::tensor::traits::Promising;
use crate::tensor::{CachedTensorPromise, Tensor, TensorPromise};

//////////////////////////////////////////////////////////////

trait ComputationDef {
    type Output: TensorElement;

    fn create_node(&self) -> NodeKind<Self::Output>;
    fn layout(&self) -> &Layout;
}

struct NodeWithLayout<T: TensorElement> {
    node: NodeKind<T>,
    layout: Layout,
}

impl<T: TensorElement> ComputationDef for NodeWithLayout<T> {
    type Output = T;

    fn create_node(&self) -> NodeKind<T> {
        self.node.clone()
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }
}

//////////////////////////////////////////////////////////////

// This works for checking broadcasting in a broad sense.
// More specific check are done after the broadcast tries to happen.
#[inline]
fn find_broadcast_target(l1: &Layout, l2: &Layout) -> Vec<usize> {
    let (largest, smallest) = if l1.len() >= l2.len() {
        (l1, l2)
    } else {
        (l2, l1)
    };
    let largest_size = largest.shape().len();
    let diff = largest.shape().len() - smallest.shape().len();

    let mut new_shape = vec![0_usize; largest_size];

    for (i, (&dim1, &dim2)) in zip(l1.shape().iter().rev(), l2.shape().iter().rev()).enumerate() {
        new_shape[largest_size - i - 1] = dim1.max(dim2);
    }

    for dim in 0..diff {
        new_shape[dim] = largest.shape()[dim];
    }

    new_shape
}

#[inline]
fn find_broadcast_target_until_batch(l1: &Layout, l2: &Layout) -> Option<(Vec<usize>, Vec<usize>)> {
    let (largest, smallest) = if l1.len() >= l2.len() {
        (l1, l2)
    } else {
        (l2, l1)
    };
    let largest_size = largest.shape().len();
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
fn is_transposed_2d(shape: &[usize], stride: &[i32]) -> bool {
    if shape.len() < 2 {
        return false;
    }

    let rs = stride[stride.len() - 2];
    let cs = stride[stride.len() - 1];

    // Gives false on broadcasting
    if rs == 0 || cs == 0 {
        return false;
    }

    rs == 1
}

#[inline]
fn is_blas_ready<D>(source: &D) -> bool
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let layout = source.layout();
    layout.is_last_axes_transposed() || layout.is_contiguous()
}

#[inline]
fn apply_transform_to_pair<D1, D2, F, N1, N2, L>(
    lhs: &D1,
    rhs: &D2,
    filter: F,
    transform_l: N1,
    transform_r: N2,
    compute_output_layout: L,
) -> Result<
    (
        NodeWithLayout<D1::Output>,
        NodeWithLayout<D1::Output>,
        Layout,
    ),
    OpError,
>
where
    D1: ComputationDef,
    D2: ComputationDef<Output = D1::Output>,
    F: FnOnce(&D1, &D2) -> (bool, bool),
    N1: FnOnce(&D1) -> Result<TensorPromise<D1::Output>, OpError>,
    N2: FnOnce(&D2) -> Result<TensorPromise<D1::Output>, OpError>,
    L: FnOnce(&Layout, &Layout) -> Result<Layout, OpError>,
{
    let (apply_l, apply_r) = filter(lhs, rhs);

    let (node1, layout1, node2, layout2) = match (apply_l, apply_r) {
        (false, false) => (
            lhs.create_node(),
            lhs.layout().clone(),
            rhs.create_node(),
            rhs.layout().clone(),
        ),
        (true, false) => {
            let temp = transform_l(lhs)?;
            let layout = temp.layout().clone();
            (
                NodeKind::Node(temp.graph),
                layout,
                rhs.create_node(),
                rhs.layout().clone(),
            )
        }
        (false, true) => {
            let temp = transform_r(rhs)?;
            let layout = temp.layout().clone();
            (
                lhs.create_node(),
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

fn view_impl<D>(source: &D, shape: &[usize]) -> Result<TensorPromise<D::Output>, OpError>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let input = Box::new([source.create_node()]);
    let layout = source.layout().view(shape)?;

    Ok(TensorPromise::with_layout(
        OpKind::View(layout.clone()),
        input,
        layout,
    ))
}

fn broadcast_impl<D>(source: &D, shape: &[usize]) -> Result<TensorPromise<D::Output>, OpError>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let input = Box::new([source.create_node()]);
    let layout = source.layout().broadcast(shape)?;

    Ok(TensorPromise::with_layout(
        OpKind::Broadcast(layout.clone()),
        input,
        layout,
    ))
}

fn reshape_impl<D>(source: &D, shape: &[usize]) -> Result<TensorPromise<D::Output>, OpError>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let cont: TensorPromise<D::Output> = as_contiguous_impl(source);
    let layout = cont.graph.layout.view(shape)?;
    let input = Box::new([NodeKind::Node(cont.graph)]);

    Ok(TensorPromise::with_layout(
        OpKind::View(layout.clone()),
        input,
        layout,
    ))
}

fn slice_impl<D>(source: &D, range: &[SliceRange]) -> Result<TensorPromise<D::Output>, OpError>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let input = Box::new([source.create_node()]);
    let layout = source.layout().slice(range)?;

    Ok(TensorPromise::with_layout(
        OpKind::Slice(layout.clone()),
        input,
        layout,
    ))
}

fn transpose_impl<D>(source: &D) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let input = Box::new([source.create_node()]);

    unsafe { TensorPromise::new(OpKind::Transpose, input).unwrap_unchecked() }
}

fn transpose_axes_impl<D>(source: &D, axes: &[usize]) -> Result<TensorPromise<D::Output>, OpError>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let input = Box::new([source.create_node()]);
    let layout = source.layout().transpose_axes(axes)?;

    Ok(TensorPromise::with_layout(
        OpKind::TransposeAxes(layout.clone()),
        input,
        layout,
    ))
}

fn as_contiguous_impl<D>(source: &D) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let node = source.create_node();

    unsafe { TensorPromise::new(OpKind::AsContiguous, Box::new([node])).unwrap_unchecked() }
}

//////////////////////////////////////////////////////////////

fn add_scalar_impl<D>(lhs: &D, rhs: D::Output) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: ComputeWrapperSpec,
{
    unsafe {
        TensorPromise::new(
            OpKind::ScalarOp(OpKindScalar::AxBy(D::Output::MUL_NEUTRAL, rhs)),
            Box::new([lhs.create_node()]),
        )
        .unwrap_unchecked()
    }
}

fn sub_scalar_impl<D>(lhs: &D, rhs: D::Output) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: ComputeWrapperSpec + Neg<Output = D::Output>,
{
    unsafe {
        TensorPromise::new(
            OpKind::ScalarOp(OpKindScalar::AxBy(D::Output::MUL_NEUTRAL, -rhs)),
            Box::new([lhs.create_node()]),
        )
        .unwrap_unchecked()
    }
}

fn mul_scalar_impl<D>(lhs: &D, rhs: D::Output) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: ComputeWrapperSpec,
{
    unsafe {
        TensorPromise::new(
            OpKind::ScalarOp(OpKindScalar::AxBy(rhs, D::Output::SUM_NEUTRAL)),
            Box::new([lhs.create_node()]),
        )
        .unwrap_unchecked()
    }
}

fn div_scalar_impl<D>(lhs: &D, rhs: D::Output) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: ComputeWrapperSpec,
{
    unsafe {
        TensorPromise::new(
            OpKind::ScalarOp(OpKindScalar::AxBy(
                D::Output::MUL_NEUTRAL / rhs,
                D::Output::SUM_NEUTRAL,
            )),
            Box::new([lhs.create_node()]),
        )
        .unwrap_unchecked()
    }
}

fn exp_impl<D>(source: &D) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let input = Box::new([source.create_node()]);

    unsafe { TensorPromise::new(OpKind::ScalarOp(OpKindScalar::Exp), input).unwrap_unchecked() }
}

fn ln_impl<D>(source: &D) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let input = Box::new([source.create_node()]);

    unsafe { TensorPromise::new(OpKind::ScalarOp(OpKindScalar::Ln), input).unwrap_unchecked() }
}

fn log2_impl<D>(source: &D) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let input = Box::new([source.create_node()]);

    unsafe { TensorPromise::new(OpKind::ScalarOp(OpKindScalar::Log2), input).unwrap_unchecked() }
}

//////////////////////////////////////////////////////////////

fn add_tensor_impl<D1, D2>(lhs: &D1, rhs: &D2) -> TensorPromise<D1::Output>
where
    D1: ComputationDef,
    D2: ComputationDef<Output = D1::Output>,
    D1::Output: ComputeWrapperSpec,
{
    let target = find_broadcast_target(lhs.layout(), rhs.layout());

    let result = apply_transform_to_pair(
        lhs,
        rhs,
        |l, r| (l.layout().shape() != &target, r.layout().shape() != &target),
        |x| broadcast_impl(x, &target),
        |x| broadcast_impl(x, &target),
        |l1, l2| compute_layout(&OpKind::<D1::Output>::Add, &[l1, l2]),
    );

    if let Err(err) = result {
        panic!("{}", err);
    }

    let (lhs_b, rhs_b, layout) = unsafe { result.unwrap_unchecked() };
    TensorPromise::with_layout(
        OpKind::Add,
        [lhs_b.create_node(), rhs_b.create_node()].into(),
        layout,
    )
}

fn sub_tensor_impl<D1, D2>(lhs: &D1, rhs: &D2) -> TensorPromise<D1::Output>
where
    D1: ComputationDef,
    D2: ComputationDef<Output = D1::Output>,
    D1::Output: ComputeWrapperSpec,
{
    let target = find_broadcast_target(lhs.layout(), rhs.layout());

    let result = apply_transform_to_pair(
        lhs,
        rhs,
        |l, r| (l.layout().shape() != &target, r.layout().shape() != &target),
        |x| broadcast_impl(x, &target),
        |x| broadcast_impl(x, &target),
        |l1, l2| compute_layout(&OpKind::<D1::Output>::Sub, &[l1, l2]),
    );

    if let Err(err) = result {
        panic!("{}", err);
    }

    let (lhs_b, rhs_b, layout) = unsafe { result.unwrap_unchecked() };
    TensorPromise::with_layout(
        OpKind::Sub,
        [lhs_b.create_node(), rhs_b.create_node()].into(),
        layout,
    )
}

fn mul_tensor_impl<D1, D2>(lhs: &D1, rhs: &D2) -> TensorPromise<D1::Output>
where
    D1: ComputationDef,
    D2: ComputationDef<Output = D1::Output>,
    D1::Output: ComputeWrapperSpec,
{
    let target = find_broadcast_target(lhs.layout(), rhs.layout());

    let result = apply_transform_to_pair(
        lhs,
        rhs,
        |l, r| (l.layout().shape() != &target, r.layout().shape() != &target),
        |x| broadcast_impl(x, &target),
        |x| broadcast_impl(x, &target),
        |l1, l2| compute_layout(&OpKind::<D1::Output>::Mul, &[l1, l2]),
    );

    if let Err(err) = result {
        panic!("{}", err);
    }

    let (lhs_b, rhs_b, layout) = unsafe { result.unwrap_unchecked() };
    TensorPromise::with_layout(
        OpKind::Mul,
        [lhs_b.create_node(), rhs_b.create_node()].into(),
        layout,
    )
}

fn div_tensor_impl<D1, D2>(lhs: &D1, rhs: &D2) -> TensorPromise<D1::Output>
where
    D1: ComputationDef,
    D2: ComputationDef<Output = D1::Output>,
    D1::Output: ComputeWrapperSpec,
{
    let target = find_broadcast_target(lhs.layout(), rhs.layout());

    let result = apply_transform_to_pair(
        lhs,
        rhs,
        |l, r| (l.layout().shape() != &target, r.layout().shape() != &target),
        |x| broadcast_impl(x, &target),
        |x| broadcast_impl(x, &target),
        |l1, l2| compute_layout(&OpKind::<D1::Output>::Div, &[l1, l2]),
    );

    if let Err(err) = result {
        panic!("{}", err);
    }

    let (lhs_b, rhs_b, layout) = unsafe { result.unwrap_unchecked() };
    TensorPromise::with_layout(
        OpKind::Div,
        [lhs_b.create_node(), rhs_b.create_node()].into(),
        layout,
    )
}

//////////////////////////////////////////////////////////////

fn matmul_tensor_impl<D1, D2>(lhs: &D1, rhs: &D2) -> Result<TensorPromise<D1::Output>, OpError>
where
    D1: ComputationDef,
    D2: ComputationDef<Output = D1::Output>,
    D1::Output: ComputeWrapperSpec,
{
    let target = find_broadcast_target_until_batch(lhs.layout(), rhs.layout());

    let (lhs_b, rhs_b, _) = apply_transform_to_pair(
        lhs,
        rhs,
        |l, r| {
            (
                target
                    .as_ref()
                    .map_or(false, |target| l.layout().shape() != target.0),
                target
                    .as_ref()
                    .map_or(false, |target| r.layout().shape() != target.1),
            )
        },
        |x| broadcast_impl(x, unsafe { &target.as_ref().unwrap_unchecked().0 }),
        |x| broadcast_impl(x, unsafe { &target.as_ref().unwrap_unchecked().1 }),
        |l1, _| Ok(l1.clone()),
    )?;

    let (lhs_c, rhs_c, layout) = apply_transform_to_pair(
        &lhs_b,
        &rhs_b,
        |l, r| (!is_blas_ready(l), !is_blas_ready(r)),
        |x| Ok(as_contiguous_impl(x)),
        |x| Ok(as_contiguous_impl(x)),
        |l1, l2| {
            compute_layout(
                &OpKind::<D1::Output>::MatMul(D1::Output::MUL_NEUTRAL),
                &[l1, l2],
            )
        },
    )?;

    Ok(TensorPromise::with_layout(
        OpKind::MatMul(D1::Output::MUL_NEUTRAL),
        [lhs_c.create_node(), rhs_c.create_node()].into(),
        layout,
    ))
}

//////////////////////////////////////////////////////////////

macro_rules! impl_computation_def {
    ($ty:ident, $variant:ident) => {
        impl<T> ComputationDef for $ty<T>
        where
            T: TensorElement,
        {
            type Output = T;

            fn create_node(&self) -> NodeKind<T> {
                NodeKind::$variant(self.graph.clone())
            }

            fn layout(&self) -> &Layout {
                self.graph.layout()
            }
        }
    };
}

//////////////////////////////////////////////////////////////

macro_rules! impl_view {
    ($ty:ident) => {
        impl<T> $ty<T>
        where
            T: TensorElement,
        {
            #[inline]
            pub fn view(&self, shape: &[usize]) -> Result<TensorPromise<T>, OpError> {
                view_impl(self, shape)
            }

            #[inline]
            pub fn reshape(&self, shape: &[usize]) -> Result<TensorPromise<T>, OpError> {
                reshape_impl(self, shape)
            }
        }
    };
}

macro_rules! impl_slice {
    ($ty:ident) => {
        impl<T> $ty<T>
        where
            T: TensorElement,
        {
            #[inline]
            pub fn slice(&self, shape: &[SliceRange]) -> Result<TensorPromise<T>, OpError> {
                slice_impl(self, shape)
            }
        }
    };
}

macro_rules! impl_transpose {
    ($ty: ident) => {
        impl<T> $ty<T>
        where
            T: TensorElement,
        {
            #[inline]
            pub fn transpose(&self) -> TensorPromise<T> {
                transpose_impl(self)
            }
        }
    };
}

macro_rules! impl_transpose_axes {
    ($ty:ident) => {
        impl<T> $ty<T>
        where
            T: TensorElement,
        {
            #[inline]
            pub fn transpose_axes(&self, axes: &[usize]) -> Result<TensorPromise<T>, OpError> {
                transpose_axes_impl(self, axes)
            }
        }
    };
}

macro_rules! impl_as_contiguous {
    ($ty: ident) => {
        impl<T> $ty<T>
        where
            T: TensorElement,
        {
            #[inline]
            pub fn as_contiguous(&self) -> TensorPromise<T> {
                as_contiguous_impl(self)
            }
        }
    };
}

macro_rules! impl_broadcast {
    ($ty:ident) => {
        impl<T> $ty<T>
        where
            T: TensorElement,
        {
            #[inline]
            pub fn broadcast(&self, shape: &[usize]) -> Result<TensorPromise<T>, OpError> {
                broadcast_impl(self, shape)
            }
        }
    };
}

macro_rules! impl_exp {
    ($ty:ident) => {
        impl<T> $ty<T>
        where
            T: TensorElement,
        {
            #[inline]
            pub fn exp(&self) -> TensorPromise<T> {
                exp_impl(self)
            }
        }
    };
}

macro_rules! impl_ln {
    ($ty:ident) => {
        impl<T> $ty<T>
        where
            T: TensorElement,
        {
            #[inline]
            pub fn ln(&self) -> TensorPromise<T> {
                ln_impl(self)
            }
        }
    };
}

macro_rules! impl_log2 {
    ($ty:ident) => {
        impl<T> $ty<T>
        where
            T: TensorElement,
        {
            #[inline]
            pub fn log2(&self) -> TensorPromise<T> {
                log2_impl(self)
            }
        }
    };
}

macro_rules! impl_unary_scalar_ops {
    ($ty:ident) => {
        impl_exp!($ty);
        impl_ln!($ty);
        impl_log2!($ty);
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
        impl<T> Add<T> for &$ty<T>
        where
            T: TensorElement,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn add(self, rhs: T) -> Self::Output {
                add_scalar_impl(self, rhs)
            }
        }

        impl<T> Add<T> for $ty<T>
        where
            T: TensorElement,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn add(self, rhs: T) -> Self::Output {
                (&self).add(rhs)
            }
        }
    };
}

macro_rules! impl_sub_scalar {
    ($ty:ident) => {
        impl<T> Sub<T> for &$ty<T>
        where
            T: TensorElement + Neg<Output = T>,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn sub(self, rhs: T) -> Self::Output {
                sub_scalar_impl(self, rhs)
            }
        }

        impl<T> Sub<T> for $ty<T>
        where
            T: TensorElement + Neg<Output = T>,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn sub(self, rhs: T) -> Self::Output {
                (&self).sub(rhs)
            }
        }
    };
}

macro_rules! impl_mul_scalar {
    ($ty:ident) => {
        impl<T> Mul<T> for &$ty<T>
        where
            T: TensorElement,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn mul(self, rhs: T) -> Self::Output {
                mul_scalar_impl(self, rhs)
            }
        }

        impl<T> Mul<T> for $ty<T>
        where
            T: TensorElement,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn mul(self, rhs: T) -> Self::Output {
                (&self).mul(rhs)
            }
        }
    };
}

macro_rules! impl_div_scalar {
    ($ty:ident) => {
        impl<T> Div<T> for &$ty<T>
        where
            T: TensorElement,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn div(self, rhs: T) -> Self::Output {
                div_scalar_impl(self, rhs)
            }
        }

        impl<T> Div<T> for $ty<T>
        where
            T: TensorElement,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn div(self, rhs: T) -> Self::Output {
                (&self).div(rhs)
            }
        }
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
        impl<T> AddAssign<T> for $ty<T>
        where
            T: TensorElement,
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
        impl<T> SubAssign<T> for $ty<T>
        where
            T: TensorElement + Neg<Output = T>,
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
        impl<T> MulAssign<T> for $ty<T>
        where
            T: TensorElement,
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
        impl<T> DivAssign<T> for $ty<T>
        where
            T: TensorElement,
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
        impl<T> $trait<&$rhs<T>> for &$lhs<T>
        where
            T: TensorElement,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn $method(self, rhs: &$rhs<T>) -> Self::Output {
                $impl_fn(self, rhs)
            }
        }

        impl<T> $trait<$rhs<T>> for &$lhs<T>
        where
            T: TensorElement,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn $method(self, rhs: $rhs<T>) -> Self::Output {
                $impl_fn(self, &rhs)
            }
        }

        impl<T> $trait<&$rhs<T>> for $lhs<T>
        where
            T: TensorElement,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn $method(self, rhs: &$rhs<T>) -> Self::Output {
                $impl_fn(&self, rhs)
            }
        }

        impl<T> $trait<$rhs<T>> for $lhs<T>
        where
            T: TensorElement,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn $method(self, rhs: $rhs<T>) -> Self::Output {
                $impl_fn(&self, &rhs)
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

macro_rules! impl_matmul {
    ($ty:ident) => {
        impl<T> $ty<T>
        where
            T: TensorElement,
        {
            #[inline]
            pub fn matmul<D>(&self, rhs: &D) -> Result<TensorPromise<T>, OpError>
            where
                D: ComputationDef<Output = T>,
            {
                matmul_tensor_impl(self, rhs)
            }
        }
    };
}

//////////////////////////////////////////////////////////////

macro_rules! impl_tensor_assign_binop {
    ($trait:ident, $method:ident, $impl_fn:ident, $rhs:ident) => {
        impl<T> $trait<$rhs<T>> for TensorPromise<T>
        where
            T: TensorElement,
        {
            #[inline]
            fn $method(&mut self, rhs: $rhs<T>) {
                *self = $impl_fn(&*self, &rhs);
            }
        }

        impl<T> $trait<&$rhs<T>> for TensorPromise<T>
        where
            T: TensorElement,
        {
            #[inline]
            fn $method(&mut self, rhs: &$rhs<T>) {
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

impl_computation_def!(Tensor, Edge);
impl_computation_def!(TensorPromise, Node);
impl_computation_def!(CachedTensorPromise, Cache);

impl_reshape_like!(Tensor);
impl_reshape_like!(TensorPromise);
impl_reshape_like!(CachedTensorPromise);

impl_unary_scalar_ops!(Tensor);
impl_unary_scalar_ops!(TensorPromise);
impl_unary_scalar_ops!(CachedTensorPromise);

impl_op_scalar!(Tensor);
impl_op_scalar!(TensorPromise);
impl_op_scalar!(CachedTensorPromise);

impl_matmul!(Tensor);
impl_matmul!(TensorPromise);
impl_matmul!(CachedTensorPromise);

impl_tensor_ops!(Tensor, Tensor);
impl_tensor_ops!(Tensor, TensorPromise);
impl_tensor_ops!(Tensor, CachedTensorPromise);

impl_tensor_ops!(TensorPromise, Tensor);
impl_tensor_ops!(TensorPromise, TensorPromise);
impl_tensor_ops!(TensorPromise, CachedTensorPromise);

impl_tensor_ops!(CachedTensorPromise, Tensor);
impl_tensor_ops!(CachedTensorPromise, TensorPromise);
impl_tensor_ops!(CachedTensorPromise, CachedTensorPromise);

impl_op_assign_scalar!(TensorPromise);

impl_tensor_assign_ops!(Tensor);
impl_tensor_assign_ops!(TensorPromise);
impl_tensor_assign_ops!(CachedTensorPromise);
