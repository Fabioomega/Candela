use std::ops::{Add, Div, Mul, Sub};

use crate::cfg_debug_only;
use crate::tensor::definitions::NumberLike;
use crate::tensor::errors::OpError;
use crate::tensor::graph::NodeKind;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::mem_formats::slice::SliceRange;
use crate::tensor::ops::ComputeWrapperSpec;
use crate::tensor::ops::compute_layout;
use crate::tensor::ops::def_op::{OpKind, OpKindScalar};
use crate::tensor::traits::Promising;
use crate::tensor::{CachedTensorPromise, Tensor, TensorPromise};

//////////////////////////////////////////////////////////////

trait ComputationDef {
    type Output: NumberLike;

    fn create_node(&self) -> NodeKind<Self::Output>;
    fn layout(&self) -> &Layout;
}

//////////////////////////////////////////////////////////////

fn view_impl<D>(source: &D, shape: &[i32]) -> Result<TensorPromise<D::Output>, OpError>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let input = Box::new([source.create_node()]);
    let layout = source.layout().view(shape);

    cfg_debug_only!({
        if let Err(err) = layout {
            return Err(err);
        }
    });

    Ok(TensorPromise::new(
        OpKind::View(unsafe { layout.unwrap_unchecked() }),
        input,
    ))
}

fn slice_impl<D>(source: &D, range: &[SliceRange]) -> Result<TensorPromise<D::Output>, OpError>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let input = Box::new([source.create_node()]);
    let layout = source.layout().slice(range);

    cfg_debug_only!({
        if let Err(err) = layout {
            return Err(err);
        }
    });

    Ok(TensorPromise::new(
        OpKind::Slice(unsafe { layout.unwrap_unchecked() }),
        input,
    ))
}

fn transpose_impl<D>(source: &D) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let input = Box::new([source.create_node()]);

    TensorPromise::new(OpKind::Transpose, input)
}

fn transpose_axes_impl<D>(source: &D, axes: &[usize]) -> Result<TensorPromise<D::Output>, OpError>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let input = Box::new([source.create_node()]);
    let layout = source.layout().transpose_axes(axes);

    cfg_debug_only!({
        if let Err(err) = layout {
            return Err(err);
        }
    });

    Ok(TensorPromise::new(
        OpKind::TransposeAxes(unsafe { layout.unwrap_unchecked() }),
        input,
    ))
}

fn as_contiguous_impl<D>(source: &D) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: NumberLike,
{
    let input = Box::new([source.create_node()]);

    TensorPromise::new(OpKind::AsContiguous, input)
}

//////////////////////////////////////////////////////////////

fn add_scalar_impl<D>(lhs: &D, rhs: D::Output) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: Copy + ComputeWrapperSpec,
{
    TensorPromise::new(
        OpKind::ScalarOp(OpKindScalar::Sum(rhs)),
        Box::new([lhs.create_node()]),
    )
}

fn sub_scalar_impl<D>(lhs: &D, rhs: D::Output) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: Copy + ComputeWrapperSpec,
{
    TensorPromise::new(
        OpKind::ScalarOp(OpKindScalar::Sub(rhs)),
        Box::new([lhs.create_node()]),
    )
}

fn mul_scalar_impl<D>(lhs: &D, rhs: D::Output) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: Copy + ComputeWrapperSpec,
{
    TensorPromise::new(
        OpKind::ScalarOp(OpKindScalar::Mul(rhs)),
        Box::new([lhs.create_node()]),
    )
}

fn div_scalar_impl<D>(lhs: &D, rhs: D::Output) -> TensorPromise<D::Output>
where
    D: ComputationDef,
    D::Output: Copy + ComputeWrapperSpec,
{
    TensorPromise::new(
        OpKind::ScalarOp(OpKindScalar::Div(rhs)),
        Box::new([lhs.create_node()]),
    )
}

//////////////////////////////////////////////////////////////

fn add_tensor_impl<D1, D2>(lhs: &D1, rhs: &D2) -> TensorPromise<D1::Output>
where
    D1: ComputationDef,
    D2: ComputationDef<Output = D1::Output>,
    D1::Output: Copy + ComputeWrapperSpec,
{
    let layout = compute_layout(&OpKind::<D1::Output>::Add, &[lhs.layout(), rhs.layout()]);

    if let Err(err) = layout {
        panic!("{}", err);
    }

    TensorPromise::with_layout(
        OpKind::Add,
        [lhs.create_node(), rhs.create_node()].into(),
        unsafe { layout.unwrap_unchecked() },
    )
}

fn sub_tensor_impl<D1, D2>(lhs: &D1, rhs: &D2) -> TensorPromise<D1::Output>
where
    D1: ComputationDef,
    D2: ComputationDef<Output = D1::Output>,
    D1::Output: Copy + ComputeWrapperSpec,
{
    let layout = compute_layout(&OpKind::<D1::Output>::Sub, &[lhs.layout(), rhs.layout()]);

    if let Err(err) = layout {
        panic!("{}", err);
    }

    TensorPromise::with_layout(
        OpKind::Sub,
        [lhs.create_node(), rhs.create_node()].into(),
        unsafe { layout.unwrap_unchecked() },
    )
}

fn mul_tensor_impl<D1, D2>(lhs: &D1, rhs: &D2) -> TensorPromise<D1::Output>
where
    D1: ComputationDef,
    D2: ComputationDef<Output = D1::Output>,
    D1::Output: Copy + ComputeWrapperSpec,
{
    let layout = compute_layout(&OpKind::<D1::Output>::Mul, &[lhs.layout(), rhs.layout()]);

    if let Err(err) = layout {
        panic!("{}", err);
    }

    TensorPromise::with_layout(
        OpKind::Mul,
        [lhs.create_node(), rhs.create_node()].into(),
        unsafe { layout.unwrap_unchecked() },
    )
}

fn div_tensor_impl<D1, D2>(lhs: &D1, rhs: &D2) -> TensorPromise<D1::Output>
where
    D1: ComputationDef,
    D2: ComputationDef<Output = D1::Output>,
    D1::Output: Copy + ComputeWrapperSpec,
{
    let layout = compute_layout(&OpKind::<D1::Output>::Div, &[lhs.layout(), rhs.layout()]);

    if let Err(err) = layout {
        panic!("{}", err);
    }

    TensorPromise::with_layout(
        OpKind::Div,
        [lhs.create_node(), rhs.create_node()].into(),
        unsafe { layout.unwrap_unchecked() },
    )
}

//////////////////////////////////////////////////////////////

macro_rules! impl_computation_def {
    ($ty:ident, $variant:ident) => {
        impl<T> ComputationDef for $ty<T>
        where
            T: NumberLike + ComputeWrapperSpec,
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
            T: NumberLike + ComputeWrapperSpec,
        {
            #[inline]
            pub fn view(&self, shape: &[i32]) -> Result<TensorPromise<T>, OpError> {
                view_impl(self, shape)
            }
        }
    };
}

macro_rules! impl_slice {
    ($ty:ident) => {
        impl<T> $ty<T>
        where
            T: NumberLike + ComputeWrapperSpec,
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
            T: NumberLike + ComputeWrapperSpec,
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
            T: NumberLike + ComputeWrapperSpec,
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
            T: NumberLike + ComputeWrapperSpec,
        {
            #[inline]
            pub fn as_contiguous(&self) -> TensorPromise<T> {
                as_contiguous_impl(self)
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
    };
}
//////////////////////////////////////////////////////////////

macro_rules! impl_add_scalar {
    ($ty:ident) => {
        impl<T> Add<T> for &$ty<T>
        where
            T: NumberLike + ComputeWrapperSpec,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn add(self, rhs: T) -> Self::Output {
                add_scalar_impl(self, rhs)
            }
        }

        impl<T> Add<T> for $ty<T>
        where
            T: NumberLike + ComputeWrapperSpec,
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
            T: NumberLike + ComputeWrapperSpec,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn sub(self, rhs: T) -> Self::Output {
                sub_scalar_impl(self, rhs)
            }
        }

        impl<T> Sub<T> for $ty<T>
        where
            T: NumberLike + ComputeWrapperSpec,
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
            T: NumberLike + ComputeWrapperSpec,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn mul(self, rhs: T) -> Self::Output {
                mul_scalar_impl(self, rhs)
            }
        }

        impl<T> Mul<T> for $ty<T>
        where
            T: NumberLike + ComputeWrapperSpec,
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
            T: NumberLike + ComputeWrapperSpec,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn div(self, rhs: T) -> Self::Output {
                div_scalar_impl(self, rhs)
            }
        }

        impl<T> Div<T> for $ty<T>
        where
            T: NumberLike + ComputeWrapperSpec,
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

macro_rules! impl_tensor_binop {
    ($trait:ident, $method:ident, $impl_fn:ident, $lhs:ident, $rhs:ident) => {
        impl<T> $trait<&$rhs<T>> for &$lhs<T>
        where
            T: NumberLike + ComputeWrapperSpec,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn $method(self, rhs: &$rhs<T>) -> Self::Output {
                $impl_fn(self, rhs)
            }
        }

        impl<T> $trait<$rhs<T>> for &$lhs<T>
        where
            T: NumberLike + ComputeWrapperSpec,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn $method(self, rhs: $rhs<T>) -> Self::Output {
                $impl_fn(self, &rhs)
            }
        }

        impl<T> $trait<&$rhs<T>> for $lhs<T>
        where
            T: NumberLike + ComputeWrapperSpec,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn $method(self, rhs: &$rhs<T>) -> Self::Output {
                $impl_fn(&self, rhs)
            }
        }

        impl<T> $trait<$rhs<T>> for $lhs<T>
        where
            T: NumberLike + ComputeWrapperSpec,
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

//////////////////////////////////////////////////////////////

impl_computation_def!(Tensor, Edge);
impl_computation_def!(TensorPromise, Node);
impl_computation_def!(CachedTensorPromise, Cache);

impl_reshape_like!(Tensor);
impl_reshape_like!(TensorPromise);
impl_reshape_like!(CachedTensorPromise);

impl_op_scalar!(Tensor);
impl_op_scalar!(TensorPromise);
impl_op_scalar!(CachedTensorPromise);

impl_tensor_ops!(Tensor, Tensor);
impl_tensor_ops!(Tensor, TensorPromise);
impl_tensor_ops!(Tensor, CachedTensorPromise);

impl_tensor_ops!(TensorPromise, Tensor);
impl_tensor_ops!(TensorPromise, TensorPromise);
impl_tensor_ops!(TensorPromise, CachedTensorPromise);

impl_tensor_ops!(CachedTensorPromise, Tensor);
impl_tensor_ops!(CachedTensorPromise, TensorPromise);
impl_tensor_ops!(CachedTensorPromise, CachedTensorPromise);
