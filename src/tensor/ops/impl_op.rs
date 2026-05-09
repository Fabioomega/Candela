#![allow(private_bounds)]
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use crate::cfg_debug_only;
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

//////////////////////////////////////////////////////////////

fn view_impl<D>(source: &D, shape: &[usize]) -> Result<TensorPromise<D::Output>, OpError>
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

    let layout = unsafe { layout.unwrap_unchecked() };

    Ok(TensorPromise::with_layout(
        OpKind::View(layout.clone()),
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
    let layout = cont.graph.layout.view(shape);

    cfg_debug_only!({
        if let Err(err) = layout {
            return Err(err);
        }
    });

    let layout = unsafe { layout.unwrap_unchecked() };

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
    let layout = source.layout().slice(range);

    cfg_debug_only!({
        if let Err(err) = layout {
            return Err(err);
        }
    });

    let layout = unsafe { layout.unwrap_unchecked() };

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
    let layout = source.layout().transpose_axes(axes);

    cfg_debug_only!({
        if let Err(err) = layout {
            return Err(err);
        }
    });

    let layout = unsafe { layout.unwrap_unchecked() };

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
    let input = Box::new([source.create_node()]);

    unsafe { TensorPromise::new(OpKind::AsContiguous, input).unwrap_unchecked() }
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
    D::Output: ComputeWrapperSpec,
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

//////////////////////////////////////////////////////////////

fn add_tensor_impl<D1, D2>(lhs: &D1, rhs: &D2) -> TensorPromise<D1::Output>
where
    D1: ComputationDef,
    D2: ComputationDef<Output = D1::Output>,
    D1::Output: ComputeWrapperSpec,
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
    D1::Output: ComputeWrapperSpec,
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
    D1::Output: ComputeWrapperSpec,
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
    D1::Output: ComputeWrapperSpec,
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
            T: TensorElement,
        {
            type Output = TensorPromise<T>;

            #[inline]
            fn sub(self, rhs: T) -> Self::Output {
                sub_scalar_impl(self, rhs)
            }
        }

        impl<T> Sub<T> for $ty<T>
        where
            T: TensorElement,
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
            T: TensorElement,
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
