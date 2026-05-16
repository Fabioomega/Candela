use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

use crate::tensor::iter::{ChunkedSliceIter, CopiedSliceIter};

pub(crate) type ChunkedIter<'a, T> =
    ChunkedSliceIter<CopiedSliceIter<'a, T>, T, { crate::tensor::PACKING_BUFFER_SIZE }>;

pub trait NumberLike:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialEq
    + Default
    + Debug
{
}

impl<T> NumberLike for T where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + PartialEq
        + Default
        + Debug
{
}
