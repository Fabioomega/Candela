use std::{
    fmt::Debug,
    iter::Cloned,
    ops::{Add, Div, Mul, Sub},
};

use crate::tensor::iter::{ChunkedSliceIter, SliceIter};

pub(crate) type ChunkedIter<'a, T> = ChunkedSliceIter<Cloned<SliceIter<'a, T>>, T>;

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
