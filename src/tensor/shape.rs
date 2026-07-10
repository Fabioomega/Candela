//! Shape arguments accepted by tensor constructors.
//!
//! [`IntoShape`] converts arrays, tuples, slices, a `Vec`, a bare `usize`, or an
//! existing `Box<[usize]>` into the `Box<[usize]>` a [`Layout`] is built from.
//!
//! [`Layout`]: crate::Layout

/// Conversion into the `Box<[usize]>` shape buffer a [`Layout`] is built from.
///
/// Implemented for `[usize; N]`, tuples up to arity 8, `&[usize]`, `Vec<usize>`,
/// a bare `usize` (read as a rank-1 shape), and `Box<[usize]>` itself. Tensor
/// constructors take `impl IntoShape`, so the shape can be written whichever way
/// suits the call site.
///
/// [`Layout`]: crate::Layout
///
/// # Examples
///
/// The array, tuple, and slice spellings are interchangeable:
///
/// ```
/// use candela::{Dimension, Tensor};
///
/// let a = Tensor::from_scalar(1.0_f64, [3, 3]);   // array
/// let b = Tensor::from_scalar(1.0_f64, (3, 3));   // tuple
/// let c = Tensor::from_scalar(1.0_f64, &[3, 3]);  // slice
///
/// assert_eq!(a.shape(), b.shape());
/// assert_eq!(b.shape(), c.shape());
/// ```
///
/// A bare `usize` is a rank-1 shape:
///
/// ```
/// use candela::{Dimension, Tensor};
///
/// let t = Tensor::from_scalar(1.0_f64, 3);
/// assert_eq!(t.shape(), &[3]);
/// ```
pub trait IntoShape {
    /// Materialize this value as a shape buffer.
    fn into_shape(self) -> Box<[usize]>;
}

impl IntoShape for usize {
    #[inline]
    fn into_shape(self) -> Box<[usize]> {
        Box::new([self])
    }
}

impl<const N: usize> IntoShape for [usize; N] {
    #[inline]
    fn into_shape(self) -> Box<[usize]> {
        Box::new(self)
    }
}

impl<const N: usize> IntoShape for &[usize; N] {
    #[inline]
    fn into_shape(self) -> Box<[usize]> {
        (*self).into_shape()
    }
}

impl IntoShape for &[usize] {
    #[inline]
    fn into_shape(self) -> Box<[usize]> {
        self.into()
    }
}

impl IntoShape for Vec<usize> {
    #[inline]
    fn into_shape(self) -> Box<[usize]> {
        self.into_boxed_slice()
    }
}

impl IntoShape for &Vec<usize> {
    #[inline]
    fn into_shape(self) -> Box<[usize]> {
        self.as_slice().into()
    }
}

// A shape already in its final form is moved through untouched.
impl IntoShape for Box<[usize]> {
    #[inline]
    fn into_shape(self) -> Box<[usize]> {
        self
    }
}

// Expands any identifier to the token `usize`, so the tuple macro can spell an
// N-element `(usize, .., usize)` type from a list of field bindings.
macro_rules! replace_usize {
    ($_field:ident) => {
        usize
    };
}

// Implements `IntoShape` for one tuple arity. Each argument names a field
// binding; every element type is `usize`.
macro_rules! impl_into_shape_for_tuple {
    ( $( $field:ident ),+ ) => {
        impl IntoShape for ( $( replace_usize!($field), )+ ) {
            #[inline]
            fn into_shape(self) -> Box<[usize]> {
                let ( $( $field, )+ ) = self;
                Box::new([ $( $field ),+ ])
            }
        }
    };
}

impl_into_shape_for_tuple!(a);
impl_into_shape_for_tuple!(a, b);
impl_into_shape_for_tuple!(a, b, c);
impl_into_shape_for_tuple!(a, b, c, d);
impl_into_shape_for_tuple!(a, b, c, d, e);
impl_into_shape_for_tuple!(a, b, c, d, e, f);
impl_into_shape_for_tuple!(a, b, c, d, e, f, g);
impl_into_shape_for_tuple!(a, b, c, d, e, f, g, h);

#[cfg(test)]
mod tests {
    use super::IntoShape;

    #[test]
    fn into_shape_accepted_forms() {
        let expected: Box<[usize]> = Box::new([2, 3, 4]);

        assert_eq!([2usize, 3, 4].into_shape(), expected);
        assert_eq!((&[2usize, 3, 4]).into_shape(), expected);
        assert_eq!((2usize, 3usize, 4usize).into_shape(), expected);
        assert_eq!(vec![2usize, 3, 4].into_shape(), expected);
        assert_eq!((&vec![2usize, 3, 4]).into_shape(), expected);
        assert_eq!([2usize, 3, 4].as_slice().into_shape(), expected);
    }

    #[test]
    fn into_shape_bare_usize() {
        let expected: Box<[usize]> = Box::new([5]);
        assert_eq!(5usize.into_shape(), expected);
    }

    #[test]
    fn into_shape_box() {
        let b: Box<[usize]> = Box::new([7, 8]);
        let expected: Box<[usize]> = Box::new([7, 8]);
        assert_eq!(b.into_shape(), expected);
    }
}
