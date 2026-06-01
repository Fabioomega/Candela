#[macro_export]
macro_rules! s {
    ($($range: expr),*) => {
        &[$($crate::tensor::SliceRange::from($range)),*]
    };
}

#[macro_export]
macro_rules! zeros {
    ($shape:expr) => {
        $crate::tensor::Tensor::from_scalar(0.0, $shape)
    };
}

#[macro_export]
macro_rules! ones {
    ($shape:expr) => {
        $crate::tensor::Tensor::from_scalar(1.0, $shape)
    };
}

#[allow(private_bounds)]
pub mod arange {
    use crate::tensor::Tensor;
    use crate::tensor::backend::{ComputeFor, DefaultBackend};
    use crate::tensor::traits::FromIndex;

    /// Build a 1D tensor of evenly spaced values, NumPy-`arange` style.
    ///
    /// Every form produces a rank-1 tensor of shape `[size]`. Use `srange!`
    /// when you want the same values reshaped to an arbitrary shape in one step.
    ///
    /// - `arange!(end)` - values `0..end`, shape `[end]`.
    /// - `arange!(start, end)` - values `start..end`, shape `[end - start]`.
    /// - `arange!(start, end, step)` - values `start..end` stepping by `step`.
    ///
    /// The element type is inferred from the binding, so annotate when it is
    /// otherwise ambiguous: `let t: Tensor<f64> = arange!(4);`.
    ///
    /// # Examples
    ///
    /// ```
    /// use candela::{arange, Dimension, Tensor};
    /// let t: Tensor<f64> = arange!(2, 6); // [2.0, 3.0, 4.0, 5.0]
    /// assert_eq!(t.shape(), &[4]);
    /// assert_eq!(t.data(), &[2.0, 3.0, 4.0, 5.0]);
    /// ```
    #[macro_export]
    macro_rules! arange {
        ($size: expr) => {
            $crate::arange::_arange_default($size)
        };

        ($start: expr, $end: expr) => {
            $crate::arange::_arange_start($start, $end)
        };

        ($start: expr, $end: expr, $step: expr) => {
            $crate::arange::_arange_step($start, $end, $step)
        };
    }

    pub fn _arange_default<T: FromIndex + ComputeFor<DefaultBackend>>(size: usize) -> Tensor<T> {
        let v: Vec<T> = (0..size).map(T::from_index).collect();
        Tensor::from_vec(v, &[size])
    }

    pub fn _arange_start<T: FromIndex + ComputeFor<DefaultBackend>>(start: usize, end: usize) -> Tensor<T> {
        let v: Vec<T> = (start..end).map(T::from_index).collect();
        let size = v.len();
        Tensor::from_vec(v, &[size])
    }

    pub fn _arange_step<T: FromIndex + ComputeFor<DefaultBackend>>(start: usize, end: usize, step: usize) -> Tensor<T> {
        let v: Vec<T> = (start..end).step_by(step).map(T::from_index).collect();
        let size = v.len();
        Tensor::from_vec(v, &[size])
    }

    #[macro_export]
    macro_rules! srange {
        ($size: expr, $shape: expr) => {
            $crate::arange::_arange_default_shape($size, $shape)
        };

        ($start: expr, $end: expr, $shape: expr) => {
            $crate::arange::_arange_start_shape($start, $end, $shape)
        };

        ($start: expr, $end: expr, $step: expr, $shape: expr) => {
            $crate::arange::_arange_step_shape($start, $end, $step, $shape)
        };
    }

    pub fn _arange_default_shape<T: FromIndex + ComputeFor<DefaultBackend>>(size: usize, shape: &[usize]) -> Tensor<T> {
        let v: Vec<T> = (0..size).map(T::from_index).collect();
        Tensor::from_vec(v, shape)
    }

    pub fn _arange_start_shape<T: FromIndex + ComputeFor<DefaultBackend>>(
        start: usize,
        end: usize,
        shape: &[usize],
    ) -> Tensor<T> {
        let v: Vec<T> = (start..end).map(T::from_index).collect();
        Tensor::from_vec(v, shape)
    }

    pub fn _arange_step_shape<T: FromIndex + ComputeFor<DefaultBackend>>(
        start: usize,
        end: usize,
        step: usize,
        shape: &[usize],
    ) -> Tensor<T> {
        let v: Vec<T> = (start..end).step_by(step).map(T::from_index).collect();
        Tensor::from_vec(v, shape)
    }
}
