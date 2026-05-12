#[macro_export]
macro_rules! debug_only {
     ($($stmt:stmt)*) => {
         #[cfg(debug_assertions)]
         { $($stmt)* }
    };
}

#[macro_export]
macro_rules! debug_shape_check {
    ($size_a: expr, $size_b: expr) => {
        debug_only!(if $size_a != $size_b {
            panic!(
                "The shape of both matrixes does not match! Expected {:?} but got {:?}",
                $size_a, $size_b
            );
        });
    };
}

#[macro_export]
macro_rules! branch_fast_iter {
    ($value:expr => $name:ident, $body:expr) => {
        match $value {
            $crate::tensor::storage::IterImpl::Contiguous($name) => $body,
            $crate::tensor::storage::IterImpl::NotContiguous($name) => $body,
        }
    };
}

#[macro_export]
macro_rules! cfg_debug_only {
    ($body:expr) => {
        #[cfg(feature = "debug_only_check")]
        {
            debug_only!($body)
        }
        #[cfg(not(feature = "debug_only_check"))]
        {
            $body
        }
    };
}

#[macro_export]
macro_rules! cfg_debug_assert {
    ($body:expr) => {
        #[cfg(feature = "debug_only_check")]
        {
            debug_assert!($body)
        }
        #[cfg(not(feature = "debug_only_check"))]
        {
            assert!($body)
        }
    };
}
