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
macro_rules! branch_duo_fast_iter {
    ($value1:expr => $name1:ident, $value2:expr => $name2:ident, $body:expr) => {
        match ($value1, $value2) {
            (
                $crate::tensor::storage::IterImpl::Contiguous($name1),
                $crate::tensor::storage::IterImpl::Contiguous($name2),
            ) => $body,
            (
                $crate::tensor::storage::IterImpl::NotContiguous($name1),
                $crate::tensor::storage::IterImpl::Contiguous($name2),
            ) => $body,
            (
                $crate::tensor::storage::IterImpl::Contiguous($name1),
                $crate::tensor::storage::IterImpl::NotContiguous($name2),
            ) => $body,
            (
                $crate::tensor::storage::IterImpl::NotContiguous($name1),
                $crate::tensor::storage::IterImpl::NotContiguous($name2),
            ) => $body,
        }
    };
}
