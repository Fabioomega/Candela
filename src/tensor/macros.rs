#[macro_export]
macro_rules! branch_fast_iter {
    ($value:expr => $name:ident, $body:expr) => {
        match $value {
            $crate::tensor::storage::IterImpl::Contiguous($name) => $body,
            $crate::tensor::storage::IterImpl::NotContiguous($name) => $body,
        }
    };
}
