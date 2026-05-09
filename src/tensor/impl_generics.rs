#[macro_export]
macro_rules! impl_display {
    ($struct_name: ty) => {
        use $crate::tensor::iter::StepInfo;

        #[allow(private_bounds)]
        impl<T: std::fmt::Display + $crate::tensor::ops::ComputeWrapperSpec> std::fmt::Display for $struct_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut indent = 0;
                let mut in_seq = false;

                let last = self.shape().len() - 1;

                for step in self.informed_iter() {
                    match step {
                        StepInfo::EnterDimension(dim) => {
                            write!(f, "{:indent$}[", "", indent = indent)?;
                            indent += 2;

                            if dim != last {
                                write!(f, "\n")?;
                            }
                        }
                        StepInfo::ExitDimension(dim) => {
                            indent -= 2;
                            in_seq = false;

                            if dim != last {
                                write!(f, "{:indent$}", "", indent = indent)?;
                            }

                            write!(f, "]\n")?;
                        }
                        StepInfo::Value(v) => {
                            if in_seq {
                                write!(f, ", ")?;
                            }

                            write!(f, "{:>4}", v)?;

                            in_seq = true;
                        }
                        _ => {}
                    }
                }

                Ok(())
            }
        }
    };
}

// #[macro_export]
// macro_rules! impl_index {
//     ($struct_name: ty) => {
//         impl<T> Index<&[i32]> for $struct_name {
//             type Output = T;

//             fn index(&self, index: &[i32]) -> &Self::Output {
//                 debug_assert_eq!(self.layout.stride.len(), index.len());

//                 let mut pos: i32 = 0;

//                 for (stride, i) in zip(&self.layout.stride, index) {
//                     pos += *stride * *i;
//                 }

//                 debug_assert_positive!(pos);

//                 self.index_memory(pos as usize)
//             }
//         }
//     };
// }

// #[macro_export]
// macro_rules! impl_index {
//     ($struct_name: ty) => {
//         impl<T: Copy> Index<&[i32]> for $struct_name {
//             type Output = T;

//             fn index(&self, index: &[i32]) -> &Self::Output {
//                 debug_assert_eq!(self.layout.stride.len(), index.len());

//                 let mut pos: i32 = 0;

//                 for (stride, i) in zip(self.stride(), index) {
//                     pos += *stride * *i;
//                 }

//                 debug_assert_positive!(pos);

//                 self.index_memory(pos as usize)
//             }
//         }
//     };
// }
