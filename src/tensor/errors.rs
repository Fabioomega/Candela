#[derive(Debug)]
pub enum OpError {
    InvalidViewShape,
    NonContiguousView,
    InvalidSliceShape(usize, usize),
    OutOfBoundSlice,
    OutOfBoundAxes,
    CannotMatmul(i32, i32),
    CannotBroadcast,
    NotEnoughAxes(usize, usize),
    NotSameShape(Box<[i32]>, Box<[i32]>),
    NotSameBatch(i32, i32),
}

impl std::fmt::Display for OpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpError::InvalidViewShape => write!(
                f,
                "the view shape does not have the same size as the original shape"
            ),
            OpError::NonContiguousView => write!(
                f,
                "the view is non-contiguous. you probably want a reshape instead"
            ),
            OpError::OutOfBoundSlice => write!(
                f,
                "you cannot reference a slice that access out of bounds memory"
            ),
            OpError::InvalidSliceShape(expected, got) => write!(
                f,
                "the slice shape is bigger than the original tensor it is slicing. expected {} found {}",
                expected, got
            ),
            OpError::OutOfBoundAxes => {
                write!(f, "cannot reference out of bounds axes")
            }
            OpError::CannotMatmul(expected, got) => {
                write!(
                    f,
                    "cannot matmul. expected the row of the second tensor to be {} found {}",
                    expected, got
                )
            }
            OpError::CannotBroadcast => {
                write!(f, "cannot broadcast to that shape")
            }
            OpError::NotEnoughAxes(expected, got) => {
                write!(
                    f,
                    "there's not enough axes for this operation. expected {} found {}",
                    expected, got
                )
            }
            OpError::NotSameShape(expected, got) => {
                write!(f, "expected {:?}, but got {:?}", *expected, *got)
            }
            OpError::NotSameBatch(expected, got) => {
                write!(
                    f,
                    "tensors do not have the same batch dimension. expected {} found {}. use broadcasting if necessary",
                    expected, got
                )
            }
        }
    }
}

impl std::error::Error for OpError {}
