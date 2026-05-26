#[derive(Debug)]
pub enum OpError {
    InvalidViewShape,
    NonContiguousView,
    InvalidSliceShape(usize, usize),
    SliceOutOfBounds,
    IndexOutOfBounds,
    AxesOutOfBounds,
    CannotMatMul(usize, usize),
    CannotBroadcast,
    NotEnoughAxes(usize, usize),
    NotSameShape(Box<[usize]>, Box<[usize]>),
    NotSameBatch(usize, usize),
    ZeroRankShape,
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
            OpError::SliceOutOfBounds => write!(
                f,
                "you cannot reference a slice that access out of bounds memory"
            ),
            OpError::IndexOutOfBounds => write!(f, "you cannot reference out of bounds memory",),
            OpError::InvalidSliceShape(expected, got) => write!(
                f,
                "the slice shape is bigger than the original tensor it is slicing. expected {} found {}",
                expected, got
            ),
            OpError::AxesOutOfBounds => {
                write!(f, "cannot reference out of bounds axes")
            }
            OpError::CannotMatMul(expected, got) => {
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
            OpError::ZeroRankShape => {
                write!(
                    f,
                    "tensor shape must have rank >= 1 (empty shape `&[]` / 0-D tensors are not supported)"
                )
            }
        }
    }
}

impl std::error::Error for OpError {}
