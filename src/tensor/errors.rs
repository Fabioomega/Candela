/// The error returned by fallible tensor operations.
///
/// Operations that can fail at runtime — `view`, `slice`, `matmul`, the axis
/// reductions, and so on — return `Result<_, OpError>`. The error is produced
/// when the operation is built, not at `.materialize()`, so a bad shape is
/// caught at the call site rather than deep inside execution. `OpError`
/// implements [`Error`](std::error::Error) and [`Display`](std::fmt::Display),
/// so it composes with `?` and `Box<dyn Error>`.
#[derive(Debug)]
#[non_exhaustive]
pub enum OpError {
    /// A `view` was requested with a shape whose element count differs from the original.
    InvalidViewShape,
    /// A `view` was requested on a non-contiguous tensor; use `reshape` instead.
    NonContiguousView,
    /// A slice resolved to more elements than the tensor holds. Carries `(tensor_len, slice_len)`.
    InvalidSliceShape(usize, usize),
    /// A slice range is empty — its end is not past its start.
    SliceOutOfBounds,
    /// An index passed to `get` is past the end of its axis.
    IndexOutOfBounds,
    /// An axis index is out of range, repeated, or there are more axes than the tensor has.
    AxesOutOfBounds,
    /// The inner dimensions of a `matmul` don't agree. Carries the two mismatched sizes.
    CannotMatMul(usize, usize),
    /// The shapes (or a `broadcast` target) aren't broadcast-compatible.
    CannotBroadcast,
    /// An operation received the wrong number of axes or indices. Carries `(expected, got)`.
    NotEnoughAxes(usize, usize),
    /// Two tensors in an elementwise op have incompatible shapes. Carries both shapes.
    NotSameShape(Box<[usize]>, Box<[usize]>),
    /// Batched `matmul` operands have incompatible batch dimensions. Carries the two batch sizes.
    NotSameBatch(usize, usize),
    /// A 0-D shape (`&[]`) was given; tensors must have rank >= 1.
    ZeroRankShape,
    // A declared slot was not the same used during construction of the skeleton
    NotSameSlot(usize),
    // The amount of slots provided to the skeleton was different than used
    IncorrectSlotAmount(usize, usize),
    // The layout of the idx is not the same declared slot layout
    NotSameLayoutAtSlot(usize),
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
            OpError::NotSameSlot(slot_idx) => {
                write!(
                    f,
                    "slot at idx {} was not used in the skeleton construction",
                    slot_idx
                )
            }
            OpError::IncorrectSlotAmount(expected, got) => {
                write!(
                    f,
                    "got {} slots binded to the skeleton but expected {}",
                    got, expected
                )
            }
            OpError::NotSameLayoutAtSlot(slot_idx) => {
                write!(
                    f,
                    "slot at idx {} did not have a compatible layout",
                    slot_idx
                )
            }
        }
    }
}

impl std::error::Error for OpError {}
