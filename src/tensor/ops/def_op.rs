#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sign {
    Plus,
    Minus,
}

// TODO: Design some way to fuse arbitrary combinations of ops
// without handling it at the runtime, because it would be annoying.
// Maybe macros?
#[derive(Clone, Debug, PartialEq)]
pub enum OpKindScalar<T> {
    AxBy(T, T),
    Exp,
    Ln,
    Log2,
    Recip, // (1 / y)
    ReLU,
    Tanh,
    // TODO: Sigmoid,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OpKind<T> {
    NoOp,
    ScalarOp(OpKindScalar<T>),
    FusedScalar(Box<[OpKindScalar<T>]>),
    View,
    Slice,
    Transpose,
    TransposeAxes,
    Broadcast,
    MatMul(T),             // a*(A @ B)
    MatMulSum(T, T, Sign), // a*(A @ B) +/- b * C
    AsContiguous,
    Add,
    Sub,
    Mul,
    Div,
    Sum,
    SumAxis(isize, bool),
    Mean,
    MeanAxis(isize, bool),
    Max,
    MaxAxis(isize, bool),
}

impl<T> OpKind<T> {
    pub fn as_str(&self) -> &'static str {
        match self {
            OpKind::NoOp => "NoOp",
            OpKind::ScalarOp(_) => "ScalarOp",
            OpKind::FusedScalar(_) => "FusedScalar",
            OpKind::View => "View",
            OpKind::Slice => "Slice",
            OpKind::Transpose => "Transpose",
            OpKind::TransposeAxes => "TransposeAxes",
            OpKind::Broadcast => "Broadcast",
            OpKind::MatMul(_) => "MatMul",
            OpKind::MatMulSum(_, _, _) => "MatMulSum",
            OpKind::AsContiguous => "AsContiguous",
            OpKind::Add => "Add",
            OpKind::Sub => "Sub",
            OpKind::Mul => "Mul",
            OpKind::Div => "Div",
            OpKind::Sum => "Sum",
            OpKind::SumAxis(_, _) => "SumAxis",
            OpKind::Mean => "Mean",
            OpKind::MeanAxis(_, _) => "MeanAxis",
            OpKind::Max => "Max",
            OpKind::MaxAxis(_, _) => "MaxAxis",
        }
    }
}
