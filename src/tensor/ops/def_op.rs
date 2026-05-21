use crate::tensor::mem_formats::layout::Layout;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sign {
    Plus,
    Minus,
}

// TODO: Design some way to fuse arbitrary combinations of ops
// without handling it at the runtime, because it would be annoying.
// Maybe macros?
#[derive(Clone, Debug)]
pub enum OpKindScalar<T: Copy> {
    AxBy(T, T),
    Exp,
    Ln,
    Log2,
    Inv, // (1 / y)
    ReLU,
    Tanh,
    // TODO: Sigmoid,
}

#[derive(Clone, Debug)]
pub enum OpKind<T: Copy> {
    NoOp,
    ScalarOp(OpKindScalar<T>),
    FusedScalar(Box<[OpKindScalar<T>]>),
    View(Layout),
    Slice(Layout),
    Transpose,
    TransposeAxes(Layout),
    Broadcast(Layout),
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

impl<T: Copy> OpKind<T> {
    pub fn as_str(&self) -> &'static str {
        match self {
            OpKind::NoOp => "NoOp",
            OpKind::ScalarOp(_) => "ScalarOp",
            OpKind::FusedScalar(_) => "FusedScalar",
            OpKind::View(_) => "View",
            OpKind::Slice(_) => "Slice",
            OpKind::Transpose => "Transpose",
            OpKind::TransposeAxes(_) => "TransposeAxes",
            OpKind::Broadcast(_) => "Broadcast",
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
