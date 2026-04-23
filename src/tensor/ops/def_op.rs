use crate::tensor::mem_formats::layout::Layout;

// TODO: Design some way to fuse arbitrary combinations of ops
// without handling it at the runtime, because it would be annoying.
// Maybe macros?
#[derive(Clone, Debug)]
pub enum OpKindScalar<T: Copy> {
    Sum(T),
    Sub(T),
    Mul(T),
    Div(T),
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
    Matmul,
    AsContiguous,
    Add,
    Sub,
    Mul,
    Div,
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
            OpKind::Matmul => "Matmul",
            OpKind::AsContiguous => "AsContiguous",
            OpKind::Add => "Add",
            OpKind::Sub => "Sub",
            OpKind::Mul => "Mul",
            OpKind::Div => "Div",
        }
    }
}
