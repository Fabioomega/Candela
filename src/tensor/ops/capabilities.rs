pub(crate) trait NumericOp {}

pub(crate) trait FloatLike: NumericOp {}

pub(crate) trait CanMatMul: NumericOp {}

impl NumericOp for f64 {}
impl NumericOp for f32 {}

impl FloatLike for f64 {}
impl FloatLike for f32 {}

impl CanMatMul for f64 {}
impl CanMatMul for f32 {}
