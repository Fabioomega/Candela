use crate::tensor::errors::OpError;
use crate::tensor::mem_formats::layout::Layout;
use crate::tensor::ops::def_op::OpKind;

#[cfg(test)]
#[path = "impl_layout_tests.rs"]
mod tests;

#[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip_all))]
pub fn compute_layout<T: Copy>(op: &OpKind<T>, inputs: &[&Layout]) -> Result<Layout, OpError> {
    match op {
        OpKind::ScalarOp(_) | OpKind::FusedScalar(_) => {
            if inputs[0].is_contiguous() {
                Ok(inputs[0].clone())
            } else {
                Ok(Layout::from_shape(inputs[0].shape(), 0))
            }
        }
        OpKind::NoOp => Ok(inputs[0].clone()),
        OpKind::View(new_layout)
        | OpKind::Slice(new_layout)
        | OpKind::TransposeAxes(new_layout)
        | OpKind::Broadcast(new_layout) => Ok(new_layout.clone()),
        OpKind::AsContiguous => Ok(Layout::from_shape(inputs[0].shape(), 0)),
        OpKind::Transpose => Ok(inputs[0].transpose()),
        OpKind::MatMul(_) => {
            // Assumes that the tensor is ALREADY BROADCASTED!
            let a_shape = inputs[0].shape_as_3d();
            let b_shape = inputs[1].shape_as_3d();

            if a_shape[2] != b_shape[1] {
                return Err(OpError::CannotMatMul(a_shape[2], b_shape[1]));
            };

            if a_shape[0] == 1 && b_shape[0] == 1 {
                return Ok(Layout::from_shape(&[a_shape[1], b_shape[2]], 0));
            }

            let mut new_shape = inputs[0].shape().to_vec();
            let last = new_shape.len() - 1;
            new_shape[last] = b_shape[2];

            Ok(Layout::from_shape(&new_shape, 0))
        }
        OpKind::MatMulSum(_, _, _) => {
            // Assumes that the tensor is ALREADY BROADCASTED!
            let a_shape = inputs[0].shape_as_3d();
            let b_shape = inputs[1].shape_as_3d();

            if a_shape[2] != b_shape[1] {
                return Err(OpError::CannotMatMul(a_shape[2], b_shape[1]));
            };

            if a_shape[0] == 1 && b_shape[0] == 1 {
                let output_layout = Layout::from_shape(&[a_shape[1], b_shape[2]], 0);
                if inputs[2].shape() != output_layout.shape() {
                    return Err(OpError::NotSameShape(
                        output_layout.shape().into(),
                        inputs[2].shape().into(),
                    ));
                }

                return Ok(output_layout);
            }

            let mut new_shape = inputs[0].shape().to_vec();
            let last = new_shape.len() - 1;
            new_shape[last] = b_shape[2];

            let output_layout = Layout::from_shape(&new_shape, 0);
            if inputs[2].shape() != output_layout.shape() {
                return Err(OpError::NotSameShape(
                    output_layout.shape().into(),
                    inputs[2].shape().into(),
                ));
            }

            Ok(output_layout)
        }
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div => {
            if inputs[0].shape() == inputs[1].shape() {
                Ok(inputs[0].clone())
            } else {
                Err(OpError::NotSameShape(
                    inputs[0].shape().into(),
                    inputs[1].shape().into(),
                ))
            }
        }
        _ => todo!("not implemented"),
    }
}
