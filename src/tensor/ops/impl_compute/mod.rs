mod cpu_compute_generic;
mod cpu_f32;
mod cpu_f64;

pub(crate) use cpu_f32::cpu_compute_op_f32;
pub(crate) use cpu_f32::cpu_compute_op_f32_inplace;
pub(crate) use cpu_f64::cpu_compute_op_f64;
pub(crate) use cpu_f64::cpu_compute_op_f64_inplace;
