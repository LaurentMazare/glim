use half::{bf16, f16};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
}

pub trait WithDType: Sized + Copy + num_traits::NumAssign + 'static + Clone + Send + Sync {
    const DTYPE: DType;
}

impl WithDType for f16 {
    const DTYPE: DType = DType::F16;
}

impl WithDType for bf16 {
    const DTYPE: DType = DType::BF16;
}

impl WithDType for f32 {
    const DTYPE: DType = DType::F32;
}
