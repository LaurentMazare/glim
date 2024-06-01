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

pub trait WithDTypeF: WithDType + num_traits::Float {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}

impl WithDType for f16 {
    const DTYPE: DType = DType::F16;
}

impl WithDTypeF for f16 {
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }

    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }
}

impl WithDType for bf16 {
    const DTYPE: DType = DType::BF16;
}

impl WithDTypeF for bf16 {
    fn to_f32(self) -> f32 {
        bf16::to_f32(self)
    }

    fn from_f32(v: f32) -> Self {
        bf16::from_f32(v)
    }
}

impl WithDType for f32 {
    const DTYPE: DType = DType::F32;
}

impl WithDTypeF for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(v: f32) -> Self {
        v
    }
}
