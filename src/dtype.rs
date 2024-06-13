use half::{bf16, f16};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
}

pub trait WithDType:
    Sized + Copy + num_traits::NumAssign + 'static + Clone + Send + Sync + std::fmt::Debug
{
    const DTYPE: DType;
    fn from_be_bytes(dst: &mut [Self], src: &[u8]);
}

pub trait WithDTypeF: WithDType + num_traits::Float {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}

impl WithDType for f16 {
    const DTYPE: DType = DType::F16;

    fn from_be_bytes(dst: &mut [Self], src: &[u8]) {
        for (i, v) in dst.iter_mut().enumerate() {
            *v = f16::from_bits(u16::from_be_bytes([src[2 * i + 1], src[2 * i]]))
        }
    }
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

    fn from_be_bytes(dst: &mut [Self], src: &[u8]) {
        for (i, v) in dst.iter_mut().enumerate() {
            *v = bf16::from_bits(u16::from_be_bytes([src[2 * i + 1], src[2 * i]]))
        }
    }
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

    fn from_be_bytes(dst: &mut [Self], src: &[u8]) {
        for (i, v) in dst.iter_mut().enumerate() {
            *v = f32::from_bits(u32::from_be_bytes([
                src[4 * i + 3],
                src[4 * i + 2],
                src[4 * i + 1],
                src[4 * i],
            ]))
        }
    }
}

impl WithDTypeF for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(v: f32) -> Self {
        v
    }
}
