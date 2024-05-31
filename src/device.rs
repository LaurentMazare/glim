use crate::DType;
use anyhow::Result;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Location {
    Cpu,
    Cuda,
}

pub trait Backend: Sized {
    const LOCATION: Location;
    type Device: Clone;

    /// # Safety
    /// This function allocates an unitialized block of memory. It is the responsibility of the
    /// caller to set the memory before using or returning the block.
    unsafe fn alloc_uninit(len: usize, dtype: DType, device: &Self::Device) -> Result<Self>;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn device(&self) -> &Self::Device;
    fn dtype(&self) -> DType;

    fn copy(&self) -> Result<Self>;

    fn add_assign(&mut self, s: &Self) -> Result<()>;
    fn mul_assign(&mut self, s: &Self) -> Result<()>;
    fn transpose(&mut self, s: &Self, dim1: usize, dim2: usize) -> Result<()>;
    fn cos(&mut self) -> Result<()>;
    fn sin(&mut self) -> Result<()>;
    fn silu(&mut self) -> Result<()>;
}
