use anyhow::Result;

pub trait Backend<T: crate::WithDType>: Sized {
    type Device;

    /// # Safety
    /// This function allocates an unitialized block of memory. It is the responsibility of the
    /// caller to set the memory before using or returning the block.
    unsafe fn alloc_uninit(len: usize, device: &Self::Device) -> Result<Self>;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn device(&self) -> &Self::Device;
    fn fill(&self, elem: T) -> Result<()>;

    fn copy(&self) -> Result<Self>;

    fn add_assign(&mut self, s: &Self) -> Result<()>;
    fn mul_assign(&mut self, s: &Self) -> Result<()>;
    fn scale(&mut self, v: T) -> Result<()>;

    fn transpose(&mut self, s: &Self, dim1: usize, dim2: usize) -> Result<()>;
}

pub trait BackendF<T: crate::WithDType + num_traits::Float>: Sized {
    fn cos(&mut self) -> Result<()>;
    fn sin(&mut self) -> Result<()>;
    fn silu(&mut self) -> Result<()>;
}
