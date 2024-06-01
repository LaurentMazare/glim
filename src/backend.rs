use anyhow::Result;

pub trait Backend<T: crate::WithDType>: Sized + 'static {
    type Device;

    /// # Safety
    /// This function allocates an unitialized block of memory. It is the responsibility of the
    /// caller to set the memory before using or returning the block.
    unsafe fn alloc_uninit(len: usize) -> Result<Self>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn device(&self) -> &Self::Device;
    fn fill(&mut self, elem: T) -> Result<()>;

    fn copy(&self) -> Result<Self>;

    fn add_assign(&mut self, s: &Self) -> Result<()>;
    fn mul_assign(&mut self, s: &Self) -> Result<()>;
    fn scale(&mut self, v: T) -> Result<()>;

    fn transpose(&mut self, s: &Self, dim1: usize, dim2: usize, dims: &[usize]) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn copy2d(
        &mut self,
        src: &Self,
        d1: usize,
        d2: usize,
        dst_s: usize,
        src_s: usize,
        dst_o: usize,
        src_o: usize,
    ) -> Result<()>;

    fn rope(&mut self, _: &Self, _: &Self, b: usize, h: usize, t: usize, d: usize) -> Result<()>;
    fn rope_i(&mut self, _: &Self, _: &Self, b: usize, h: usize, t: usize, d: usize) -> Result<()>;
}

pub trait BackendF<T: crate::WithDType + num_traits::Float>: Backend<T> {
    fn cos(&mut self) -> Result<()>;
    fn sin(&mut self) -> Result<()>;
    fn silu(&mut self) -> Result<()>;
    fn apply_causality_mask(&mut self, bh: usize, t1: usize, t2: usize) -> Result<()>;
    fn softmax(&mut self, src: &Self, dim_m1: usize) -> Result<()>;
}
