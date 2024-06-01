use anyhow::Result;

pub trait Backend<T: crate::WithDType>: Sized + 'static {
    type Device;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// # Safety
    /// This function allocates an unitialized block of memory. It is the responsibility of the
    /// caller to set the memory before using or returning the block.
    unsafe fn alloc_uninit(len: usize) -> Result<Self>;
    fn from_vec(v: Vec<T>) -> Result<Self>;
    fn cst(v: T, len: usize) -> Result<Self> {
        let mut res = unsafe { Self::alloc_uninit(len)? };
        res.fill(v, len)?;
        Ok(res)
    }

    fn device(&self) -> &Self::Device;
    fn fill(&mut self, elem: T, len: usize) -> Result<()>;
    fn copy(&self, len: usize) -> Result<Self>;

    fn add_assign(&mut self, s: &Self, len: usize) -> Result<()>;
    fn mul_assign(&mut self, s: &Self, len: usize) -> Result<()>;
    fn scale(&mut self, v: T, len: usize) -> Result<()>;

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

    #[allow(clippy::too_many_arguments)]
    fn rope(
        &mut self,
        _: &Self,
        _: &Self,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn rope_i(
        &mut self,
        _: &Self,
        _: &Self,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &mut self,
        lhs: (&Self, usize),
        rhs: (&Self, usize),
        m: usize,
        n: usize,
        k: usize,
        lhs_b: usize,
        b_stride: usize,
        _: (usize, usize),
        _: (usize, usize),
        _: (usize, usize),
    ) -> Result<()>;

    fn index_select(&mut self, src: &Self, ids: &[u32], dim: usize) -> Result<()>;
}

pub trait BackendF<T: crate::WithDTypeF>: Backend<T> {
    fn cos(&mut self, len: usize) -> Result<()>;
    fn sin(&mut self, len: usize) -> Result<()>;
    fn silu(&mut self, len: usize) -> Result<()>;
    fn apply_causality_mask(&mut self, bh: usize, t1: usize, t2: usize) -> Result<()>;
    fn softmax(&mut self, src: &Self, dim_m1: usize, d: usize) -> Result<()>;
    fn rms_norm(
        &mut self,
        src: &Self,
        alpha: &Self,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()>;
}
