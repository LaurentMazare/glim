use anyhow::Result;

pub trait BackendAlloc<T: crate::WithDType>: Sized + 'static {
    type Slice: ?Sized;

    fn len(&self) -> usize;
    fn slice(&self) -> &Self::Slice;
    fn slice_mut(&mut self) -> &mut Self::Slice;

    /// # Safety
    /// This function allocates an unitialized block of memory. It is the responsibility of the
    /// caller to set the memory before using or returning the block.
    unsafe fn alloc_uninit(len: usize) -> Result<Self>;
    fn cst(v: T, len: usize) -> Result<Self>;
    fn from_vec(v: Vec<T>) -> Result<Self>;
}

pub trait BackendSlice<T: crate::WithDType>: 'static {
    type Device;
    type Allocated: BackendAlloc<T, Slice = Self>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn device(&self) -> &Self::Device;
    fn fill(&mut self, elem: T) -> Result<()>;

    fn copy(&self) -> Result<Self::Allocated>;
    fn index(&self, a: Option<usize>, b: Option<usize>) -> &Self;
    fn index_mut(&mut self, a: Option<usize>, b: Option<usize>) -> &mut Self;

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

    fn gemm(
        &mut self,
        lhs: &Self,
        rhs: &Self,
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

pub trait BackendSliceF<T: crate::WithDType + num_traits::Float>: BackendSlice<T> {
    fn cos(&mut self) -> Result<()>;
    fn sin(&mut self) -> Result<()>;
    fn silu(&mut self) -> Result<()>;
    fn apply_causality_mask(&mut self, bh: usize, t1: usize, t2: usize) -> Result<()>;
    fn softmax(&mut self, src: &Self, dim_m1: usize) -> Result<()>;
    fn rms_norm(&mut self, src: &Self, alpha: &Self, dim_m1: usize, eps: f32) -> Result<()>;
}
