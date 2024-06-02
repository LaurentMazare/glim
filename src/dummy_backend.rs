#![allow(unused)]
use anyhow::Result;

pub struct Dummy;

impl<T: crate::WithDType> crate::Backend<T> for Dummy {
    type Device = ();
    fn len(&self) -> usize {
        0
    }

    fn fill(&mut self, _: T, _: usize) -> Result<()> {
        anyhow::bail!("not implemented")
    }

    fn copy(&self, len: usize) -> Result<Self> {
        anyhow::bail!("not implemented")
    }
    fn rope(
        &mut self,
        _: &Self,
        _: &Self,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()> {
        anyhow::bail!("not implemented")
    }
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
    ) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn scale(&mut self, v: T, len: usize) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn device(&self) -> &Self::Device {
        &()
    }
    fn copy2d(
        &mut self,
        src: &Self,
        d1: usize,
        d2: usize,
        dst_s: usize,
        src_s: usize,
        dst_o: usize,
        src_o: usize,
    ) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn rope_i(
        &mut self,
        _: &Self,
        _: &Self,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn is_empty(&self) -> bool {
        true
    }
    fn transpose(&mut self, s: &Self, dim1: usize, dim2: usize, dims: &[usize]) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn add_assign(&mut self, s: &Self, len: usize) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn mul_assign(&mut self, s: &Self, len: usize) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn index_select(&mut self, src: &Self, ids: &[u32], dim: usize) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn from_vec(v: Vec<T>, _: &Self::Device) -> Result<Self> {
        anyhow::bail!("not implemented")
    }
    fn data(&self, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        anyhow::bail!("not implemented")
    }
    unsafe fn alloc_uninit(len: usize, _: &Self::Device) -> Result<Self> {
        anyhow::bail!("not implemented")
    }
}

impl<T: crate::WithDTypeF> crate::BackendF<T> for Dummy {
    fn cos(&mut self, len: usize) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn sin(&mut self, len: usize) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn silu(&mut self, len: usize) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn softmax(&mut self, src: &Self, dim_m1: usize, d: usize) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn rms_norm(
        &mut self,
        src: &Self,
        alpha: &Self,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn apply_causality_mask(&mut self, bh: usize, t1: usize, t2: usize) -> Result<()> {
        anyhow::bail!("not implemented")
    }
}
