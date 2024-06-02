#![allow(unused)]
use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, DeviceSlice};
use cudarc::driver::{CudaFunction, LaunchAsync, LaunchConfig};

#[derive(Clone)]
pub struct Device {
    cuda: std::sync::Arc<CudaDevice>,
}

impl Device {
    fn get_or_load_func(&self, module_name: &str, ptx: &'static str) -> Result<CudaFunction> {
        if !self.cuda.has_func(module_name, module_name) {
            // Leaking the string here is a bit sad but we need a &'static str and this is only
            // done once per kernel name.
            let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
            self.cuda.load_ptx(ptx.into(), module_name, &[static_module_name])?;
        }
        let func = self
            .cuda
            .get_func(module_name, module_name)
            // Clippy recommends this `ok_or` rather than `ok_or_else` so hopefully the compiler is
            // able to only build the error value if needed.
            .with_context(|| format!("missing kernel {module_name}"))?;
        Ok(func)
    }
}

pub struct Storage<T> {
    data: CudaSlice<T>,
    device: Device,
}

impl<T: crate::WithDType + DeviceRepr> crate::Backend<T> for Storage<T> {
    type Device = Device;
    fn len(&self) -> usize {
        self.data.len()
    }

    fn fill(&mut self, _v: T, elem_count: usize) -> Result<()> {
        // TODO: Use the proper dtype here.
        let func = self.device.get_or_load_func("fill_f32", candle_kernels::FILL)?;
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let params = (&self.data, 42., elem_count);
        unsafe { func.launch(cfg, params) }?;
        Ok(())
    }

    fn copy(&self, len: usize) -> Result<Self> {
        let mut dst = unsafe { self.device.cuda.alloc::<T>(len) }?;
        let src = self.data.slice(..len);
        self.device.cuda.dtod_copy(&src, &mut dst)?;
        Ok(Self { data: dst, device: self.device.clone() })
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
        &self.device
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
        self.data.is_empty()
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
    fn from_vec(v: Vec<T>, device: &Self::Device) -> Result<Self> {
        let data = device.cuda.htod_sync_copy(&v)?;
        Ok(Self { data, device: device.clone() })
    }
    fn data(&self, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        anyhow::bail!("not implemented")
    }
    unsafe fn alloc_uninit(len: usize, device: &Self::Device) -> Result<Self> {
        let data = unsafe { device.cuda.alloc::<T>(len) }?;
        Ok(Self { data, device: device.clone() })
    }
}

impl<T: crate::WithDTypeF + DeviceRepr> crate::BackendF<T> for Storage<T> {
    fn cos(&mut self, len: usize) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn sin(&mut self, len: usize) -> Result<()> {
        anyhow::bail!("not implemented")
    }
    fn silu(&mut self, len: usize) -> Result<()> {
        // TODO: Proper dtype kernel.
        let func = self.device.get_or_load_func("usilu_f32", candle_kernels::UNARY)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let params = (len, 1usize, 0usize, 0usize, &mut self.data);
        unsafe { func.launch(cfg, params) }?;

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
