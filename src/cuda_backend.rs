#![allow(unused)]
use anyhow::{Context, Result};
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceRepr, DeviceSlice};
use cudarc::driver::{CudaFunction, LaunchAsync, LaunchConfig};
use half::{bf16, f16};
use std::sync::Arc;

#[derive(Clone)]
pub struct Device {
    cuda: Arc<CudaDevice>,
    blas: Arc<cudarc::cublas::CudaBlas>,
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        let cuda = cudarc::driver::CudaDevice::new(ordinal)?;
        let blas = cudarc::cublas::CudaBlas::new(cuda.clone())?;
        Ok(Self { cuda, blas: Arc::new(blas) })
    }

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

fn kernel_name<T: crate::WithDType>(base_name: &str) -> String {
    let dtype_str = match T::DTYPE {
        crate::DType::F16 => "f16",
        crate::DType::BF16 => "bf16",
        crate::DType::F32 => "f32",
    };
    format!("{base_name}_{dtype_str}")
}

trait CudaType: crate::WithDType + DeviceRepr {
    /// # Safety: ...
    unsafe fn gemm(
        cublas: &cudarc::cublas::CudaBlas,
        cfg: StridedBatchedConfig<Self>,
        a: &cudarc::driver::CudaView<Self>,
        b: &cudarc::driver::CudaView<Self>,
        c: &mut CudaSlice<Self>,
    ) -> std::result::Result<(), cudarc::cublas::result::CublasError>;
}

impl<T: CudaType> crate::Backend<T> for Storage<T> {
    type Device = Device;
    fn len(&self) -> usize {
        self.data.len()
    }

    fn fill(&mut self, v: T, elem_count: usize) -> Result<()> {
        let kname = kernel_name::<T>("fill");
        let func = self.device.get_or_load_func(&kname, candle_kernels::FILL)?;
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let params = (&self.data, v, elem_count);
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
        (dst_cs, dst_rs): (usize, usize),
        (lhs_cs, lhs_rs): (usize, usize),
        (rhs_cs, rhs_rs): (usize, usize),
    ) -> Result<()> {
        // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
        use cudarc::cublas::sys::cublasOperation_t;
        let gemm = GemmConfig {
            alpha: T::one(),
            beta: T::zero(),
            m: n as i32,
            n: m as i32,
            k: k as i32,
            lda: lhs_cs as i32,
            ldb: rhs_cs as i32,
            ldc: dst_cs as i32,
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
        };
        let cfg = StridedBatchedConfig {
            batch_size: lhs_b as i32,
            gemm,
            stride_a: (m * k) as i64,
            stride_b: b_stride as i64,
            stride_c: (m * n) as i64,
        };
        let lhs = &lhs.0.data.slice(lhs.1..);
        let rhs = &rhs.0.data.slice(rhs.1..);
        unsafe { T::gemm(&self.device.blas, cfg, lhs, rhs, &mut self.data)? };
        Ok(())
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

impl<T: crate::WithDTypeF + CudaType> crate::BackendF<T> for Storage<T> {
    fn cos(&mut self, len: usize) -> Result<()> {
        let kname = kernel_name::<T>("ucos");
        let func = self.device.get_or_load_func(&kname, candle_kernels::UNARY)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let params = (len, 1usize, 0usize, 0usize, &mut self.data);
        unsafe { func.launch(cfg, params) }?;
        Ok(())
    }

    fn sin(&mut self, len: usize) -> Result<()> {
        let kname = kernel_name::<T>("usin");
        let func = self.device.get_or_load_func(&kname, candle_kernels::UNARY)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let params = (len, 1usize, 0usize, 0usize, &mut self.data);
        unsafe { func.launch(cfg, params) }?;
        Ok(())
    }

    fn silu(&mut self, len: usize) -> Result<()> {
        let kname = kernel_name::<T>("usilu");
        let func = self.device.get_or_load_func(&kname, candle_kernels::UNARY)?;
        let cfg = LaunchConfig::for_num_elems(len as u32);
        let params = (len, 1usize, 0usize, 0usize, &mut self.data);
        unsafe { func.launch(cfg, params) }?;
        Ok(())
    }

    fn softmax(&mut self, src: &Self, dim_m1: usize, d: usize) -> Result<()> {
        let kname = kernel_name::<T>("softmax");
        let func = self.device.get_or_load_func(&kname, candle_kernels::REDUCE)?;
        let cfg = LaunchConfig::for_num_elems((d * dim_m1) as u32);
        let params = (&src.data, &mut self.data, dim_m1 as i32);
        unsafe { func.launch(cfg, params) }?;
        Ok(())
    }

    fn rms_norm(
        &mut self,
        src: &Self,
        alpha: &Self,
        dim_m1: usize,
        d: usize,
        eps: f32,
    ) -> Result<()> {
        let kname = kernel_name::<T>("rmsnorm");
        let func = self.device.get_or_load_func(&kname, candle_kernels::REDUCE)?;
        let cfg = LaunchConfig::for_num_elems((d * dim_m1) as u32);
        let params = (&src.data, &mut self.data, &alpha.data, dim_m1 as i32, eps);
        unsafe { func.launch(cfg, params) }?;
        Ok(())
    }

    fn apply_causality_mask(&mut self, bh: usize, t1: usize, t2: usize) -> Result<()> {
        anyhow::bail!("not implemented")
    }
}

// Default for the reduced precision setting is false, similar to pytorch.
// https://github.com/pytorch/pytorch/issues/123157
static MM_F16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_BF16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_F32_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
/// allowed with f32 GEMMs.
pub fn gemm_reduced_precision_f32() -> bool {
    MM_F32_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with tf32 accumulation type) are
/// allowed with f32 GEMMs.
pub fn set_gemm_reduced_precision_f32(b: bool) {
    MM_F32_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn gemm_reduced_precision_f16() -> bool {
    MM_F16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with f16 GEMMs.
pub fn set_gemm_reduced_precision_f16(b: bool) {
    MM_F16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn gemm_reduced_precision_bf16() -> bool {
    MM_BF16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// This bool controls whether reduced precision reductions (e.g., with fp16 accumulation type) are
/// allowed with bf16 GEMMs.
pub fn set_gemm_reduced_precision_bf16(b: bool) {
    MM_BF16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

impl CudaType for f32 {
    unsafe fn gemm(
        cublas: &cudarc::cublas::CudaBlas,
        cfg: StridedBatchedConfig<Self>,
        a: &cudarc::driver::CudaView<Self>,
        b: &cudarc::driver::CudaView<Self>,
        c: &mut CudaSlice<Self>,
    ) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
        use cudarc::cublas::sys;
        use cudarc::driver::DevicePtrMut;

        let compute_type = if gemm_reduced_precision_f32() {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
        } else {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        };
        let alpha = &cfg.gemm.alpha as *const f32 as *const _;
        let beta = &cfg.gemm.beta as *const f32 as *const _;

        cudarc::cublas::result::gemm_strided_batched_ex(
            *cublas.handle(),
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            alpha,
            *a.device_ptr() as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.gemm.lda,
            cfg.stride_a,
            *b.device_ptr() as *const _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.gemm.ldb,
            cfg.stride_b,
            beta,
            *c.device_ptr_mut() as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            compute_type,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
}

impl CudaType for f16 {
    unsafe fn gemm(
        cublas: &cudarc::cublas::CudaBlas,
        cfg: StridedBatchedConfig<Self>,
        a: &cudarc::driver::CudaView<Self>,
        b: &cudarc::driver::CudaView<Self>,
        c: &mut CudaSlice<Self>,
    ) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
        use cudarc::cublas::sys;
        use cudarc::driver::DevicePtrMut;

        let alpha = cfg.gemm.alpha;
        let beta = cfg.gemm.beta;
        let alpha_f32: f32 = cfg.gemm.alpha.to_f32();
        let beta_f32: f32 = cfg.gemm.beta.to_f32();
        let (compute_type, alpha, beta) = if gemm_reduced_precision_f16() {
            (
                sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
                (&alpha) as *const f16 as *const _,
                (&beta) as *const f16 as *const _,
            )
        } else {
            (
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                (&alpha_f32) as *const f32 as *const _,
                (&beta_f32) as *const f32 as *const _,
            )
        };

        cudarc::cublas::result::gemm_strided_batched_ex(
            *cublas.handle(),
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            alpha,
            *a.device_ptr() as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            cfg.gemm.lda,
            cfg.stride_a,
            *b.device_ptr() as *const _,
            sys::cudaDataType_t::CUDA_R_16F,
            cfg.gemm.ldb,
            cfg.stride_b,
            beta,
            *c.device_ptr_mut() as *mut _,
            sys::cudaDataType_t::CUDA_R_16F,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            compute_type,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
}

impl CudaType for bf16 {
    unsafe fn gemm(
        cublas: &cudarc::cublas::CudaBlas,
        cfg: StridedBatchedConfig<bf16>,
        a: &cudarc::driver::CudaView<bf16>,
        b: &cudarc::driver::CudaView<bf16>,
        c: &mut CudaSlice<bf16>,
    ) -> std::result::Result<(), cudarc::cublas::result::CublasError> {
        use cudarc::cublas::sys;
        use cudarc::driver::DevicePtrMut;

        let alpha_f32: f32 = cfg.gemm.alpha.to_f32();
        let beta_f32: f32 = cfg.gemm.beta.to_f32();
        let alpha = f16::from_f32(alpha_f32);
        let beta = f16::from_f32(beta_f32);
        // The type for alpha and beta depends on the computeType.
        // https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex
        let (compute_type, alpha, beta) = if gemm_reduced_precision_bf16() {
            (
                sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
                (&alpha) as *const f16 as *const _,
                (&beta) as *const f16 as *const _,
            )
        } else {
            (
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                (&alpha_f32) as *const f32 as *const _,
                (&beta_f32) as *const f32 as *const _,
            )
        };

        cudarc::cublas::result::gemm_strided_batched_ex(
            *cublas.handle(),
            cfg.gemm.transa,
            cfg.gemm.transb,
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            alpha,
            *a.device_ptr() as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            cfg.gemm.lda,
            cfg.stride_a,
            *b.device_ptr() as *const _,
            sys::cudaDataType_t::CUDA_R_16BF,
            cfg.gemm.ldb,
            cfg.stride_b,
            beta,
            *c.device_ptr_mut() as *mut _,
            sys::cudaDataType_t::CUDA_R_16BF,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            compute_type,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
}
