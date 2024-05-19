use crate::{Dim, Shape, TensorView};
use anyhow::Result;
use rayon::prelude::*;

#[derive(Clone)]
pub struct Tensor {
    // [data] can hold more data than what is used in [shape].
    data: Vec<f32>,
    shape: Shape,
}

impl Tensor {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// The dimension size for a specified dimension index.
    pub fn dim<D: Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(self.shape(), "dim")?;
        Ok(self.dims()[dim])
    }

    pub fn add(&mut self, src: &Self) -> Result<()> {
        if self.shape != src.shape {
            anyhow::bail!("shape mismatch in add {:?} {:?}", self.shape, src.shape)
        }
        src.data().iter().zip(self.data_mut().iter_mut()).for_each(|(src, dst)| *dst += *src);
        Ok(())
    }

    pub fn mult(&mut self, src: &Self) -> Result<()> {
        if self.shape != src.shape {
            anyhow::bail!("shape mismatch in mult {:?} {:?}", self.shape, src.shape)
        }
        src.data().iter().zip(self.data_mut().iter_mut()).for_each(|(src, dst)| *dst *= *src);
        Ok(())
    }

    pub fn scale(&mut self, m: f32) {
        self.data_mut().iter_mut().for_each(|v| *v *= m)
    }

    pub fn data(&self) -> &[f32] {
        let elem_count = self.elem_count();
        &self.data[..elem_count]
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        let elem_count = self.elem_count();
        &mut self.data[..elem_count]
    }

    pub fn new(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        let shape: Shape = shape.into();
        if shape.elem_count() > data.len() {
            anyhow::bail!("unexpected shape in new {shape:?} {}", data.len())
        }
        Ok(Self { data, shape })
    }

    pub fn cst(data: f32, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let data = vec![data; shape.elem_count()];
        Ok(Self { data, shape })
    }

    pub fn cos(&mut self) {
        for d in self.data_mut().iter_mut() {
            *d = d.cos();
        }
    }

    pub fn sin(&mut self) {
        for d in self.data_mut().iter_mut() {
            *d = d.sin();
        }
    }

    pub fn silu(&mut self) {
        for d in self.data_mut().iter_mut() {
            *d /= 1. + f32::exp(-*d)
        }
    }

    // There is no stride so all tensors are always using the C layout
    pub fn reshape(&mut self, s: impl Into<Shape>) -> Result<()> {
        let s = s.into();
        if s.elem_count() != self.shape.elem_count() {
            anyhow::bail!("num-elems mismatch {s:?} {:?}", self.shape)
        }
        self.shape = s;
        Ok(())
    }

    // TODO: This should probably be merged with the main matmul method using a trait to dispatch
    // between TensorView and Tensor for lhs and rhs.
    pub fn matmul_v(&mut self, lhs: &Self, rhs: &TensorView<'_>, rhs_t: bool) -> Result<()> {
        let (lhs_b, lhs_m, lhs_k) = match lhs.dims() {
            [a, b] => (1, *a, *b),
            [a, b, c] => (*a, *b, *c),
            _ => anyhow::bail!("unexpected shape for matmul lhs {:?}", &lhs.shape),
        };
        let (rhs_b, rhs_k, rhs_n) = match rhs.dims() {
            [a, b] => (1, *a, *b),
            [a, b, c] => (*a, *b, *c),
            _ => anyhow::bail!("unexpected shape for matmul rhs {:?}", &rhs.shape()),
        };
        let (rhs_k, rhs_n) = if rhs_t { (rhs_n, rhs_k) } else { (rhs_k, rhs_n) };
        // Having rhs_b = 1 is ok if dst_b = lhs_b > 1
        if rhs_b != 1 && rhs_b != lhs_b {
            anyhow::bail!(
                "matmul shape mismatch lhs {:?}, rhs {:?} {rhs_t}",
                lhs.shape(),
                rhs.shape()
            )
        }
        if rhs_k != lhs_k {
            anyhow::bail!(
                "matmul shape mismatch lhs {:?}, rhs {:?} {rhs_t}",
                lhs.shape(),
                rhs.shape()
            )
        }
        let dst_elems = lhs_b * lhs_m * rhs_n;
        if dst_elems > self.data.len() {
            anyhow::bail!(
                "matmul dst is too small, dst {} < {dst_elems}, lhs {:?} rhs {:?}",
                self.data.len(),
                lhs.shape(),
                rhs.shape()
            )
        }
        let (m, n, k) = (lhs_m, rhs_n, lhs_k);
        self.shape =
            if lhs.rank() == 2 && rhs.rank() == 2 { (m, n).into() } else { (lhs_b, m, n).into() };
        let rhs_data = rhs.data();
        let b_stride = rhs.strides()[0];
        let (dst_rs, dst_cs) = (n, 1);
        let (lhs_rs, lhs_cs) = (k, 1);
        let (rhs_stride_m2, rhs_stride_m1) = {
            let l = rhs.strides().len();
            (rhs.strides()[l - 2], rhs.strides()[l - 1])
        };
        let (rhs_rs, rhs_cs) =
            if rhs_t { (rhs_stride_m1, rhs_stride_m2) } else { (rhs_stride_m2, rhs_stride_m1) };

        for b_idx in 0..lhs_b {
            let dst = &mut self.data[b_idx * m * n..(b_idx + 1) * m * n];
            let lhs = &lhs.data[b_idx * m * k..(b_idx + 1) * m * k];
            let rhs = &rhs_data[b_idx * b_stride..];
            unsafe {
                gemm::gemm(
                    /* m: usize = */ m,
                    /* n: usize = */ n,
                    /* k: usize = */ k,
                    /* dst: *mut T = */ dst.as_mut_ptr(),
                    /* dst_cs: isize = */ dst_cs as isize,
                    /* dst_rs: isize = */ dst_rs as isize,
                    /* read_dst: bool = */ false,
                    /* lhs: *const T = */ lhs.as_ptr(),
                    /* lhs_cs: isize = */ lhs_cs as isize,
                    /* lhs_rs: isize = */ lhs_rs as isize,
                    /* rhs: *const T = */ rhs.as_ptr(),
                    /* rhs_cs: isize = */ rhs_cs as isize,
                    /* rhs_rs: isize = */ rhs_rs as isize,
                    /* alpha: T = */ 0f32,
                    /* beta: T = */ 1f32,
                    /* conj_dst: bool = */ false,
                    /* conj_lhs: bool = */ false,
                    /* conj_rhs: bool = */ false,
                    gemm::Parallelism::Rayon(get_num_threads()),
                )
            }
        }
        Ok(())
    }

    pub fn matmul(&mut self, lhs: &Self, rhs: &Self, rhs_t: bool) -> Result<()> {
        let (lhs_b, lhs_m, lhs_k) = match lhs.dims() {
            [a, b] => (1, *a, *b),
            [a, b, c] => (*a, *b, *c),
            _ => anyhow::bail!("unexpected shape for matmul lhs {:?}", &lhs.shape),
        };
        let (rhs_b, rhs_k, rhs_n) = match rhs.dims() {
            [a, b] => (1, *a, *b),
            [a, b, c] => (*a, *b, *c),
            _ => anyhow::bail!("unexpected shape for matmul rhs {:?}", &rhs.shape),
        };
        let (rhs_k, rhs_n) = if rhs_t { (rhs_n, rhs_k) } else { (rhs_k, rhs_n) };
        // Having rhs_b = 1 is ok if dst_b = lhs_b > 1
        if rhs_b != 1 && rhs_b != lhs_b {
            anyhow::bail!(
                "matmul shape mismatch lhs {:?}, rhs {:?} {rhs_t}",
                lhs.shape(),
                rhs.shape()
            )
        }
        if rhs_k != lhs_k {
            anyhow::bail!(
                "matmul shape mismatch lhs {:?}, rhs {:?} {rhs_t}",
                lhs.shape(),
                rhs.shape()
            )
        }
        let dst_elems = lhs_b * lhs_m * rhs_n;
        if dst_elems > self.data.len() {
            anyhow::bail!(
                "matmul dst is too small, dst {} < {dst_elems}, lhs {:?} rhs {:?}",
                self.data.len(),
                lhs.shape(),
                rhs.shape()
            )
        }
        let (m, n, k) = (lhs_m, rhs_n, lhs_k);
        let rhs_stride = if rhs_b == 1 { 0 } else { k * n };
        self.shape =
            if lhs.rank() == 2 && rhs.rank() == 2 { (m, n).into() } else { (lhs_b, m, n).into() };
        for b_idx in 0..lhs_b {
            let dst = &mut self.data[b_idx * m * n..(b_idx + 1) * m * n];
            let lhs = &lhs.data[b_idx * m * k..(b_idx + 1) * m * k];
            let rhs = &rhs.data[b_idx * rhs_stride..b_idx * rhs_stride + n * k];
            matmul(dst, lhs, rhs, (m, n, k), rhs_t)?;
        }
        Ok(())
    }

    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn transpose(&mut self, src: &Self, dim1: usize, dim2: usize) -> Result<()> {
        if src.elem_count() != self.elem_count() {
            anyhow::bail!(
                "num-elems mismatch in transpose, dst {:?} src {:?}",
                self.shape(),
                src.shape()
            )
        }
        if dim1 >= src.rank() || dim2 >= src.rank() {
            anyhow::bail!("dim out of bounds in transpose {:?} {dim1} {dim2}", self.shape())
        }
        if dim1 == dim2 {
            self.data.copy_from_slice(&src.data);
            return Ok(());
        }
        let (dim1, dim2) = (usize::min(dim1, dim2), usize::max(dim1, dim2));
        let dims = src.shape().dims();
        let d_i = dims[..dim1].iter().product::<usize>();
        let d_j = dims[dim1 + 1..dim2].iter().product::<usize>();
        let d_k = dims[(dim2 + 1)..].iter().product::<usize>();
        let d1 = dims[dim1];
        let d2 = dims[dim2];
        // Inefficient, we should blit the data where possible.
        // i: pre
        for i in 0..d_i {
            for a1 in 0..d1 {
                // j: mid
                for j in 0..d_j {
                    for a2 in 0..d2 {
                        // k: post
                        for k in 0..d_k {
                            let src_idx = i * d1 * d_j * d2 * d_k
                                + a1 * d_j * d2 * d_k
                                + j * d2 * d_k
                                + a2 * d_k
                                + k;
                            let dst_idx = i * d2 * d_j * d1 * d_k
                                + a2 * d_j * d1 * d_k
                                + j * d1 * d_k
                                + a1 * d_k
                                + k;
                            self.data[dst_idx] = src.data[src_idx]
                        }
                    }
                }
            }
        }
        let mut shape = dims.to_vec();
        shape.swap(dim1, dim2);
        self.shape = shape.into();
        Ok(())
    }

    pub fn softmax(&mut self, src: &Self) -> Result<()> {
        if src.shape.elem_count() > self.capacity() {
            anyhow::bail!("missing capacity for softmax {} {:?}", self.capacity(), src.shape)
        }
        self.shape = src.shape.clone();
        let dim_m1 = self.dim(crate::D::Minus1)?;
        softmax(self.data_mut(), src.data(), dim_m1)
    }

    pub fn rope(&mut self, cos: &Self, sin: &Self) -> Result<()> {
        let (b, h, t, d) = self.shape().dims4()?;
        match cos.dims() {
            [_t, d_over_2] if 2 * d_over_2 == d => {}
            s => anyhow::bail!("unexpected shape for rope-cos {s:?} (head-dim {d})"),
        };
        match sin.dims() {
            [_t, d_over_2] if 2 * d_over_2 == d => {}
            s => anyhow::bail!("unexpected shape for rope-sin {s:?} (head-dim {d})"),
        };
        rope(self.data_mut(), &cos.data, &sin.data, b, h, t, d)
    }

    pub fn rope_i(&mut self, cos: &Self, sin: &Self) -> Result<()> {
        let (b, h, t, d) = self.shape().dims4()?;
        match cos.dims() {
            [_t, d_over_2] if 2 * d_over_2 == d => {}
            s => anyhow::bail!("unexpected shape for rope-cos {s:?} (head-dim {d})"),
        };
        match sin.dims() {
            [_t, d_over_2] if 2 * d_over_2 == d => {}
            s => anyhow::bail!("unexpected shape for rope-sin {s:?} (head-dim {d})"),
        };
        rope_i(self.data_mut(), &cos.data, &sin.data, b, h, t, d)
    }

    pub fn apply_causality_mask(&mut self) -> Result<()> {
        let (bh, t1, t2) = self.shape().dims3()?;
        for idx_b in 0..bh {
            for idx1 in 0..t1 {
                for idx2 in idx1 + 1..t2 {
                    let idx = idx_b * t1 * t2 + idx1 * t2 + idx2;
                    self.data[idx] = f32::NEG_INFINITY
                }
            }
        }
        Ok(())
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    pub fn slice_assign<D: Dim>(&mut self, src: &Self, dim: D, offset: usize) -> Result<()> {
        let dim = dim.to_index(self.shape(), "slice-set")?;
        if self.rank() != src.rank() {
            anyhow::bail!("rank mismatch in slice_assign {} <> {}", self.rank(), src.rank())
        }
        for (dim_idx, (v1, v2)) in self.dims().iter().zip(src.dims().iter()).enumerate() {
            if dim_idx == dim && *v2 + offset > *v1 {
                anyhow::bail!("shape mismatch on target dim, dst: {v1}, src: {v2} + {offset}")
            }
            if dim_idx != dim && v1 != v2 {
                anyhow::bail!("shape mismatch on dim {dim_idx}, {v1} <> {v2}")
            }
        }
        let block_size: usize = src.dims().iter().skip(1 + dim).product();
        let d1: usize = src.dims().iter().take(dim).product();
        let d2 = block_size * src.dims()[dim];
        let dst_o = offset * block_size;
        let src_o = 0;
        let dst_s = block_size * self.dims()[dim];
        let src_s = d2;
        copy2d(self.data_mut(), src.data(), d1, d2, dst_s, src_s, dst_o, src_o);
        Ok(())
    }

    #[cfg(feature = "candle")]
    pub fn to_candle(&self) -> Result<candle::Tensor> {
        let t = candle::Tensor::from_slice(self.data(), self.dims(), &candle::Device::Cpu)?;
        Ok(t)
    }

    #[cfg(feature = "candle")]
    pub fn from_candle(t: &candle::Tensor) -> Result<Self> {
        let data = t.flatten_all()?.to_vec1::<f32>()?;
        Tensor::new(data, t.dims())
    }
}

#[allow(clippy::too_many_arguments)]
fn copy2d(
    dst: &mut [f32],
    src: &[f32],
    d1: usize,
    d2: usize,
    dst_s: usize,
    src_s: usize,
    dst_o: usize,
    src_o: usize,
) {
    for i1 in 0..d1 {
        let dst_idx = i1 * dst_s + dst_o;
        let src_idx = i1 * src_s + src_o;
        let dst = &mut dst[dst_idx..dst_idx + d2];
        let src = &src[src_idx..src_idx + d2];
        dst.copy_from_slice(src)
    }
}

fn matmul(
    dst: &mut [f32],
    lhs: &[f32],
    rhs: &[f32],
    (m, n, k): (usize, usize, usize),
    rhs_t: bool,
) -> Result<()> {
    let (dst_rs, dst_cs) = (n, 1);
    let (lhs_rs, lhs_cs) = (k, 1);
    let (rhs_rs, rhs_cs) = if rhs_t { (1, k) } else { (n, 1) };
    unsafe {
        gemm::gemm(
            /* m: usize = */ m,
            /* n: usize = */ n,
            /* k: usize = */ k,
            /* dst: *mut T = */ dst.as_mut_ptr(),
            /* dst_cs: isize = */ dst_cs as isize,
            /* dst_rs: isize = */ dst_rs as isize,
            /* read_dst: bool = */ false,
            /* lhs: *const T = */ lhs.as_ptr(),
            /* lhs_cs: isize = */ lhs_cs as isize,
            /* lhs_rs: isize = */ lhs_rs as isize,
            /* rhs: *const T = */ rhs.as_ptr(),
            /* rhs_cs: isize = */ rhs_cs as isize,
            /* rhs_rs: isize = */ rhs_rs as isize,
            /* alpha: T = */ 0f32,
            /* beta: T = */ 1f32,
            /* conj_dst: bool = */ false,
            /* conj_lhs: bool = */ false,
            /* conj_rhs: bool = */ false,
            gemm::Parallelism::Rayon(get_num_threads()),
        )
    }
    Ok(())
}

fn softmax(dst: &mut [f32], src: &[f32], dim_m1: usize) -> Result<()> {
    src.par_chunks(dim_m1).zip(dst.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
        let mut max = f32::NEG_INFINITY;
        for &v in src.iter() {
            max = f32::max(v, max)
        }
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            *d = (*s - max).exp();
        }
        let sum_exp = dst.iter().sum::<f32>();
        for d in dst.iter_mut() {
            *d /= sum_exp
        }
    });
    Ok(())
}

fn rope(
    dst: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    b: usize,
    h: usize,
    t: usize,
    d: usize,
) -> Result<()> {
    if dst.len() != b * h * t * d {
        anyhow::bail!("rope unexpected size for dst {} {b} {h} {t} {d}", dst.len())
    }
    dst.par_chunks_mut(t * d).for_each(|dst| {
        for i_t in 0..t {
            for i_d in 0..d / 2 {
                let i1 = i_t * d + i_d;
                let i2 = i1 + d / 2;
                let i_cs = i_t * (d / 2) + i_d;
                let (src_i1, src_i2) = (dst[i1], dst[i2]);
                dst[i1] = src_i1 * cos[i_cs] - src_i2 * sin[i_cs];
                dst[i2] = src_i1 * sin[i_cs] + src_i2 * cos[i_cs];
            }
        }
    });

    Ok(())
}

fn rope_i(
    dst: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    b: usize,
    h: usize,
    t: usize,
    d: usize,
) -> Result<()> {
    if dst.len() != b * h * t * d {
        anyhow::bail!("rope-i unexpected size for dst {} {b} {h} {t} {d}", dst.len())
    }
    dst.par_chunks_mut(t * d).for_each(|dst| {
        for i_over_2 in 0..t * d / 2 {
            let i = 2 * i_over_2;
            let (s_i, s_ip) = (dst[i], dst[i + 1]);
            dst[i] = s_i * cos[i_over_2] - s_ip * sin[i_over_2];
            dst[i + 1] = s_i * sin[i_over_2] + s_ip * cos[i_over_2];
        }
    });
    Ok(())
}

fn get_num_threads() -> usize {
    use std::str::FromStr;
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| usize::from_str(&s).ok()) {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}
