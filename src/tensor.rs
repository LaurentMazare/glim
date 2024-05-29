use crate::storage::{CowMut, Storage};
use crate::{Dim, Shape, WithDType};
use anyhow::Result;
use rayon::prelude::*;

// TODO: separate type for StridedTensor? (cannot be owned)
pub struct Tensor<'a, T: WithDType> {
    data: CowMut<'a, Storage<T>>,
    shape: Shape,
}

impl<'a, T: WithDType> Tensor<'a, T> {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// The dimension size for a specified dimension index.
    pub fn dim<D: Dim>(&self, dim: D) -> Result<usize> {
        let dim = dim.to_index(self.shape(), "dim")?;
        Ok(self.dims()[dim])
    }

    pub fn capacity(&self) -> usize {
        self.data.inner.len()
    }

    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn new<S: Into<Shape>>(data: &'a mut Storage<T>, shape: S) -> Result<Self> {
        let shape = shape.into();
        if data.inner.len() < shape.elem_count() {
            anyhow::bail!("not enough elements in storage {} for shape {shape:?}", data.inner.len())
        }
        Ok(Self { data: CowMut::Borrowed(data), shape })
    }

    pub fn into_storage(self) -> CowMut<'a, Storage<T>> {
        self.data
    }

    pub fn zeros(&mut self) {
        self.data.inner.fill(T::zero())
    }

    pub fn scale(&mut self, m: T) {
        self.data_mut().iter_mut().for_each(|v| *v *= m)
    }

    pub fn add(&mut self, src: &Tensor<'_, T>) -> Result<()> {
        if self.shape != src.shape {
            anyhow::bail!("shape mismatch in add {:?} {:?}", self.shape, src.shape)
        }
        src.data().iter().zip(self.data_mut().iter_mut()).for_each(|(src, dst)| *dst += *src);
        Ok(())
    }

    pub fn mult(&mut self, src: &Tensor<'_, T>) -> Result<()> {
        if self.shape != src.shape {
            anyhow::bail!("shape mismatch in mult {:?} {:?}", self.shape, src.shape)
        }
        src.data().iter().zip(self.data_mut().iter_mut()).for_each(|(src, dst)| *dst *= *src);
        Ok(())
    }

    pub fn data(&self) -> &[T] {
        let elem_count = self.elem_count();
        &self.data.inner[..elem_count]
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        let elem_count = self.elem_count();
        &mut self.data.inner[..elem_count]
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

    pub fn transpose<'b>(
        &self,
        dst: &'b mut Storage<T>,
        dim1: usize,
        dim2: usize,
    ) -> Result<Tensor<'b, T>> {
        if dst.inner.len() != self.elem_count() {
            anyhow::bail!(
                "num-elems mismatch in transpose, dst {} src {:?}",
                dst.inner.len(),
                self.shape(),
            )
        }
        if dim1 >= self.rank() || dim2 >= self.rank() {
            anyhow::bail!("dim out of bounds in transpose {:?} {dim1} {dim2}", self.shape())
        }
        let dims = self.dims();
        if dim1 == dim2 {
            dst.inner.copy_from_slice(self.data());
        } else {
            let dst_data = &mut dst.inner;
            let src_data = self.data();
            let (dim1, dim2) = (usize::min(dim1, dim2), usize::max(dim1, dim2));
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
                                dst_data[dst_idx] = src_data[src_idx]
                            }
                        }
                    }
                }
            }
        }
        let mut shape = dims.to_vec();
        shape.swap(dim1, dim2);
        Ok(Tensor { data: CowMut::Borrowed(dst), shape: shape.into() })
    }

    pub fn slice_assign<D: Dim>(
        &mut self,
        src: &Tensor<'_, T>,
        dim: D,
        offset: usize,
    ) -> Result<()> {
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

    pub fn copy(&self) -> Result<Tensor<'static, T>> {
        let storage = match &self.data {
            CowMut::Owned(v) => v.clone(),
            CowMut::Borrowed(v) => Storage::clone(v),
        };
        Ok(Tensor { data: CowMut::Owned(storage), shape: self.shape.clone() })
    }

    pub fn rope(&mut self, cos: &Tensor<'_, T>, sin: &Tensor<'_, T>, pos: usize) -> Result<()> {
        let (b, h, t, d) = self.shape().dims4()?;
        match cos.dims() {
            [_t, d_over_2] if 2 * d_over_2 == d => {}
            s => anyhow::bail!("unexpected shape for rope-cos {s:?} (head-dim {d})"),
        };
        match sin.dims() {
            [_t, d_over_2] if 2 * d_over_2 == d => {}
            s => anyhow::bail!("unexpected shape for rope-sin {s:?} (head-dim {d})"),
        };
        let cos_data = cos.data();
        let sin_data = sin.data();
        rope(self.data_mut(), &cos_data[pos * d / 2..], &sin_data[pos * d / 2..], b, h, t, d)
    }

    pub fn rope_i(&mut self, cos: &Tensor<'_, T>, sin: &Tensor<'_, T>, pos: usize) -> Result<()> {
        let (b, h, t, d) = self.shape().dims4()?;
        match cos.dims() {
            [_t, d_over_2] if 2 * d_over_2 == d => {}
            s => anyhow::bail!("unexpected shape for rope-cos {s:?} (head-dim {d})"),
        };
        match sin.dims() {
            [_t, d_over_2] if 2 * d_over_2 == d => {}
            s => anyhow::bail!("unexpected shape for rope-sin {s:?} (head-dim {d})"),
        };
        let cos_data = cos.data();
        let sin_data = sin.data();
        rope_i(self.data_mut(), &cos_data[pos * d / 2..], &sin_data[pos * d / 2..], b, h, t, d)
    }
}

impl<'a, T: WithDType + num_traits::Float> Tensor<'a, T> {
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
            *d /= T::one() + (T::zero() - *d).exp()
        }
    }

    pub fn apply_causality_mask(&mut self) -> Result<()> {
        let (bh, t1, t2) = self.shape().dims3()?;
        let dst_data = self.data_mut();
        for idx_b in 0..bh {
            for idx1 in 0..t1 {
                for idx2 in idx1 + 1..t2 {
                    let idx = idx_b * t1 * t2 + idx1 * t2 + idx2;
                    dst_data[idx] = T::neg_infinity()
                }
            }
        }
        Ok(())
    }

    pub fn matmul_<V1: crate::TensorOrView<Elem = T>, V2: crate::TensorOrView<Elem = T>>(
        &mut self,
        lhs: &V1,
        rhs: &V2,
        rhs_t: bool,
    ) -> Result<()> {
        let (lhs_b, lhs_m, lhs_k) = match lhs.dims() {
            [a, b] => (1, *a, *b),
            [a, b, c] => (*a, *b, *c),
            _ => anyhow::bail!("unexpected shape for matmul lhs {:?}", &lhs.shape()),
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
        if dst_elems > self.data.inner.len() {
            anyhow::bail!(
                "matmul dst is too small, dst {} < {dst_elems}, lhs {:?} rhs {:?}",
                self.data.inner.len(),
                lhs.shape(),
                rhs.shape()
            )
        }
        let (m, n, k) = (lhs_m, rhs_n, lhs_k);
        self.shape =
            if lhs.rank() == 2 && rhs.rank() == 2 { (m, n).into() } else { (lhs_b, m, n).into() };
        let b_stride = rhs.strides()[0];
        let (dst_rs, dst_cs) = (n, 1);
        let (lhs_stride_m2, lhs_stride_m1) = {
            let l = lhs.strides().len();
            (lhs.strides()[l - 2], lhs.strides()[l - 1])
        };

        let (lhs_rs, lhs_cs) = (lhs_stride_m2, lhs_stride_m1);
        let (rhs_stride_m2, rhs_stride_m1) = {
            let l = rhs.strides().len();
            (rhs.strides()[l - 2], rhs.strides()[l - 1])
        };

        let (rhs_rs, rhs_cs) =
            if rhs_t { (rhs_stride_m1, rhs_stride_m2) } else { (rhs_stride_m2, rhs_stride_m1) };

        let lhs_data = lhs.data();
        let rhs_data = rhs.data();

        for b_idx in 0..lhs_b {
            let dst = &mut self.data.inner[b_idx * m * n..(b_idx + 1) * m * n];
            let lhs = &lhs_data[b_idx * m * k..(b_idx + 1) * m * k];
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
                    /* alpha: T = */ T::zero(),
                    /* beta: T = */ T::one(),
                    /* conj_dst: bool = */ false,
                    /* conj_lhs: bool = */ false,
                    /* conj_rhs: bool = */ false,
                    gemm::Parallelism::Rayon(crate::tensor::get_num_threads()),
                )
            }
        }
        Ok(())
    }
}

pub fn matmul<
    'a,
    T: WithDType + num_traits::Float,
    V1: crate::TensorOrView<Elem = T>,
    V2: crate::TensorOrView<Elem = T>,
>(
    dst: &'a mut Storage<T>,
    lhs: &V1,
    rhs: &V2,
    rhs_t: bool,
) -> Result<Tensor<'a, T>> {
    // TODO: Use the proper shape here rather than relying on matmul to do the reshape?
    let mut dst = Tensor::new(dst, 1)?;
    dst.matmul_(lhs, rhs, rhs_t)?;
    Ok(dst)
}

impl<'a> Tensor<'a, f32> {
    pub fn softmax<'b>(&self, dst: &'b mut Storage<f32>) -> Result<Tensor<'b, f32>> {
        if self.shape.elem_count() > dst.inner.len() {
            anyhow::bail!("missing capacity for softmax {} {:?}", dst.inner.len(), self.shape)
        }
        let dim_m1 = self.dim(crate::D::Minus1)?;
        let shape = self.shape.clone();
        softmax(&mut dst.inner, self.data(), dim_m1)?;
        Ok(Tensor { data: CowMut::Borrowed(dst), shape })
    }
}

impl<T: WithDType> Tensor<'static, T> {
    // Create a tensor with an owned storage.
    pub fn cst<S: Into<Shape>>(t: T, shape: S) -> Result<Self> {
        let shape = shape.into();
        let data = Storage { inner: vec![t; shape.elem_count()] };
        Ok(Self { data: CowMut::Owned(data), shape })
    }

    pub fn owned<S: Into<Shape>>(data: Vec<T>, shape: S) -> Result<Self> {
        let shape = shape.into();
        if shape.elem_count() > data.len() {
            anyhow::bail!("not enough elements in input vector {} for shape {shape:?}", data.len())
        }
        let data = Storage { inner: data };
        Ok(Self { data: CowMut::Owned(data), shape })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn copy2d<T: Copy>(
    dst: &mut [T],
    src: &[T],
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

pub(crate) fn softmax<T: WithDType + num_traits::Float + Into<f32> + From<f32>>(
    dst: &mut [T],
    src: &[T],
    dim_m1: usize,
) -> Result<()> {
    src.par_chunks(dim_m1).zip(dst.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
        let mut max = T::neg_infinity();
        for &v in src.iter() {
            max = T::max(v, max)
        }
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            *d = (*s - max).exp();
        }
        let sum_exp = dst.iter().map(|v| <T as Into<f32>>::into(*v)).sum::<f32>();
        for d in dst.iter_mut() {
            *d = <T as From<f32>>::from(<T as Into<f32>>::into(*d) / sum_exp)
        }
    });
    Ok(())
}

pub(crate) fn rope<T: WithDType>(
    dst: &mut [T],
    cos: &[T],
    sin: &[T],
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

pub(crate) fn rope_i<T: WithDType>(
    dst: &mut [T],
    cos: &[T],
    sin: &[T],
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

pub(crate) fn get_num_threads() -> usize {
    use std::str::FromStr;
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| usize::from_str(&s).ok()) {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}
