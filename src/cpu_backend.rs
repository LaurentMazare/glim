use crate::{WithDType, WithDTypeF};
use anyhow::Result;
use rayon::prelude::*;

pub type Storage<T> = Vec<T>;

impl<T: WithDType> crate::Backend<T> for Vec<T> {
    type Device = ();

    unsafe fn alloc_uninit(len: usize, _: &Self::Device) -> Result<Self> {
        Ok(vec![T::zero(); len])
    }

    fn from_vec(v: Vec<T>, _: &Self::Device) -> Result<Self> {
        Ok(v)
    }

    fn data(&self, len: usize) -> Result<std::borrow::Cow<'_, [T]>> {
        Ok(std::borrow::Cow::Borrowed(&self[..len]))
    }

    fn device(&self) -> &Self::Device {
        &()
    }

    fn add_assign(&mut self, s: &Self, l: usize) -> Result<()> {
        s[..l].iter().zip(self[..l].iter_mut()).for_each(|(src, dst)| *dst += *src);
        Ok(())
    }

    fn mul_assign(&mut self, s: &Self, l: usize) -> Result<()> {
        s[..l].iter().zip(self[..l].iter_mut()).for_each(|(src, dst)| *dst *= *src);
        Ok(())
    }

    fn scale(&mut self, m: T, l: usize) -> Result<()> {
        self[..l].iter_mut().for_each(|v| *v *= m);
        Ok(())
    }

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
    ) -> Result<()> {
        for i1 in 0..d1 {
            let dst_idx = i1 * dst_s + dst_o;
            let src_idx = i1 * src_s + src_o;
            let dst = &mut self[dst_idx..dst_idx + d2];
            let src = &src[src_idx..src_idx + d2];
            dst.copy_from_slice(src)
        }
        Ok(())
    }

    fn transpose(&mut self, src: &Self, dim1: usize, dim2: usize, dims: &[usize]) -> Result<()> {
        if dim1 == dim2 {
            self.copy_from_slice(src);
        } else {
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
                                self[dst_idx] = src[src_idx]
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn copy(&self, l: usize) -> Result<Self> {
        Ok(self[..l].to_vec())
    }

    fn fill(&mut self, v: T, l: usize) -> Result<()> {
        self[..l].fill(v);
        Ok(())
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn rope(
        &mut self,
        cos: &Self,
        sin: &Self,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()> {
        if self.len() != b * h * t * d {
            anyhow::bail!("rope unexpected size for dst {} {b} {h} {t} {d}", self.len())
        }
        let cos = &cos[pos * d / 2..];
        let sin = &sin[pos * d / 2..];
        self.par_chunks_mut(t * d).for_each(|dst| {
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
        &mut self,
        cos: &Self,
        sin: &Self,
        b: usize,
        h: usize,
        t: usize,
        d: usize,
        pos: usize,
    ) -> Result<()> {
        if self.len() != b * h * t * d {
            anyhow::bail!("rope-i unexpected size for dst {} {b} {h} {t} {d}", self.len())
        }
        let cos = &cos[pos * d / 2..];
        let sin = &sin[pos * d / 2..];
        self.par_chunks_mut(t * d).for_each(|dst| {
            for i_over_2 in 0..t * d / 2 {
                let i = 2 * i_over_2;
                let (s_i, s_ip) = (dst[i], dst[i + 1]);
                dst[i] = s_i * cos[i_over_2] - s_ip * sin[i_over_2];
                dst[i + 1] = s_i * sin[i_over_2] + s_ip * cos[i_over_2];
            }
        });
        Ok(())
    }

    fn gemm(
        &mut self,
        (lhs, lhs_o): (&Self, usize),
        (rhs, rhs_o): (&Self, usize),
        m: usize,
        n: usize,
        k: usize,
        lhs_b: usize,
        b_stride: usize,
        (dst_cs, dst_rs): (usize, usize),
        (lhs_cs, lhs_rs): (usize, usize),
        (rhs_cs, rhs_rs): (usize, usize),
    ) -> Result<()> {
        let lhs = &lhs[lhs_o..];
        let rhs = &rhs[rhs_o..];
        for b_idx in 0..lhs_b {
            let dst = &mut self[b_idx * m * n..(b_idx + 1) * m * n];
            let lhs = &lhs[b_idx * m * k..(b_idx + 1) * m * k];
            let rhs = &rhs[b_idx * b_stride..];
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
                    gemm::Parallelism::Rayon(get_num_threads()),
                )
            }
        }
        Ok(())
    }

    fn index_select(&mut self, src: &Self, ids: &[u32], h: usize) -> Result<()> {
        for (i, id) in ids.iter().enumerate() {
            let id = *id as usize;
            self[i * h..(i + 1) * h].copy_from_slice(&src[id * h..(id + 1) * h]);
        }
        Ok(())
    }
}

impl<T: WithDTypeF> crate::BackendF<T> for Vec<T> {
    fn cos(&mut self, l: usize) -> Result<()> {
        for d in self[..l].iter_mut() {
            *d = d.cos();
        }
        Ok(())
    }

    fn sin(&mut self, l: usize) -> Result<()> {
        for d in self[..l].iter_mut() {
            *d = d.sin();
        }
        Ok(())
    }

    fn silu(&mut self, l: usize) -> Result<()> {
        for d in self[..l].iter_mut() {
            *d /= T::one() + (T::zero() - *d).exp()
        }
        Ok(())
    }
    fn apply_causality_mask(&mut self, bh: usize, t1: usize, t2: usize) -> Result<()> {
        for idx_b in 0..bh {
            for idx1 in 0..t1 {
                for idx2 in idx1 + 1..t2 {
                    let idx = idx_b * t1 * t2 + idx1 * t2 + idx2;
                    self[idx] = T::neg_infinity()
                }
            }
        }
        Ok(())
    }

    fn softmax(&mut self, src: &Self, dim_m1: usize, d: usize) -> Result<()> {
        let src = &src[..d * dim_m1];
        let dst = &mut self[..d * dim_m1];
        src.par_chunks(dim_m1).zip(dst.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
            let mut max = T::neg_infinity();
            for &v in src.iter() {
                max = T::max(v, max)
            }
            for (s, d) in src.iter().zip(dst.iter_mut()) {
                *d = (*s - max).exp();
            }
            let sum_exp = dst.iter().map(|v| <T as WithDTypeF>::to_f32(*v)).sum::<f32>();
            for d in dst.iter_mut() {
                *d = T::from_f32(d.to_f32() / sum_exp)
            }
        });
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
        let src = &src[..d * dim_m1];
        let dst = &mut self[..d * dim_m1];
        src.par_chunks(dim_m1).zip(dst.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
            let sum2 = src.iter().map(|&v| v.to_f32() * v.to_f32()).sum::<f32>();
            let m = (sum2 / dim_m1 as f32 + eps).sqrt();
            for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
                *d = T::from_f32((*s).to_f32() / m * (*alpha).to_f32())
            }
        });
        Ok(())
    }
}

pub(crate) fn get_num_threads() -> usize {
    use std::str::FromStr;
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS").ok().and_then(|s| usize::from_str(&s).ok()) {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}
