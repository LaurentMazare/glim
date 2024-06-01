use crate::WithDType;
use anyhow::Result;
use rayon::prelude::*;

pub enum CowMut<'a, T> {
    Owned(T),
    Borrowed(&'a mut T),
}

impl<'a, T> std::ops::Deref for CowMut<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(o) => o,
            Self::Borrowed(r) => r,
        }
    }
}

impl<'a, T> std::ops::DerefMut for CowMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(o) => o,
            Self::Borrowed(r) => r,
        }
    }
}

#[derive(Clone)]
pub struct Storage<T: WithDType> {
    pub inner: Vec<T>,
}

impl<T: WithDType> Storage<T> {
    pub fn cst(t: T, elts: usize) -> Result<Self> {
        Ok(Self { inner: vec![t; elts] })
    }
}

impl<T: WithDType> crate::Backend<T> for Vec<T> {
    type Device = ();

    fn device(&self) -> &Self::Device {
        &()
    }

    // fn fill(&self, elem: T) -> Result<()> {
    //     self.fill(elem);
    //     Ok(())
    // }
    fn add_assign(&mut self, s: &Self) -> Result<()> {
        s.iter().zip(self.iter_mut()).for_each(|(src, dst)| *dst += *src);
        Ok(())
    }

    fn mul_assign(&mut self, s: &Self) -> Result<()> {
        s.iter().zip(self.iter_mut()).for_each(|(src, dst)| *dst *= *src);
        Ok(())
    }

    fn scale(&mut self, m: T) -> Result<()> {
        self.iter_mut().for_each(|v| *v *= m);
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
}

impl<T: WithDType + num_traits::Float> crate::BackendF<T> for Vec<T> {
    fn cos(&mut self) -> Result<()> {
        for d in self.iter_mut() {
            *d = d.cos();
        }
        Ok(())
    }

    fn sin(&mut self) -> Result<()> {
        for d in self.iter_mut() {
            *d = d.sin();
        }
        Ok(())
    }

    fn silu(&mut self) -> Result<()> {
        for d in self.iter_mut() {
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
