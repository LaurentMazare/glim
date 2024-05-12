use anyhow::Result;
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Shape {
    D0,
    D1(usize),
    D2(usize, usize),
    D3(usize, usize, usize),
    D4(usize, usize, usize, usize),
    D5(usize, usize, usize, usize, usize),
}

impl Shape {
    pub fn num_elems(&self) -> usize {
        match *self {
            Self::D0 => 1,
            Self::D1(u0) => u0,
            Self::D2(u0, u1) => u0 * u1,
            Self::D3(u0, u1, u2) => u0 * u1 * u2,
            Self::D4(u0, u1, u2, u3) => u0 * u1 * u2 * u3,
            Self::D5(u0, u1, u2, u3, u4) => u0 * u1 * u2 * u3 * u4,
        }
    }

    pub fn rank(&self) -> usize {
        match *self {
            Self::D0 => 0,
            Self::D1(_) => 1,
            Self::D2(..) => 2,
            Self::D3(..) => 3,
            Self::D4(..) => 4,
            Self::D5(..) => 5,
        }
    }

    pub fn dims(&self) -> Vec<usize> {
        match *self {
            Self::D0 => vec![],
            Self::D1(u0) => vec![u0],
            Self::D2(u0, u1) => vec![u0, u1],
            Self::D3(u0, u1, u2) => vec![u0, u1, u2],
            Self::D4(u0, u1, u2, u3) => vec![u0, u1, u2, u3],
            Self::D5(u0, u1, u2, u3, u4) => vec![u0, u1, u2, u3, u4],
        }
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Self::D0
    }
}

impl From<usize> for Shape {
    fn from(v: usize) -> Self {
        Self::D1(v)
    }
}

impl From<(usize, usize)> for Shape {
    fn from(v: (usize, usize)) -> Self {
        Self::D2(v.0, v.1)
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from(v: (usize, usize, usize)) -> Self {
        Self::D3(v.0, v.1, v.2)
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from(v: (usize, usize, usize, usize)) -> Self {
        Self::D4(v.0, v.1, v.2, v.3)
    }
}

impl From<(usize, usize, usize, usize, usize)> for Shape {
    fn from(v: (usize, usize, usize, usize, usize)) -> Self {
        Self::D5(v.0, v.1, v.2, v.3, v.4)
    }
}

#[derive(Clone)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
}

impl Tensor {
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn add(&mut self, src: &Self) -> Result<()> {
        if self.shape != src.shape {
            anyhow::bail!("shape mismatch in add {:?} {:?}", self.shape, src.shape)
        }
        src.data.iter().zip(self.data.iter_mut()).for_each(|(src, dst)| *dst += *src);
        Ok(())
    }

    pub fn mult(&mut self, src: &Self) -> Result<()> {
        if self.shape != src.shape {
            anyhow::bail!("shape mismatch in mult {:?} {:?}", self.shape, src.shape)
        }
        src.data.iter().zip(self.data.iter_mut()).for_each(|(src, dst)| *dst *= *src);
        Ok(())
    }

    pub fn scale(&mut self, m: f32) {
        self.data.iter_mut().for_each(|v| *v *= m)
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn new(data: Vec<f32>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        if shape.num_elems() != data.len() {
            anyhow::bail!("unexpected shape in new {shape:?} {}", data.len())
        }
        Ok(Self { data, shape })
    }

    pub fn cst(data: f32, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let data = vec![data; shape.num_elems()];
        Ok(Self { data, shape })
    }

    pub fn silu(&mut self) {
        for d in self.data.iter_mut() {
            *d /= 1. + f32::exp(-*d)
        }
    }

    // There is no stride so all tensors are always using the C layout
    pub fn reshape(&mut self, s: impl Into<Shape>) -> Result<()> {
        let s = s.into();
        if s.num_elems() != self.shape.num_elems() {
            anyhow::bail!("num-elems mismatch {s:?} {:?}", self.shape)
        }
        self.shape = s;
        Ok(())
    }

    pub fn matmul(&mut self, lhs: &Self, rhs: &Self, rhs_t: bool) -> Result<()> {
        let (lhs_b, lhs_m, lhs_k) = match lhs.shape {
            Shape::D2(a, b) => (1, a, b),
            Shape::D3(a, b, c) => (a, b, c),
            _ => anyhow::bail!("unexpected shape for matmul lhs {:?}", &lhs.shape),
        };
        let (rhs_b, rhs_k, rhs_n) = match rhs.shape {
            Shape::D2(a, b) => (1, a, b),
            Shape::D3(a, b, c) => (a, b, c),
            _ => anyhow::bail!("unexpected shape for matmul rhs {:?}", &rhs.shape),
        };
        let (rhs_k, rhs_n) = if rhs_t { (rhs_n, rhs_k) } else { (rhs_k, rhs_n) };
        // Having rhs_b = 1 is ok if dst_b = lhs_b > 1
        if rhs_b != 1 && rhs_b != lhs_b {
            anyhow::bail!(
                "matmul shape mismatch dst {:?}, rhs {:?} {rhs_t}",
                self.shape(),
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
        if dst_elems != self.shape.num_elems() {
            anyhow::bail!(
                "matmul shape mismatch, dst {:?} lhs {:?} rhs {:?}",
                self.shape(),
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

    pub fn num_elems(&self) -> usize {
        self.shape.num_elems()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn transpose(&mut self, src: &Self, dim1: usize, dim2: usize) -> Result<()> {
        if src.num_elems() != self.num_elems() {
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
        Ok(())
    }

    pub fn softmax(&mut self, src: &Self) -> Result<()> {
        if self.shape.num_elems() != src.shape.num_elems() {
            anyhow::bail!("shape mismatch in softmax {:?} {:?}", self.shape, src.shape)
        }
        self.shape = src.shape.clone();
        let dim_m1 = match self.shape {
            Shape::D0 => 1,
            Shape::D1(u)
            | Shape::D2(_, u)
            | Shape::D3(_, _, u)
            | Shape::D4(_, _, _, u)
            | Shape::D5(_, _, _, _, u) => u,
        };
        softmax(&mut self.data, &src.data, dim_m1)
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
            gemm::Parallelism::None,
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
