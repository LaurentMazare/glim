use crate::tensor::{rope, rope_i, softmax};
use crate::{Dim, Shape};
use anyhow::Result;
use half::{bf16, f16};

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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
}

pub trait WithDType: Sized + Copy + num_traits::NumAssign + 'static {
    const DTYPE: DType;
}

impl WithDType for f16 {
    const DTYPE: DType = DType::F16;
}

impl WithDType for bf16 {
    const DTYPE: DType = DType::BF16;
}

impl WithDType for f32 {
    const DTYPE: DType = DType::F32;
}

#[derive(Clone)]
pub struct Storage<T: WithDType> {
    pub inner: Vec<T>,
}

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
            self.data_mut().copy_from_slice(src.data());
            return Ok(());
        }
        let dst_data = self.data_mut();
        let src_data = src.data();
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
                            dst_data[dst_idx] = src_data[src_idx]
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

    // TODO: use a TensorView or a StridedTensor.
    pub fn matmul(&mut self, lhs: &Self, rhs: &Self, rhs_t: bool) -> Result<()> {
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
        /* let b_stride = rhs.strides()[0]; */
        let b_stride = k * n;
        let (dst_rs, dst_cs) = (n, 1);
        let (lhs_rs, lhs_cs) = (k, 1);
        let (rhs_stride_m2, rhs_stride_m1) = { (n, 1) };
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

impl<'a> Tensor<'a, f32> {
    pub fn softmax(&mut self, src: &Self) -> Result<()> {
        if src.shape.elem_count() > self.capacity() {
            anyhow::bail!("missing capacity for softmax {} {:?}", self.capacity(), src.shape)
        }
        self.shape = src.shape.clone();
        let dim_m1 = self.dim(crate::D::Minus1)?;
        softmax(self.data_mut(), src.data(), dim_m1)
    }

    pub fn rope(&mut self, cos: &Self, sin: &Self, pos: usize) -> Result<()> {
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

    pub fn rope_i(&mut self, cos: &Self, sin: &Self, pos: usize) -> Result<()> {
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

impl<T: WithDType> Tensor<'static, T> {
    pub fn owned<S: Into<Shape>>(t: T, shape: S) -> Self {
        let shape = shape.into();
        let data = Storage { inner: vec![t; shape.elem_count()] };
        Self { data: CowMut::Owned(data), shape }
    }
}
