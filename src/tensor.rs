use crate::{BackendAlloc, BackendSlice, BackendSliceF, Dim, Shape, WithDType, WithDTypeT};
use anyhow::Result;

pub enum CowMut<'a, T: WithDType, B: BackendSlice<T>> {
    Owned(B::Allocated),
    Borrowed(&'a mut B),
}

impl<'a, T: WithDType, B: BackendSlice<T>> CowMut<'a, T, B> {
    fn as_ref(&self) -> &B {
        match self {
            Self::Owned(o) => o.slice(),
            Self::Borrowed(b) => b,
        }
    }
}

impl<'a, T: WithDType, B: BackendSlice<T>> CowMut<'a, T, B> {
    fn as_mut_ref(&mut self) -> &mut B {
        match self {
            Self::Owned(o) => o.slice_mut(),
            Self::Borrowed(b) => b,
        }
    }
}

// TODO: separate type for StridedTensor? (cannot be owned)
pub struct Tensor<'a, T: WithDType, B: BackendSlice<T>> {
    data: CowMut<'a, T, B>,
    shape: Shape,
}

impl<'a, T: WithDType, B: BackendSlice<T>> Tensor<'a, T, B> {
    pub fn dtype(&self) -> crate::DType {
        T::DTYPE
    }

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
        self.data.as_ref().len()
    }

    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn new<S: Into<Shape>>(data: &'a mut B, shape: S) -> Result<Self> {
        let shape: Shape = shape.into();
        if data.len() < shape.elem_count() {
            anyhow::bail!("not enough elements in storage {} for shape {shape:?}", data.len())
        }
        Ok(Self { data: CowMut::Borrowed(data), shape })
    }

    pub fn zeros(&mut self) -> Result<()> {
        self.data_mut().fill(T::zero())
    }

    pub fn scale(&mut self, m: T) -> Result<()> {
        self.data_mut().scale(m)
    }

    pub fn add(&mut self, src: &Tensor<'_, T, B>) -> Result<()> {
        if self.shape != src.shape {
            anyhow::bail!("shape mismatch in add {:?} {:?}", self.shape, src.shape)
        }
        self.data_mut().add_assign(&src.data())
    }

    pub fn mult(&mut self, src: &Tensor<'_, T, B>) -> Result<()> {
        if self.shape != src.shape {
            anyhow::bail!("shape mismatch in mult {:?} {:?}", self.shape, src.shape)
        }
        self.data_mut().mul_assign(&src.data())
    }

    pub fn data_mut(&mut self) -> &mut B {
        self.data.as_mut_ref()
    }

    pub fn data(&self) -> &B {
        self.data.as_ref()
    }

    // There is no stride so all tensors are always using the C layout
    pub fn reshape(&mut self, s: impl Into<Shape>) -> Result<()> {
        let s: Shape = s.into();
        if s.elem_count() != self.shape.elem_count() {
            anyhow::bail!("num-elems mismatch {s:?} {:?}", self.shape)
        }
        self.shape = s;
        Ok(())
    }

    pub fn transpose<'b>(
        &self,
        dst: &'b mut B,
        dim1: usize,
        dim2: usize,
    ) -> Result<Tensor<'b, T, B>> {
        if dst.len() != self.elem_count() {
            anyhow::bail!(
                "num-elems mismatch in transpose, dst {} src {:?}",
                dst.len(),
                self.shape(),
            )
        }
        if dim1 >= self.rank() || dim2 >= self.rank() {
            anyhow::bail!("dim out of bounds in transpose {:?} {dim1} {dim2}", self.shape())
        }
        dst.transpose(&self.data(), dim1, dim2, self.dims())?;
        let mut shape = self.dims().to_vec();
        shape.swap(dim1, dim2);
        Ok(Tensor { data: CowMut::Borrowed(dst), shape: shape.into() })
    }

    pub fn slice_assign<D: Dim>(
        &mut self,
        src: &Tensor<'_, T, B>,
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
        self.data_mut().copy2d(src.data(), d1, d2, dst_s, src_s, dst_o, src_o);
        Ok(())
    }

    pub fn copy(&self) -> Result<Tensor<'static, T, B>> {
        let storage = self.data.as_ref().copy()?;
        Ok(Tensor { data: CowMut::Owned(storage), shape: self.shape.clone() })
    }

    pub fn rope(
        &mut self,
        cos: &Tensor<'_, T, B>,
        sin: &Tensor<'_, T, B>,
        pos: usize,
    ) -> Result<()> {
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
        self.data.rope(&cos_data[pos * d / 2..], &sin_data[pos * d / 2..], b, h, t, d)
    }

    pub fn rope_i(
        &mut self,
        cos: &Tensor<'_, T, B>,
        sin: &Tensor<'_, T, B>,
        pos: usize,
    ) -> Result<()> {
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
        self.data.rope_i(&cos_data[pos * d / 2..], &sin_data[pos * d / 2..], b, h, t, d)
    }
}

#[cfg(feature = "candle")]
impl<'a, T: WithDType + candle::WithDType> Tensor<'a, T> {
    pub fn to_candle(&self) -> Result<candle::Tensor> {
        let t = candle::Tensor::from_slice(self.data(), self.dims(), &candle::Device::Cpu)?;
        Ok(t)
    }

    pub fn from_candle(t: &candle::Tensor) -> Result<Self> {
        let data = t.flatten_all()?.to_vec1::<T>()?;
        Tensor::owned(data, t.dims())
    }
}

impl<'a, T: WithDTypeT, B: BackendSliceF<T>> Tensor<'a, T, B> {
    pub fn cos(&mut self) -> Result<()> {
        self.data_mut().cos()
    }

    pub fn sin(&mut self) -> Result<()> {
        self.data_mut().sin()
    }

    pub fn silu(&mut self) -> Result<()> {
        self.data_mut().silu()
    }

    pub fn apply_causality_mask(&mut self) -> Result<()> {
        let (bh, t1, t2) = self.shape().dims3()?;
        self.data_mut().apply_causality_mask(bh, t1, t2)
    }

    pub fn matmul_<V1: crate::TensorOrView<T, B>, V2: crate::TensorOrView<T, B>>(
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
        if dst_elems > self.data().len() {
            anyhow::bail!(
                "matmul dst is too small, dst {} < {dst_elems}, lhs {:?} rhs {:?}",
                self.data().len(),
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

        self.data_mut().gemm(
            lhs.data(),
            rhs.data(),
            m,
            n,
            k,
            lhs_b,
            b_stride,
            (dst_cs, dst_rs),
            (lhs_cs, lhs_rs),
            (rhs_cs, rhs_rs),
        )?;
        Ok(())
    }

    pub fn softmax<'b>(&self, dst: &'b mut B) -> Result<Tensor<'b, T, B>> {
        if self.shape.elem_count() > dst.len() {
            anyhow::bail!("missing capacity for softmax {} {:?}", dst.len(), self.shape)
        }
        let dim_m1 = self.dim(crate::D::Minus1)?;
        let shape = self.shape.clone();
        dst.softmax(self.data(), dim_m1)?;
        Ok(Tensor { data: CowMut::Borrowed(dst), shape })
    }
}

pub fn matmul<
    'a,
    T: WithDTypeT,
    B: BackendSliceF<T>,
    V1: crate::TensorOrView<T, B>,
    V2: crate::TensorOrView<T, B>,
>(
    dst: &'a mut B,
    lhs: &V1,
    rhs: &V2,
    rhs_t: bool,
) -> Result<Tensor<'a, T, B>> {
    // TODO: Use the proper shape here rather than relying on matmul to do the reshape?
    let mut dst = Tensor::new(dst, 1)?;
    dst.matmul_(lhs, rhs, rhs_t)?;
    Ok(dst)
}

impl<T: WithDType, B: BackendSlice<T>> Tensor<'static, T, B> {
    // Create a tensor with an owned storage.
    pub fn cst<S: Into<Shape>>(t: T, shape: S) -> Result<Self> {
        let shape: Shape = shape.into();
        let mut data = unsafe { B::Allocated::alloc_uninit(shape.elem_count())? };
        data.slice_mut().fill(t)?;
        Ok(Self { data: CowMut::Owned(data), shape })
    }

    pub fn owned<S: Into<Shape>>(data: B::Allocated, shape: S) -> Result<Self> {
        let shape: Shape = shape.into();
        if shape.elem_count() > data.len() {
            anyhow::bail!("not enough elements in input vector {} for shape {shape:?}", data.len())
        }
        Ok(Self { data: CowMut::Owned(data), shape })
    }
}
