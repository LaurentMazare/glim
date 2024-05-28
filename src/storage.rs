use crate::Shape;
use anyhow::Result;

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

pub trait WithDType: Sized + Copy + num_traits::NumAssign {
    const DTYPE: DType;
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
}

impl<T: WithDType> Tensor<'static, T> {
    pub fn owned<S: Into<Shape>>(t: T, shape: S) -> Self {
        let shape = shape.into();
        let data = Storage { inner: vec![t; shape.elem_count()] };
        Self { data: CowMut::Owned(data), shape }
    }
}
