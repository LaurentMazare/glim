use crate::{shape::Dim, Shape, Tensor};
use anyhow::Result;

#[derive(Clone)]
pub struct TensorView<'a> {
    inner: &'a Tensor,
    shape: Shape,
    strides: Vec<usize>,
    start_offset: usize,
}

impl<'a> From<&'a Tensor> for TensorView<'a> {
    fn from(inner: &'a Tensor) -> Self {
        let shape = inner.shape().clone();
        let strides = shape.stride_contiguous();
        Self { inner, shape, strides, start_offset: 0 }
    }
}

impl<'a> TensorView<'a> {
    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    pub fn inner(&self) -> &Tensor {
        self.inner
    }

    pub fn data(&self) -> &[f32] {
        &self.inner.data()[self.start_offset..]
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.strides)
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    // TODO: proper reshape or squeeze.
    pub fn squeeze0_hack(&self) -> Self {
        let shape = self.shape.dims()[1..].to_vec().into();
        let strides = self.strides[1..].to_vec();
        Self { inner: self.inner, shape, strides, start_offset: self.start_offset }
    }

    pub fn narrow<D: Dim>(&self, dim: D, start: usize, len: Option<usize>) -> Result<Self> {
        let dim = dim.to_index(&self.shape, "narrow")?;
        let mut dims = self.shape.dims().to_vec();
        let len = len.unwrap_or(dims[dim].saturating_sub(start));
        if start + len > dims[dim] {
            anyhow::bail!("out-of-bounds in narrow on {dim}, {start} + {len} > {}", dims[dim])
        }
        dims[dim] = len;
        Ok(Self {
            inner: self.inner,
            start_offset: self.start_offset + self.strides[dim] * start,
            shape: Shape::from(dims),
            strides: self.strides.clone(),
        })
    }

    pub fn transpose<D1: Dim, D2: Dim>(&self, dim1: D1, dim2: D2) -> Result<Self> {
        let dim1 = dim1.to_index(&self.shape, "transpose")?;
        let dim2 = dim2.to_index(&self.shape, "transpose")?;
        let mut strides = self.strides.to_vec();
        let mut dims = self.dims().to_vec();
        dims.swap(dim1, dim2);
        strides.swap(dim1, dim2);
        Ok(Self {
            shape: Shape::from(dims),
            strides,
            start_offset: self.start_offset,
            inner: self.inner,
        })
    }

    pub fn permute(&self, idxs: &[usize]) -> Result<Self> {
        let is_permutation =
            idxs.len() == self.shape.rank() && (0..idxs.len()).all(|i| idxs.contains(&i));
        if !is_permutation {
            anyhow::bail!(
                "dimension mismatch in permute, tensor {:?}, dims: {:?}",
                self.dims(),
                idxs
            )
        }
        let strides = self.strides();
        let dims = self.dims();
        let mut perm_strides = strides.to_vec();
        let mut perm_dims = dims.to_vec();
        for (i, &idx) in idxs.iter().enumerate() {
            perm_strides[i] = strides[idx];
            perm_dims[i] = dims[idx];
        }
        Ok(Self {
            shape: Shape::from(perm_dims),
            strides: perm_strides,
            start_offset: self.start_offset,
            inner: self.inner,
        })
    }
}
