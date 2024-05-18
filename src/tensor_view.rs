use crate::{Shape, Tensor};
use anyhow::Result;

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

    pub fn narrow<D: crate::shape::Dim>(
        &self,
        dim: D,
        start: usize,
        len: Option<usize>,
    ) -> Result<Self> {
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
}
