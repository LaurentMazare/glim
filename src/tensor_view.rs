use crate::{shape::Dim, BackendSlice, Shape, Tensor, WithDType};
use anyhow::Result;

#[derive(Clone)]
pub struct TensorView<'a, T: WithDType, B: ?Sized + BackendSlice<T>> {
    inner: &'a Tensor<'a, T, B>,
    shape: Shape,
    strides: Vec<usize>,
    start_offset: usize,
}

impl<'a, T: WithDType, B: ?Sized + BackendSlice<T>> From<&'a Tensor<'a, T, B>>
    for TensorView<'a, T, B>
{
    fn from(inner: &'a Tensor<'a, T, B>) -> Self {
        let shape = inner.shape().clone();
        let strides = shape.stride_contiguous();
        Self { inner, shape, strides, start_offset: 0 }
    }
}

impl<'a, T: WithDType, B: ?Sized + BackendSlice<T>> TensorView<'a, T, B> {
    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    pub fn inner(&self) -> &Tensor<'a, T, B> {
        self.inner
    }

    pub fn data(&self) -> &B {
        &self.inner.data().index(Some(self.start_offset), None)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn elem_count(&self) -> usize {
        self.shape.elem_count()
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

    /// Flatten dimensions d1 to d2 (inclusive on both sides).
    pub fn flatten<D1: Dim, D2: Dim>(&self, d1: D1, d2: D2) -> Result<Self> {
        let d1 = d1.to_index(&self.shape, "flatten")?;
        let d2 = d2.to_index(&self.shape, "flatten")?;
        if d2 < d1 {
            anyhow::bail!("flatten incorrect dim ordering {d1} {d2}")
        }
        let dims = self.dims();
        let strides = self.strides();
        for i in d1..d2 {
            if strides[i + 1] * dims[i + 1] != strides[i] {
                anyhow::bail!(
                    "cannot flatten, block is not contiguous {dims:?} {strides:?} {d1} {d2}"
                )
            }
        }
        let d = dims[d1..d2 + 1].iter().product();
        let dst_dims = [&dims[..d1], &[d], &dims[d2 + 1..]].concat();
        let dst_strides = [&strides[..d1], &strides[d2..]].concat();
        Ok(Self {
            inner: self.inner,
            shape: dst_dims.into(),
            strides: dst_strides,
            start_offset: self.start_offset,
        })
    }

    /// Expand the specified dimension into a list of subdimensions.
    pub fn expand<D: Dim, S: Into<Shape>>(&self, d: D, s: S) -> Result<Self> {
        let s = s.into();
        let d = d.to_index(&self.shape, "expand")?;
        let dims = self.dims();
        let strides = self.strides();
        if dims[d] != s.elem_count() {
            anyhow::bail!("expand incorrect number of elements in target {s:?} {}", dims[d])
        }
        let dst_dims = [&dims[..d], s.dims(), &dims[d + 1..]].concat();
        let s_strides = s.stride_contiguous();
        let dst_strides = [&strides[..d], &s_strides, &strides[d + 1..]].concat();
        Ok(Self {
            inner: self.inner,
            shape: dst_dims.into(),
            strides: dst_strides,
            start_offset: self.start_offset,
        })
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

pub trait TensorOrView<T: WithDType, B: ?Sized + BackendSlice<T>> {
    fn shape(&self) -> &Shape;
    fn strides(&self) -> std::borrow::Cow<'_, [usize]>;
    fn data(&self) -> &B;
    fn rank(&self) -> usize {
        self.shape().rank()
    }
    fn dims(&self) -> &[usize] {
        self.shape().dims()
    }
}

impl<'a, T: WithDType, B: ?Sized + BackendSlice<T>> TensorOrView<T, B> for Tensor<'a, T, B> {
    fn shape(&self) -> &Shape {
        self.shape()
    }

    fn data(&self) -> &B {
        self.data()
    }

    fn strides(&self) -> std::borrow::Cow<'_, [usize]> {
        std::borrow::Cow::Owned(self.shape().stride_contiguous())
    }
}

impl<'a, T: WithDType, B: ?Sized + BackendSlice<T>> TensorOrView<T, B> for TensorView<'a, T, B> {
    fn shape(&self) -> &Shape {
        self.shape()
    }
    fn data(&self) -> &B {
        self.inner.data()
    }
    fn strides(&self) -> std::borrow::Cow<'_, [usize]> {
        std::borrow::Cow::Borrowed(self.strides())
    }
}
