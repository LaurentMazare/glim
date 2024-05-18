use crate::{shape::Dim, Shape, Tensor, TensorView};
use anyhow::Result;

#[derive(Clone)]
pub struct Cache {
    all_data: Tensor,
    dim: usize,
    current_seq_len: usize,
    max_seq_len: usize,
}

impl Cache {
    pub fn new<S: Into<Shape>, D: Dim>(dim: D, shape: S) -> Result<Self> {
        let shape = shape.into();
        let dim = dim.to_index(&shape, "kv-cache")?;
        let max_seq_len = shape.dims()[dim];
        let all_data = Tensor::cst(0., shape)?;
        Ok(Self { all_data, dim, current_seq_len: 0, max_seq_len })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn all_data(&self) -> &Tensor {
        &self.all_data
    }

    pub fn current_data(&self) -> Result<TensorView<'_>> {
        let view = TensorView::from(&self.all_data);
        view.narrow(self.dim, 0, Some(self.current_seq_len))
    }

    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let seq_len = src.dim(self.dim)?;
        if self.current_seq_len + seq_len > self.max_seq_len {
            anyhow::bail!(
                "kv-cache: above max-seq-len {}+{seq_len}>{}",
                self.current_seq_len,
                self.max_seq_len
            )
        }
        self.all_data.slice_assign(src, self.dim, self.current_seq_len)?;
        self.current_seq_len += seq_len;
        Ok(())
    }
}

#[derive(Clone)]
pub struct KvCache {
    k: Cache,
    v: Cache,
}

impl KvCache {
    pub fn new<S: Into<Shape>, D: Dim>(dim: D, shape: S) -> Result<Self> {
        let shape = shape.into();
        let dim = dim.to_index(&shape, "kv-cache")?;
        let k = Cache::new(dim, &shape)?;
        let v = Cache::new(dim, &shape)?;
        Ok(Self { k, v })
    }

    pub fn k(&self) -> Result<TensorView<'_>> {
        self.k.current_data()
    }

    pub fn v(&self) -> Result<TensorView<'_>> {
        self.v.current_data()
    }

    pub fn append<'a>(
        &'a mut self,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(TensorView<'a>, TensorView<'a>)> {
        self.k.append(k)?;
        self.v.append(v)?;
        let k = self.k.current_data()?;
        let v = self.v.current_data()?;
        Ok((k, v))
    }
}
