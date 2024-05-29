use crate::{shape::Dim, Shape, Tensor, TensorView, WithDType};
use anyhow::Result;

pub struct Cache<'a, T: WithDType> {
    all_data: Tensor<'a, T>,
    dim: usize,
    current_seq_len: usize,
    max_seq_len: usize,
}

impl<'a, T: WithDType> Cache<'a, T> {
    pub fn new<S: Into<Shape>, D: Dim>(dim: D, shape: S) -> Result<Self> {
        let shape = shape.into();
        let dim = dim.to_index(&shape, "kv-cache")?;
        let max_seq_len = shape.dims()[dim];
        let all_data = Tensor::cst(T::zero(), shape)?;
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

    pub fn all_data(&self) -> &Tensor<'a, T> {
        &self.all_data
    }

    pub fn current_data(&self) -> Result<TensorView<'_, T>> {
        let view = TensorView::from(&self.all_data);
        view.narrow(self.dim, 0, Some(self.current_seq_len))
    }

    pub fn append<'b>(&mut self, src: &Tensor<'b, T>) -> Result<()> {
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

pub struct KvCache<'a, T: WithDType> {
    k: Cache<'a, T>,
    v: Cache<'a, T>,
}

impl<'a, T: WithDType> KvCache<'a, T> {
    pub fn new<S: Into<Shape>, D: Dim>(dim: D, shape: S) -> Result<Self> {
        let shape = shape.into();
        let dim = dim.to_index(&shape, "kv-cache")?;
        let k = Cache::new(dim, &shape)?;
        let v = Cache::new(dim, &shape)?;
        Ok(Self { k, v })
    }

    pub fn k(&self) -> &Cache<'a, T> {
        &self.k
    }

    pub fn v(&self) -> &Cache<'a, T> {
        &self.v
    }

    pub fn append<'b, 'c>(
        &'b mut self,
        k: &Tensor<'c, T>,
        v: &Tensor<'c, T>,
    ) -> Result<(TensorView<'b, T>, TensorView<'b, T>)> {
        self.k.append(k)?;
        self.v.append(v)?;
        let k = self.k.current_data()?;
        let v = self.v.current_data()?;
        Ok((k, v))
    }
}
