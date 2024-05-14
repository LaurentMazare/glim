use crate::Tensor;
use anyhow::Result;

pub struct Cache {
    inner: Tensor,
    seq_len: usize,
    // TODO: make it a ring buffer + attention sink?
}

impl Cache {
    pub fn new(b_size: usize, n_head: usize, max_seq_len: usize, head_dim: usize) -> Result<Self> {
        let inner = Tensor::cst(0., (b_size, n_head, max_seq_len, head_dim))?;
        Ok(Self { inner, seq_len: 0 })
    }

    pub fn append(&mut self, src: &Tensor) -> Result<()> {
        let (b, h, t, d) = self.inner.dims4()?;
        let (src_b, src_h, src_t, src_d) = src.dims4()?;
        if b != src_b || h != src_h || src_d != d {
            anyhow::bail!(
                "unexpected shapes in kv-cache {:?} {:?}",
                self.inner.shape(),
                src.shape()
            )
        }
        if src_t + self.seq_len > t {
            anyhow::bail!(
                "kv-cache is too short {:?} {:?} {}",
                self.inner.shape(),
                src.shape(),
                self.seq_len
            )
        }
        for idx_b in 0..b {
            for idx_h in 0..h {
                let _dst_offest = idx_b * h * t * d + idx_h * t * d + self.seq_len * t;
                let _src_offest = idx_b * h * src_t * d + idx_h * src_t * d;
                // TODO: blit src_t * d elements
            }
        }
        self.seq_len += src_t;
        Ok(())
    }
}

pub struct KvCache {
    k_cache: Cache,
    v_cache: Cache,
}

impl KvCache {
    pub fn new(b_size: usize, n_head: usize, max_seq_len: usize, head_dim: usize) -> Result<Self> {
        let k_cache = Cache::new(b_size, n_head, max_seq_len, head_dim)?;
        let v_cache = Cache::new(b_size, n_head, max_seq_len, head_dim)?;
        Ok(Self { k_cache, v_cache })
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        self.k_cache.append(k)?;
        self.v_cache.append(v)?;
        Ok(())
    }
}
