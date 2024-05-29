use crate::{tensor, Shape};
use anyhow::Result;
use rayon::prelude::*;

type Storage = crate::storage::Storage<f32>;
type Tensor = crate::Tensor<'static, f32>;

#[derive(Debug, Clone)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
    pub norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_theta: f32,
}

impl Config {
    pub fn tiny_15m() -> Self {
        Self {
            dim: 288,
            hidden_dim: 768,
            n_layers: 6,
            n_heads: 6,
            n_kv_heads: 6,
            vocab_size: 32000,
            seq_len: 256,
            norm_eps: 1e-5,
            max_seq_len: 256,
            rope_theta: 10000.,
        }
    }

    fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }
}

struct Linear {
    w: Tensor,
    #[allow(unused)]
    in_c: usize,
    #[allow(unused)]
    out_c: usize,
}

impl Linear {
    fn new(w: Tensor, in_c: usize, out_c: usize) -> Result<Self> {
        if w.dims() != [out_c, in_c] {
            anyhow::bail!("unexpected shape in linear {:?}, in: {in_c}, out: {out_c}", w.shape())
        }
        Ok(Self { w, in_c, out_c })
    }

    fn fwd<'a>(
        &self,
        dst: &'a mut Storage,
        src: &tensor::Tensor<'_, f32>,
    ) -> Result<tensor::Tensor<'a, f32>> {
        // TODO: use the proper dst shape here though 1 will work as matmul will reshape its dst.
        let mut dst = tensor::Tensor::new(dst, 1)?;
        self.fwd_inplace(&mut dst, src)?;
        Ok(dst)
    }

    fn fwd_inplace(
        &self,
        dst: &mut tensor::Tensor<'_, f32>,
        src: &tensor::Tensor<'_, f32>,
    ) -> Result<()> {
        dst.matmul(src, &self.w, true)
    }
}

struct RmsNorm {
    alpha: Tensor,
    eps: f32,
    dim_m1: usize,
}

impl RmsNorm {
    fn new(w: Tensor, eps: f32, dim_m1: usize) -> Result<Self> {
        if w.dims() != [dim_m1] {
            anyhow::bail!("unexpected shape in rms_norm {:?} {dim_m1}", w.shape())
        }
        Ok(Self { alpha: w, dim_m1, eps })
    }

    fn fwd<'a>(
        &self,
        dst: &'a mut Storage,
        src: &tensor::Tensor<'_, f32>,
    ) -> Result<tensor::Tensor<'a, f32>> {
        let mut dst = tensor::Tensor::new(dst, src.shape())?;
        self.fwd_inplace(&mut dst, src)?;
        Ok(dst)
    }

    fn fwd_inplace(
        &self,
        dst: &mut tensor::Tensor<'_, f32>,
        src: &tensor::Tensor<'_, f32>,
    ) -> Result<()> {
        let alpha = self.alpha.data();
        let src = src.data();
        let dst = dst.data_mut();
        let dim_m1 = self.dim_m1;
        src.par_chunks(dim_m1).zip(dst.par_chunks_mut(dim_m1)).for_each(|(src, dst)| {
            let sum2 = src.iter().map(|&v| v * v).sum::<f32>();
            let m = (sum2 / dim_m1 as f32 + self.eps).sqrt();
            for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
                *d = *s / m * *alpha
            }
        });
        Ok(())
    }
}

struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    head_dim: usize,
}

struct Layer {
    rms1: RmsNorm,
    attn: Attention,
    rms2: RmsNorm,
    mlp: Mlp,
}

pub struct Model {
    embedding: Tensor,
    layers: Vec<Layer>,
    ln_f: RmsNorm,
    lm_head: Linear,
    config: Config,
}

pub struct State {
    xs: Tensor,
    fc1_xs: Storage,
    fc2_xs: Storage,
    rms_xs: Storage,
    attn_q: Storage,
    attn_k: Storage,
    attn_v: Storage,
    attn_q_t: Tensor,
    attn_k_t: Tensor,
    attn_v_t: Tensor,
    attn_sm: Tensor,
    attn_scores: Tensor,
    attn_xs: Tensor,
    attn_xs_t: Tensor,
    logits: Tensor,
    cos: Tensor,
    sin: Tensor,
    b_sz: usize,
    kv_caches: Vec<crate::kv_cache::KvCache<'static, f32>>,
}

impl State {
    pub fn new(b_sz: usize, cfg: &Config) -> Result<Self> {
        let seq_len = 1;
        let max_seq_len = cfg.max_seq_len;
        let logits = Tensor::cst(0., (b_sz, seq_len, cfg.vocab_size))?;
        let xs = Tensor::cst(0., (b_sz, seq_len, cfg.dim))?;
        let fc1_xs = Storage::cst(0., b_sz * seq_len * cfg.hidden_dim)?;
        let fc2_xs = Storage::cst(0., b_sz * seq_len * cfg.hidden_dim)?;
        let rms_xs = Storage::cst(0., b_sz * seq_len * cfg.dim)?;
        let attn_xs = Tensor::cst(0., (b_sz * cfg.n_heads, seq_len, cfg.head_dim()))?;
        let attn_xs_t = Tensor::cst(0., (b_sz, seq_len, cfg.n_heads * cfg.head_dim()))?;
        let attn_scores = Tensor::cst(0., (b_sz * cfg.n_heads, seq_len, max_seq_len))?;
        let attn_sm = Tensor::cst(0., (b_sz * cfg.n_heads, seq_len, max_seq_len))?;
        let attn_q = Storage::cst(0., b_sz * seq_len * cfg.dim)?;
        let attn_k = Storage::cst(0., b_sz * seq_len * cfg.dim)?;
        let attn_v = Storage::cst(0., b_sz * seq_len * cfg.dim)?;
        let attn_q_t = Tensor::cst(0., (b_sz, seq_len, cfg.dim))?;
        let attn_k_t = Tensor::cst(0., (b_sz, seq_len, cfg.dim))?;
        let attn_v_t = Tensor::cst(0., (b_sz, seq_len, cfg.dim))?;
        let head_dim = cfg.head_dim();
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::owned(theta, (1, head_dim / 2))?;
        let idx_theta = Tensor::owned(
            (0..max_seq_len).map(|v| v as f32).collect::<Vec<_>>(),
            (max_seq_len, 1),
        )?;
        let mut mm = Tensor::cst(0., theta.elem_count() * idx_theta.elem_count())?;
        mm.matmul(&idx_theta, &theta, false)?;
        let mut cos = mm.copy()?;
        cos.cos();
        let mut sin = mm.copy()?;
        sin.sin();

        let mut kv_caches = Vec::with_capacity(cfg.n_layers);
        for _layer_idx in 0..cfg.n_layers {
            let kv_cache =
                crate::kv_cache::KvCache::new(2, (b_sz, cfg.n_heads, max_seq_len, cfg.head_dim()))?;
            kv_caches.push(kv_cache)
        }

        Ok(Self {
            xs,
            fc1_xs,
            fc2_xs,
            rms_xs,
            attn_xs,
            attn_xs_t,
            attn_scores,
            attn_sm,
            attn_q,
            attn_v,
            attn_k,
            attn_q_t,
            attn_v_t,
            attn_k_t,
            logits,
            cos,
            sin,
            b_sz,
            kv_caches,
        })
    }

    pub fn logits(&self) -> &Tensor {
        &self.logits
    }
}

impl Model {
    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn fwd(&self, tokens: &[u32], state: &mut State) -> Result<()> {
        let (b_sz, seq_len) = (1, tokens.len());
        if state.b_sz != b_sz {
            anyhow::bail!("batch size mismatch {} {b_sz}", state.b_sz)
        }
        if seq_len != 1 {
            anyhow::bail!("seq-len is not one, {seq_len}")
        }
        let h = self.config.n_heads;
        let d = self.config.dim / h;
        for (i, token) in tokens.iter().enumerate() {
            let h = self.config.dim;
            let token = *token as usize;
            state.xs.data_mut()[i * h..(i + 1) * h]
                .copy_from_slice(&self.embedding.data()[token * h..(token + 1) * h]);
        }
        let pos = state.kv_caches[0].k().current_seq_len();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            {
                {
                    let rms_xs = layer.rms1.fwd(&mut state.rms_xs, &state.xs)?;
                    // Attention
                    let mut attn_q = layer.attn.q_proj.fwd(&mut state.attn_q, &rms_xs)?;
                    let mut attn_k = layer.attn.k_proj.fwd(&mut state.attn_k, &rms_xs)?;
                    let mut attn_v = layer.attn.v_proj.fwd(&mut state.attn_v, &rms_xs)?;

                    attn_q.reshape((b_sz, seq_len, h, d))?;
                    state.attn_q_t.transpose(&attn_q, 1, 2)?;
                    state.attn_q_t.rope_i(&state.cos, &state.sin, pos)?;
                    state.attn_q_t.reshape((b_sz * h, seq_len, d))?;

                    attn_k.reshape((b_sz, seq_len, h, d))?;
                    state.attn_k_t.transpose(&attn_k, 1, 2)?;
                    state.attn_k_t.rope_i(&state.cos, &state.sin, pos)?;

                    attn_v.reshape((b_sz, seq_len, h, d))?;
                    state.attn_v_t.transpose(&attn_v, 1, 2)?;
                    // kv-cache
                    let (k, v) =
                        state.kv_caches[layer_idx].append(&state.attn_k_t, &state.attn_v_t)?;
                    let k = k.flatten(0, 1)?;
                    let v = v.flatten(0, 1)?;
                    // TODO: repeat-kv
                    state.attn_scores.matmul(&state.attn_q_t, &k, true)?;
                    state.attn_scores.scale(1f32 / (layer.attn.head_dim as f32).sqrt());
                    // no causal mask, as the sequence length is 1.
                    // state.attn_scores.apply_causality_mask()?;
                    state.attn_sm.softmax(&state.attn_scores)?;
                    // get values, attn_sm has shape (b, h, t, t), v has shape (b, h, t, d)
                    state.attn_xs.matmul(&state.attn_sm, &v, false)?;
                    state.attn_xs.reshape((b_sz, h, seq_len, d))?;
                    state.attn_xs_t.transpose(&state.attn_xs, 1, 2)?;
                    state.attn_xs_t.reshape((b_sz, seq_len, h * d))?;
                }
                {
                    let o = layer.attn.o_proj.fwd(&mut state.rms_xs, &state.attn_xs_t)?;
                    state.xs.add(&o)?;
                }
            }

            {
                let rms_xs = layer.rms2.fwd(&mut state.rms_xs, &state.xs)?;
                // MLP
                let mut fc1_xs = layer.mlp.c_fc1.fwd(&mut state.fc1_xs, &rms_xs)?;
                let fc2_xs = layer.mlp.c_fc2.fwd(&mut state.fc2_xs, &rms_xs)?;
                fc1_xs.silu();
                fc1_xs.mult(&fc2_xs)?;
                let o = layer.mlp.c_proj.fwd(&mut state.rms_xs, &fc1_xs)?;
                state.xs.add(&o)?;
            }
        }
        let rms_xs = self.ln_f.fwd(&mut state.rms_xs, &state.xs)?;
        self.lm_head.fwd_inplace(&mut state.logits, &rms_xs)?;
        Ok(())
    }

    pub fn new<P: AsRef<std::path::Path>>(config: Config, p: P) -> Result<Self> {
        let data = std::fs::read(p)?;
        let data = safetensors::SafeTensors::deserialize(&data)?;
        let get = |name: &str| {
            let data = data.tensor(name)?;
            let shape: Shape = data.shape().into();
            let mut data = std::io::Cursor::new(data.data());
            let mut f32_data = vec![0f32; shape.elem_count()];
            byteorder::ReadBytesExt::read_f32_into::<byteorder::LittleEndian>(
                &mut data,
                &mut f32_data,
            )?;
            let data = Tensor::owned(f32_data, shape)?;
            Ok::<_, anyhow::Error>(data)
        };
        let embedding = get("tok_embeddings.weight")?;
        let mut layers = Vec::with_capacity(config.n_layers);
        for idx_layer in 0..config.n_layers {
            let rms1 = {
                let alpha = get(&format!("layers.{idx_layer}.attention_norm.weight"))?;
                RmsNorm::new(alpha, config.norm_eps, config.dim)?
            };
            let rms2 = {
                let alpha = get(&format!("layers.{idx_layer}.ffn_norm.weight"))?;
                RmsNorm::new(alpha, config.norm_eps, config.dim)?
            };
            let proj = |name| {
                let w = get(&format!("layers.{idx_layer}.attention.{name}.weight"))?;
                Linear::new(w, config.dim, config.dim)
            };
            let attn = Attention {
                head_dim: config.head_dim(),
                q_proj: proj("wq")?,
                k_proj: proj("wk")?,
                v_proj: proj("wv")?,
                o_proj: proj("wo")?,
            };
            let c_fc1 = {
                let w = get(&format!("layers.{idx_layer}.feed_forward.w1.weight"))?;
                Linear::new(w, config.dim, config.hidden_dim)?
            };
            let c_fc2 = {
                let w = get(&format!("layers.{idx_layer}.feed_forward.w3.weight"))?;
                Linear::new(w, config.dim, config.hidden_dim)?
            };
            let c_proj = {
                let w = get(&format!("layers.{idx_layer}.feed_forward.w2.weight"))?;
                Linear::new(w, config.hidden_dim, config.dim)?
            };
            let mlp = Mlp { c_fc1, c_fc2, c_proj };
            layers.push(Layer { rms1, attn, rms2, mlp })
        }
        let ln_f = {
            let alpha = get("norm.weight")?;
            RmsNorm::new(alpha, config.norm_eps, config.dim)?
        };
        let lm_head = {
            let w = get("output.weight")?;
            Linear::new(w, config.dim, config.vocab_size)?
        };
        Ok(Self { embedding, layers, ln_f, lm_head, config })
    }
}
