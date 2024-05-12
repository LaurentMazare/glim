use crate::{Shape, Tensor};
use anyhow::Result;
use rayon::prelude::*;

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
        if w.shape() != &Shape::D2(out_c, in_c) {
            anyhow::bail!("unexpected shape in linear {:?}, in: {in_c}, out: {out_c}", w.shape())
        }
        Ok(Self { w, in_c, out_c })
    }

    fn fwd(&self, dst: &mut Tensor, src: &Tensor) -> Result<()> {
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
        if w.shape() != &Shape::D1(dim_m1) {
            anyhow::bail!("unexpected shape in rms_norm {:?} {dim_m1}", w.shape())
        }
        Ok(Self { alpha: w, dim_m1, eps })
    }

    fn fwd(&self, dst: &mut Tensor, src: &Tensor) -> Result<()> {
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
    fc1_xs: Tensor,
    fc2_xs: Tensor,
    rms_xs: Tensor,
    attn_q: Tensor,
    attn_k: Tensor,
    attn_v: Tensor,
    attn_sm: Tensor,
    attn_scores: Tensor,
    attn_xs: Tensor,
    logits: Tensor,
}

impl State {
    pub fn new(b_sz: usize, seqlen: usize, cfg: &Config) -> Result<Self> {
        let logits = Tensor::cst(0., (b_sz, seqlen, cfg.vocab_size))?;
        let xs = Tensor::cst(0., (b_sz, seqlen, cfg.dim))?;
        let fc1_xs = Tensor::cst(0., (b_sz, seqlen, cfg.hidden_dim))?;
        let fc2_xs = Tensor::cst(0., (b_sz, seqlen, cfg.hidden_dim))?;
        let rms_xs = Tensor::cst(0., (b_sz, seqlen, cfg.dim))?;
        let attn_xs = Tensor::cst(0., (b_sz * cfg.n_heads, seqlen, cfg.head_dim()))?;
        let attn_scores = Tensor::cst(0., (b_sz * cfg.n_heads, seqlen, seqlen))?;
        let attn_sm = Tensor::cst(0., (b_sz * cfg.n_heads, seqlen, seqlen))?;
        let attn_q = Tensor::cst(0., (b_sz, seqlen, cfg.dim))?;
        let attn_k = Tensor::cst(0., (b_sz, seqlen, cfg.dim))?;
        let attn_v = Tensor::cst(0., (b_sz, seqlen, cfg.dim))?;
        Ok(Self {
            xs,
            fc1_xs,
            fc2_xs,
            rms_xs,
            attn_xs,
            attn_scores,
            attn_sm,
            attn_v,
            attn_k,
            attn_q,
            logits,
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
        let (b_sz, seqlen) = (1, tokens.len());
        let h = self.config.n_heads;
        let d = self.config.dim / h;
        for (i, token) in tokens.iter().enumerate() {
            let h = self.config.dim;
            let token = *token as usize;
            state.xs.data_mut()[i * h..(i + 1) * h]
                .copy_from_slice(&self.embedding.data()[token * h..(token + 1) * h]);
        }
        for layer in self.layers.iter() {
            layer.rms1.fwd(&mut state.rms_xs, &state.xs)?;
            {
                // Attention
                layer.attn.q_proj.fwd(&mut state.attn_q, &state.rms_xs)?;
                layer.attn.k_proj.fwd(&mut state.attn_k, &state.rms_xs)?;
                layer.attn.v_proj.fwd(&mut state.attn_v, &state.rms_xs)?;
                // TODO: rotary embeddings
                // kv-cache
                // repeat-kv
                // TODO: transpose q, k, v -> (b, h, t, d)
                state.attn_q.reshape((b_sz * h, seqlen, d))?;
                state.attn_k.reshape((b_sz * h, seqlen, d))?;
                state.attn_v.reshape((b_sz * h, seqlen, d))?;
                state.attn_scores.matmul(&state.attn_q, &state.attn_k, true)?;
                state.attn_scores.scale(1f32 / (layer.attn.head_dim as f32).sqrt());
                // causal mask
                state.attn_sm.softmax(&state.attn_scores)?;
                // get values, attn_sm has shape (b, h, t, t), v has shape (b, h, t, d)
                state.attn_xs.matmul(&state.attn_sm, &state.attn_v, false)?;
                // TODO: transpose(1, 2)
                state.attn_xs.reshape((b_sz, seqlen, h * d))?;
                layer.attn.o_proj.fwd(&mut state.rms_xs, &state.attn_xs)?;
            }
            state.xs.add(&state.rms_xs)?;

            layer.rms2.fwd(&mut state.rms_xs, &state.xs)?;
            {
                // MLP
                layer.mlp.c_fc1.fwd(&mut state.fc1_xs, &state.rms_xs)?;
                layer.mlp.c_fc2.fwd(&mut state.fc2_xs, &state.rms_xs)?;
                state.fc1_xs.silu();
                state.fc1_xs.mult(&state.fc2_xs)?;
                layer.mlp.c_proj.fwd(&mut state.rms_xs, &state.fc1_xs)?;
            }
            state.xs.add(&state.rms_xs)?;
        }
        self.ln_f.fwd(&mut state.rms_xs, &state.xs)?;
        self.lm_head.fwd(&mut state.logits, &state.rms_xs)?;
        Ok(())
    }

    pub fn new<P: AsRef<std::path::Path>>(config: Config, p: P) -> Result<Self> {
        let data = std::fs::read(p)?;
        let data = safetensors::SafeTensors::deserialize(&data)?;
        let get = |name: &str| {
            let data = data.tensor(name)?;
            let shape = match data.shape() {
                &[] => Shape::D0,
                &[u] => Shape::D1(u),
                &[u1, u2] => Shape::D2(u1, u2),
                &[u1, u2, u3] => Shape::D3(u1, u2, u3),
                s => anyhow::bail!("unsupported shapes {s:?}"),
            };
            let mut data = std::io::Cursor::new(data.data());
            let mut f32_data = vec![0f32; shape.num_elems()];
            byteorder::ReadBytesExt::read_f32_into::<byteorder::LittleEndian>(
                &mut data,
                &mut f32_data,
            )?;
            let data = Tensor::new(f32_data, shape)?;
            Ok(data)
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