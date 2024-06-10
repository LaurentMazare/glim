use crate::{tensor, BackendF, Shape, Tensor};
use anyhow::Result;
use half::f16;

type TensorS<B> = crate::TensorS<f16, B>;

#[derive(Debug, Clone)]
pub struct Config {
    // `dim` is `hidden_size` in transformers
    pub dim: usize,
    // `hidden_dim` is `intermediate_size` in transformers
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
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
            norm_eps: 1e-5,
            max_seq_len: 256,
            rope_theta: 10000.,
        }
    }

    pub fn tiny_110m() -> Self {
        Self {
            dim: 768,
            hidden_dim: 2048,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 12,
            vocab_size: 32000,
            norm_eps: 1e-5,
            max_seq_len: 1024,
            rope_theta: 10000.,
        }
    }

    pub fn llama2_7b() -> Self {
        Self {
            dim: 4096,
            hidden_dim: 11008,
            n_layers: 32,
            n_heads: 32,
            n_kv_heads: 32,
            vocab_size: 32000,
            norm_eps: 1e-5,
            max_seq_len: 4096,
            rope_theta: 10000.,
        }
    }

    fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }
}

struct Linear<B: BackendF<f16> + 'static> {
    w: TensorS<B>,
    #[allow(unused)]
    in_c: usize,
    #[allow(unused)]
    out_c: usize,
}

impl<B: BackendF<f16>> Linear<B> {
    fn new(w: TensorS<B>, in_c: usize, out_c: usize) -> Result<Self> {
        if w.dims() != [out_c, in_c] {
            anyhow::bail!("unexpected shape in linear {:?}, in: {in_c}, out: {out_c}", w.shape())
        }
        Ok(Self { w, in_c, out_c })
    }

    fn fwd<'a>(&self, dst: &'a mut B, src: &Tensor<'_, f16, B>) -> Result<Tensor<'a, f16, B>> {
        // TODO: use the proper dst shape here though 1 will work as matmul will reshape its dst.
        let mut dst = Tensor::new(dst, 1)?;
        self.fwd_inplace(&mut dst, src)?;
        Ok(dst)
    }

    fn fwd_inplace(&self, dst: &mut Tensor<'_, f16, B>, src: &Tensor<'_, f16, B>) -> Result<()> {
        dst.matmul_(src, &self.w, true)
    }
}

struct RmsNorm<B: BackendF<f16>> {
    alpha: TensorS<B>,
    eps: f32,
}

impl<B: BackendF<f16>> RmsNorm<B> {
    fn new(w: TensorS<B>, eps: f32, dim_m1: usize) -> Result<Self> {
        if w.dims() != [dim_m1] {
            anyhow::bail!("unexpected shape in rms_norm {:?} {dim_m1}", w.shape())
        }
        Ok(Self { alpha: w, eps })
    }

    fn fwd<'a>(&self, dst: &'a mut B, src: &Tensor<'_, f16, B>) -> Result<Tensor<'a, f16, B>> {
        let mut dst = Tensor::new(dst, src.shape())?;
        self.fwd_inplace(&mut dst, src)?;
        Ok(dst)
    }

    fn fwd_inplace(&self, dst: &mut Tensor<'_, f16, B>, src: &Tensor<'_, f16, B>) -> Result<()> {
        dst.rms_norm(src, &self.alpha, self.eps)
    }
}

struct Mlp<B: BackendF<f16>> {
    c_fc1: Linear<B>,
    c_fc2: Linear<B>,
    c_proj: Linear<B>,
}

struct Attention<B: BackendF<f16>> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    head_dim: usize,
}

struct Layer<B: BackendF<f16>> {
    rms1: RmsNorm<B>,
    attn: Attention<B>,
    rms2: RmsNorm<B>,
    mlp: Mlp<B>,
}

pub struct Model<B: BackendF<f16>> {
    embedding: TensorS<B>,
    layers: Vec<Layer<B>>,
    ln_f: RmsNorm<B>,
    lm_head: Linear<B>,
    config: Config,
}

pub struct State<B: BackendF<f16>> {
    xs: TensorS<B>,
    fc1_xs: B,
    fc2_xs: B,
    rms_xs: B,
    attn_q: B,
    attn_k: B,
    attn_v: B,
    attn_q_t: B,
    attn_k_t: B,
    attn_v_t: B,
    attn_sm: B,
    attn_scores: B,
    attn_xs: B,
    attn_xs_t: B,
    logits: TensorS<B>,
    cos: TensorS<B>,
    sin: TensorS<B>,
    b_sz: usize,
    kv_caches: Vec<crate::kv_cache::KvCache<'static, f16, B>>,
}

impl<B: BackendF<f16>> State<B> {
    pub fn new(b_sz: usize, cfg: &Config, dev: &B::Device) -> Result<Self> {
        let b_cst = |s| B::cst(f16::ZERO, s, dev);
        let t_cst = |s| Tensor::cst(f16::ZERO, s, dev);
        let seq_len = 1;
        let max_seq_len = cfg.max_seq_len;
        let logits = t_cst((b_sz, seq_len, cfg.vocab_size))?;
        let xs = t_cst((b_sz, seq_len, cfg.dim))?;
        let fc1_xs = b_cst(b_sz * seq_len * cfg.hidden_dim)?;
        let fc2_xs = b_cst(b_sz * seq_len * cfg.hidden_dim)?;
        let rms_xs = b_cst(b_sz * seq_len * cfg.dim)?;
        let attn_xs = b_cst(b_sz * cfg.n_heads * seq_len * cfg.head_dim())?;
        let attn_xs_t = b_cst(b_sz * seq_len * cfg.n_heads * cfg.head_dim())?;
        let attn_scores = b_cst(b_sz * cfg.n_heads * seq_len * max_seq_len)?;
        let attn_sm = b_cst(b_sz * cfg.n_heads * seq_len * max_seq_len)?;
        let attn_q = b_cst(b_sz * seq_len * cfg.dim)?;
        let attn_k = b_cst(b_sz * seq_len * cfg.dim)?;
        let attn_v = b_cst(b_sz * seq_len * cfg.dim)?;
        let attn_q_t = b_cst(b_sz * seq_len * cfg.dim)?;
        let attn_k_t = b_cst(b_sz * seq_len * cfg.dim)?;
        let attn_v_t = b_cst(b_sz * seq_len * cfg.dim)?;
        let head_dim = cfg.head_dim();
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| f16::from_f32(1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32)))
            .collect();
        let theta = Tensor::from_vec(theta, (1, head_dim / 2), dev)?;
        let idx_theta = Tensor::from_vec(
            (0..max_seq_len).map(|v| f16::from_f32(v as f32)).collect::<Vec<_>>(),
            (max_seq_len, 1),
            dev,
        )?;
        let mut mm = Tensor::cst(f16::ZERO, theta.elem_count() * idx_theta.elem_count(), dev)?;
        mm.matmul_(&idx_theta, &theta, false)?;
        let mut cos = mm.copy()?;
        cos.cos()?;
        let mut sin = mm.copy()?;
        sin.sin()?;

        let mut kv_caches = Vec::with_capacity(cfg.n_layers);
        for _layer_idx in 0..cfg.n_layers {
            let kv_cache = crate::kv_cache::KvCache::new(
                2,
                (b_sz, cfg.n_heads, max_seq_len, cfg.head_dim()),
                dev,
            )?;
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

    pub fn logits(&self) -> &TensorS<B> {
        &self.logits
    }
}

impl<B: BackendF<f16>> Model<B> {
    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn fwd(&self, tokens: &[u32], state: &mut State<B>) -> Result<()> {
        let (b_sz, seq_len) = (1, tokens.len());
        if state.b_sz != b_sz {
            anyhow::bail!("batch size mismatch {} {b_sz}", state.b_sz)
        }
        if seq_len != 1 {
            anyhow::bail!("seq-len is not one, {seq_len}")
        }
        let h = self.config.n_heads;
        let d = self.config.dim / h;
        state.xs.index_select(&self.embedding, tokens)?;

        let pos = state.kv_caches[0].k().current_seq_len();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            {
                let attn_xs = {
                    let rms_xs = layer.rms1.fwd(&mut state.rms_xs, &state.xs)?;
                    // Attention
                    let mut attn_q = layer.attn.q_proj.fwd(&mut state.attn_q, &rms_xs)?;
                    let mut attn_k = layer.attn.k_proj.fwd(&mut state.attn_k, &rms_xs)?;
                    let mut attn_v = layer.attn.v_proj.fwd(&mut state.attn_v, &rms_xs)?;

                    attn_q.reshape((b_sz, seq_len, h, d))?;
                    let mut attn_q = attn_q.transpose(&mut state.attn_q_t, 1, 2)?;
                    attn_q.rope(&state.cos, &state.sin, pos)?;
                    attn_q.reshape((b_sz * h, seq_len, d))?;

                    attn_k.reshape((b_sz, seq_len, h, d))?;
                    let mut attn_k = attn_k.transpose(&mut state.attn_k_t, 1, 2)?;
                    attn_k.rope(&state.cos, &state.sin, pos)?;

                    attn_v.reshape((b_sz, seq_len, h, d))?;
                    let attn_v = attn_v.transpose(&mut state.attn_v_t, 1, 2)?;
                    // kv-cache
                    let (k, v) = state.kv_caches[layer_idx].append(&attn_k, &attn_v)?;
                    let k = k.flatten(0, 1)?;
                    let v = v.flatten(0, 1)?;
                    // TODO: repeat-kv
                    let mut attn_scores =
                        tensor::matmul(&mut state.attn_scores, &attn_q, &k, true)?;
                    attn_scores.scale(f16::from_f32(1f32 / (layer.attn.head_dim as f32).sqrt()))?;
                    // no causal mask, as the sequence length is 1.
                    // state.attn_scores.apply_causality_mask()?;
                    let attn_sm = attn_scores.softmax(&mut state.attn_sm)?;
                    // get values, attn_sm has shape (b, h, t, t), v has shape (b, h, t, d)
                    let mut attn_xs = tensor::matmul(&mut state.attn_xs, &attn_sm, &v, false)?;
                    attn_xs.reshape((b_sz, h, seq_len, d))?;
                    let mut attn_xs = attn_xs.transpose(&mut state.attn_xs_t, 1, 2)?;
                    attn_xs.reshape((b_sz, seq_len, h * d))?;
                    attn_xs
                };
                let o = layer.attn.o_proj.fwd(&mut state.rms_xs, &attn_xs)?;
                state.xs.add(&o)?;
            }
            {
                let rms_xs = layer.rms2.fwd(&mut state.rms_xs, &state.xs)?;
                // MLP
                let mut fc1_xs = layer.mlp.c_fc1.fwd(&mut state.fc1_xs, &rms_xs)?;
                let fc2_xs = layer.mlp.c_fc2.fwd(&mut state.fc2_xs, &rms_xs)?;
                fc1_xs.silu()?;
                fc1_xs.mult(&fc2_xs)?;
                let o = layer.mlp.c_proj.fwd(&mut state.rms_xs, &fc1_xs)?;
                state.xs.add(&o)?;
            }
        }
        let rms_xs = self.ln_f.fwd(&mut state.rms_xs, &state.xs)?;
        self.lm_head.fwd_inplace(&mut state.logits, &rms_xs)?;
        Ok(())
    }

    pub fn new<P: AsRef<std::path::Path>>(config: Config, dev: &B::Device, p: P) -> Result<Self> {
        let data = std::fs::read(p)?;
        let data = safetensors::SafeTensors::deserialize(&data)?;
        let get = |name: &str| {
            let data = data.tensor(name)?;
            let shape: Shape = data.shape().into();
            let mut data = std::io::Cursor::new(data.data());
            let mut f16_data = vec![0u16; shape.elem_count()];
            byteorder::ReadBytesExt::read_u16_into::<byteorder::LittleEndian>(
                &mut data,
                &mut f16_data,
            )?;
            let f16_data = f16_data.into_iter().map(f16::from_bits).collect::<Vec<_>>();
            let data = Tensor::from_vec(f16_data, shape, dev)?;
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
