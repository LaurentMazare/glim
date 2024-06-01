use crate::{tensor, Backend, BackendF, Shape, Tensor};
use anyhow::Result;

type Storage = crate::cpu_backend::Storage<f32>;
type TensorS<B> = crate::TensorS<f32, B>;

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

struct Linear<B: BackendF<f32> + 'static> {
    w: TensorS<B>,
    #[allow(unused)]
    in_c: usize,
    #[allow(unused)]
    out_c: usize,
}

impl<B: BackendF<f32>> Linear<B> {
    fn new(w: TensorS<B>, in_c: usize, out_c: usize) -> Result<Self> {
        if w.dims() != [out_c, in_c] {
            anyhow::bail!("unexpected shape in linear {:?}, in: {in_c}, out: {out_c}", w.shape())
        }
        Ok(Self { w, in_c, out_c })
    }

    fn fwd<'a>(&self, dst: &'a mut B, src: &Tensor<'_, f32, B>) -> Result<Tensor<'a, f32, B>> {
        // TODO: use the proper dst shape here though 1 will work as matmul will reshape its dst.
        let mut dst = Tensor::new(dst, 1)?;
        self.fwd_inplace(&mut dst, src)?;
        Ok(dst)
    }

    fn fwd_inplace(&self, dst: &mut Tensor<'_, f32, B>, src: &Tensor<'_, f32, B>) -> Result<()> {
        dst.matmul_(src, &self.w, true)
    }
}

struct RmsNorm<B: BackendF<f32>> {
    alpha: TensorS<B>,
    eps: f32,
    dim_m1: usize,
}

impl<B: BackendF<f32>> RmsNorm<B> {
    fn new(w: TensorS<B>, eps: f32, dim_m1: usize) -> Result<Self> {
        if w.dims() != [dim_m1] {
            anyhow::bail!("unexpected shape in rms_norm {:?} {dim_m1}", w.shape())
        }
        Ok(Self { alpha: w, dim_m1, eps })
    }

    fn fwd<'a>(
        &self,
        dst: &'a mut Storage,
        src: &Tensor<'_, f32, B>,
    ) -> Result<Tensor<'a, f32, B>> {
        let mut dst = Tensor::new(dst, src.shape())?;
        self.fwd_inplace(&mut dst, src)?;
        Ok(dst)
    }

    fn fwd_inplace<B: Backend<f32>>(
        &self,
        dst: &mut Tensor<'_, f32, B>,
        src: &Tensor<'_, f32, B>,
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

struct Mlp<B: BackendF<f32>> {
    c_fc1: Linear<B>,
    c_fc2: Linear<B>,
    c_proj: Linear<B>,
}

struct Attention<B: BackendF<f32>> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    head_dim: usize,
}

struct Layer<B: BackendF<f32>> {
    rms1: RmsNorm<B>,
    attn: Attention<B>,
    rms2: RmsNorm<B>,
    mlp: Mlp<B>,
}

pub struct Model<B: BackendF<f32>> {
    embedding: TensorS<B>,
    layers: Vec<Layer<B>>,
    ln_f: RmsNorm<B>,
    lm_head: Linear<B>,
    config: Config,
}

pub struct State<B: BackendF<f32>> {
    xs: TensorS<B>,
    fc1_xs: Storage,
    fc2_xs: Storage,
    rms_xs: Storage,
    attn_q: Storage,
    attn_k: Storage,
    attn_v: Storage,
    attn_q_t: Storage,
    attn_k_t: Storage,
    attn_v_t: Storage,
    attn_sm: Storage,
    attn_scores: Storage,
    attn_xs: Storage,
    attn_xs_t: Storage,
    logits: TensorS<B>,
    cos: TensorS<B>,
    sin: TensorS<B>,
    b_sz: usize,
    kv_caches: Vec<crate::kv_cache::KvCache<'static, f32, B>>,
}

impl<B: BackendF<f32>> State<B> {
    pub fn new(b_sz: usize, cfg: &Config) -> Result<Self> {
        let seq_len = 1;
        let max_seq_len = cfg.max_seq_len;
        let logits = Tensor::cst(0., (b_sz, seq_len, cfg.vocab_size))?;
        let xs = Tensor::cst(0., (b_sz, seq_len, cfg.dim))?;
        let fc1_xs = Storage::cst(0., b_sz * seq_len * cfg.hidden_dim)?;
        let fc2_xs = Storage::cst(0., b_sz * seq_len * cfg.hidden_dim)?;
        let rms_xs = Storage::cst(0., b_sz * seq_len * cfg.dim)?;
        let attn_xs = Storage::cst(0., b_sz * cfg.n_heads * seq_len * cfg.head_dim())?;
        let attn_xs_t = Storage::cst(0., b_sz * seq_len * cfg.n_heads * cfg.head_dim())?;
        let attn_scores = Storage::cst(0., b_sz * cfg.n_heads * seq_len * max_seq_len)?;
        let attn_sm = Storage::cst(0., b_sz * cfg.n_heads * seq_len * max_seq_len)?;
        let attn_q = Storage::cst(0., b_sz * seq_len * cfg.dim)?;
        let attn_k = Storage::cst(0., b_sz * seq_len * cfg.dim)?;
        let attn_v = Storage::cst(0., b_sz * seq_len * cfg.dim)?;
        let attn_q_t = Storage::cst(0., b_sz * seq_len * cfg.dim)?;
        let attn_k_t = Storage::cst(0., b_sz * seq_len * cfg.dim)?;
        let attn_v_t = Storage::cst(0., b_sz * seq_len * cfg.dim)?;
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
        mm.matmul_(&idx_theta, &theta, false)?;
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

    pub fn logits(&self) -> &TensorS<B> {
        &self.logits
    }
}

impl<B: BackendF<f32>> Model<B> {
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
        for (i, token) in tokens.iter().enumerate() {
            let h = self.config.dim;
            let token = *token as usize;
            state.xs.data_mut()[i * h..(i + 1) * h]
                .copy_from_slice(&self.embedding.data()[token * h..(token + 1) * h]);
        }
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
                    attn_q.rope_i(&state.cos, &state.sin, pos)?;
                    attn_q.reshape((b_sz * h, seq_len, d))?;

                    attn_k.reshape((b_sz, seq_len, h, d))?;
                    let mut attn_k = attn_k.transpose(&mut state.attn_k_t, 1, 2)?;
                    attn_k.rope_i(&state.cos, &state.sin, pos)?;

                    attn_v.reshape((b_sz, seq_len, h, d))?;
                    let attn_v = attn_v.transpose(&mut state.attn_v_t, 1, 2)?;
                    // kv-cache
                    let (k, v) = state.kv_caches[layer_idx].append(&attn_k, &attn_v)?;
                    let k = k.flatten(0, 1)?;
                    let v = v.flatten(0, 1)?;
                    // TODO: repeat-kv
                    let mut attn_scores =
                        tensor::matmul(&mut state.attn_scores, &attn_q, &k, true)?;
                    attn_scores.scale(1f32 / (layer.attn.head_dim as f32).sqrt());
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
