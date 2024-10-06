extern crate glim;
use anyhow::{Context, Result};
use glim::BackendF;
use half::f16;

use glim::llama::{Config, Model, State};
use rand::{distributions::Distribution, SeedableRng};
use tokenizers::Tokenizer;

#[derive(Debug, Copy, Clone)]
enum Which {
    Tiny15m,
    Tiny110m,
    Llama2_7b,
}

impl Which {
    fn config(&self) -> Config {
        match self {
            Self::Tiny15m => Config::tiny_15m(),
            Self::Tiny110m => Config::tiny_110m(),
            Self::Llama2_7b => Config::llama2_7b(),
        }
    }

    fn weight_file(&self) -> &'static str {
        match self {
            // Converted from https://huggingface.co/karpathy/tinyllamas/blob/main/stories15M.pt
            Self::Tiny15m => "stories15M.safetensors",
            Self::Tiny110m => "stories110M.safetensors",
            Self::Llama2_7b => "llama2-7b.safetensors",
        }
    }

    fn from_cli_args() -> Result<Self> {
        let args: Vec<String> = std::env::args().collect();
        let args: Vec<&str> = args.iter().map(|v| v.as_str()).collect();
        let w = match args.as_slice() {
            [_] | [_, "tiny15m"] => Self::Tiny15m,
            [_, "tiny110m"] => Self::Tiny110m,
            [_, "7b"] => Self::Llama2_7b,
            args => anyhow::bail!("unexpected cli arguments {args:?}"),
        };
        Ok(w)
    }
}

fn run<B, T>(which: Which, dev: &B::Device) -> anyhow::Result<()>
where
    B: BackendF<T>,
    T: glim::WithDTypeF
        + rand::distributions::uniform::SampleUniform
        + Default
        + Copy
        + for<'a> std::ops::AddAssign<&'a T>,
{
    #[cfg(feature = "candle")]
    {
        candle::display::set_line_width(140);
        candle::display::set_edge_items(6);
    }

    // https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json
    let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42424242);

    let config = which.config();
    let vocab_size = config.vocab_size;
    let model = Model::new(config, dev, which.weight_file())?;
    let mut state = State::new(1, model.config(), dev)?;
    let start_time = std::time::Instant::now();
    let bos_token = tokenizer.token_to_id("<s>").context("no bos token")?;
    let mut tokens = vec![bos_token];
    let mut prs_storage = unsafe { B::alloc_uninit(vocab_size, dev)? };
    for _ in 0..200 {
        let prev_token = tokens.last().unwrap();
        model.fwd(&[*prev_token], &mut state)?;
        let prs = state.logits().softmax(&mut prs_storage)?;
        let prs = prs.data()?;
        let distr = rand::distributions::WeightedIndex::new(prs.as_ref())?;
        let token = distr.sample(&mut rng) as u32;
        tokens.push(token);
    }
    let dt = start_time.elapsed();
    let s = tokenizer.decode(&tokens, false).unwrap();
    println!("{s}");
    println!(
        "generated {} tokens, {:.2} tokens/s",
        tokens.len() - 1,
        (tokens.len() - 1) as f64 / dt.as_secs_f64()
    );
    #[cfg(feature = "candle")]
    println!("{}", state.logits().to_candle()?);
    Ok(())
}

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let device = glim::cuda_backend::Device::new(0)?;
    let which = Which::from_cli_args()?;
    match which {
        Which::Llama2_7b => {
            type B = glim::cuda_backend::Storage<f16>;
            run::<B, f16>(which, &device)?;
        }
        _ => {
            type B = glim::cuda_backend::Storage<f32>;
            run::<B, f32>(which, &device)?;
        }
    };
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    let which = Which::from_cli_args()?;
    match which {
        Which::Llama2_7b => {
            type B = glim::cpu_backend::Storage<f16>;
            run::<B, f16>(which, &())?;
        }
        _ => {
            type B = glim::cpu_backend::Storage<f32>;
            run::<B, f32>(which, &())?;
        }
    }
    Ok(())
}
