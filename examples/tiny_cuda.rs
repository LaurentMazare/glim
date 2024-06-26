extern crate glim;
use anyhow::Context;
use glim::Backend;
use rand::{distributions::Distribution, SeedableRng};
use tokenizers::Tokenizer;

type B = glim::cuda_backend::Storage<f32>;

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "candle")]
    {
        candle::display::set_line_width(140);
        candle::display::set_edge_items(6);
    }

    // https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json
    let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42424242);

    let device = glim::cuda_backend::Device::new(0)?;

    let config = glim::llama::Config::tiny_110m();
    let vocab_size = config.vocab_size;
    // Converted from https://huggingface.co/karpathy/tinyllamas/blob/main/stories110M.pt
    let model: glim::llama::Model<B> =
        glim::llama::Model::new(config, &device, "stories110M.safetensors")?;
    let mut state: glim::llama::State<B> = glim::llama::State::new(1, model.config(), &device)?;
    let start_time = std::time::Instant::now();
    let bos_token = tokenizer.token_to_id("<s>").context("no bos token")?;
    let mut tokens = vec![bos_token];
    let mut prs_storage = unsafe { B::alloc_uninit(vocab_size, &device)? };
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
