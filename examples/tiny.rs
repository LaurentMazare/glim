extern crate glim;

use rand::{distributions::Distribution, SeedableRng};
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    // https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json
    let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42424242);

    let config = glim::llama::Config::tiny_15m();
    let mut prs = glim::Tensor::cst(0., config.vocab_size)?;
    // Converted from https://huggingface.co/karpathy/tinyllamas/blob/main/stories15M.pt
    let model = glim::llama::Model::new(config, "stories15M.safetensors")?;
    let mut state = glim::llama::State::new(1, model.config())?;
    let mut tokens = vec![42];
    for _ in 0..200 {
        let prev_token = tokens.last().unwrap();
        model.fwd(&[*prev_token], &mut state)?;
        // println!("logits: {:?}", &state.logits().data()[..20]);
        prs.softmax(state.logits())?;
        let distr = rand::distributions::WeightedIndex::new(prs.data())?;
        let token = distr.sample(&mut rng) as u32;
        tokens.push(token);
        // #[cfg(feature = "candle")]
        // {
        //     candle::display::set_line_width(140);
        //     candle::display::set_edge_items(6);
        //     println!("{}", state.logits().to_candle()?);
        // }
    }
    let s = tokenizer.decode(&tokens, false).unwrap();
    println!(">>> {s}");
    Ok(())
}
