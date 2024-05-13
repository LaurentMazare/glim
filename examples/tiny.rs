extern crate glim;

fn main() -> anyhow::Result<()> {
    let config = glim::llama::Config::tiny_15m();
    // Converted from https://huggingface.co/karpathy/tinyllamas/blob/main/stories15M.pt
    let model = glim::llama::Model::new(config, "stories15M.safetensors")?;
    let mut state = glim::llama::State::new(1, 1, model.config())?;
    model.fwd(&[42], &mut state)?;
    println!("logits: {:?}", &state.logits().data()[..20]);
    #[cfg(feature = "candle")]
    println!("{}", state.logits().to_candle()?);
    Ok(())
}
