extern crate glim;

fn main() -> anyhow::Result<()> {
    let config = glim::llama::Config::tiny_15m();
    // Converted from https://huggingface.co/karpathy/tinyllamas/blob/main/stories15M.pt
    let model = glim::llama::Model::new(config, "stories15M.safetensors")?;
    let mut state = glim::llama::State::new(1, model.config())?;
    for token in [42, 1337, 1, 2, 3, 4] {
        model.fwd(&[token], &mut state)?;
        println!("logits: {:?}", &state.logits().data()[..20]);
        #[cfg(feature = "candle")]
        {
            candle::display::set_line_width(140);
            candle::display::set_edge_items(6);
            println!("{}", state.logits().to_candle()?);
        }
    }
    Ok(())
}
