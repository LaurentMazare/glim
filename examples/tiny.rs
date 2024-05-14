extern crate glim;

fn main() -> anyhow::Result<()> {
    let config = glim::llama::Config::tiny_15m();
    // Converted from https://huggingface.co/karpathy/tinyllamas/blob/main/stories15M.pt
    let model = glim::llama::Model::new(config, "stories15M.safetensors")?;
    let mut state = glim::llama::State::new(1, 2, model.config())?;
    model.fwd(&[42, 1337], &mut state)?;
    println!("logits: {:?}", &state.logits().data()[..20]);
    #[cfg(feature = "candle")]
    {
        candle::display::set_line_width(140);
        candle::display::set_edge_items(6);
        println!("{}", state.logits().to_candle()?);
    }
    Ok(())
}
