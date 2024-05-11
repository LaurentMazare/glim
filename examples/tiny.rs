extern crate glim;

fn main() -> anyhow::Result<()> {
    let config = glim::Config::tiny_15m();
    // Converted from https://huggingface.co/karpathy/tinyllamas/blob/main/stories15M.pt
    let model = glim::Model::new(config, "stories15M.safetensors")?;
    let mut state = glim::State::new(1, 1, model.config())?;
    model.fwd(&[42], &mut state)?;
    println!("logits: {:?}", &state.logits().data()[..20]);
    Ok(())
}
