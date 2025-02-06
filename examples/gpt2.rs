use std::{
    io::{self, Write},
    path::Path
};

use ort::{
    execution_providers::CPUExecutionProvider,
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor
};
use rand::Rng;
use tokenizers::Tokenizer;

const PROMPT: &str = "The corsac fox (Vulpes corsac), also known simply as a corsac, is a medium-sized fox found in";
/// max tokens to generate
const GEN_TOKENS: i32 = 90;
/// sample from the k most likely next tokens at each step. lower k focuses on higher probability tokens
const TOP_K: usize = 5;

/// gpt-2 text generation
///
/// this rust program demonstrates text generation using the gpt-2 language model with `ort`
fn main() -> ort::Result<()> {
    // initialize tracing to receive debug messages from `ort`
    tracing_subscriber::fmt::init();

    // create the onnx runtime environment, enabling CPU execution providers for all sessions created in this process.
    ort::init()
        .with_name("GPT-2")
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .commit()?;

    let mut stdout: io::Stdout = io::stdout();
    let mut rng = rand::thread_rng();

    // load our model
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        // .commit_from_url("https://parcel.pyke.io/v2/cdn/assetdelivery/ortrsv2/ex_models/gpt2.onnx")?; 
	.commit_from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("gpt2.onnx"))?;
    
    // load the tokenizer and encode the prompt into a sequence of tokens.
    let tokenizer = Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("tokenizer.json")).unwrap();
    let tokens = tokenizer.encode(PROMPT, false).unwrap();
    let mut tokens = tokens.get_ids().iter().map(|i| *i as i64).collect::<Vec<_>>();

    print!("{PROMPT}");
    stdout.flush().unwrap();

    for _ in 0..GEN_TOKENS {
        // raw tensor construction takes a tuple of (dimensions, data).
        // the model expects our input to have shape [B, _, S]
        let input = Tensor::from_array((vec![1, 1, tokens.len() as i64], tokens.as_slice()))?;
        let outputs = session.run(inputs![input]?)?;
        let (dim, mut probabilities) = outputs["output1"].try_extract_raw_tensor()?;

        // the output tensor will have shape [B, _, S, V]
        // we want only the probabilities for the last token in this sequence, which will be the next most likely token according to the model
        let (seq_len, vocab_size) = (dim[2] as usize, dim[3] as usize);
        probabilities = &probabilities[(seq_len - 1) * vocab_size..];

        // sort each token by probability
        let mut probabilities: Vec<(usize, f32)> = probabilities.iter().copied().enumerate().collect();
        probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

        // sample using top-k sampling
        let token = probabilities[rng.gen_range(0..=TOP_K)].0 as i64;

        // add our generated token to the input sequence
        tokens.push(token);

        let token_str = tokenizer.decode(&[token as u32], true).unwrap();
        print!("{}", token_str);
        stdout.flush().unwrap();
    }

    println!();

    Ok(())
}
