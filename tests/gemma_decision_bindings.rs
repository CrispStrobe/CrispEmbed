// Built as a temporary hosted consumer, against the already-built shared library.
use crispembed::CrispEmbed;
use std::{env, fs};

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut model = CrispEmbed::new(&args[1], 2).unwrap();
    assert!(model.has_decision());
    let bytes = fs::read(&args[3]).unwrap();
    let vectors: Vec<Vec<f32>> = bytes
        .chunks_exact(768 * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect()
        })
        .collect();
    let (logits, probabilities) = model
        .decision_scores(
            &vectors[0],
            &vectors[1],
            &vectors[2..],
            args[4].parse().unwrap(),
        )
        .unwrap();
    println!(
        "{{\"logits\":{:?},\"probabilities\":{:?}}}",
        logits, probabilities
    );
    let request = fs::read_to_string(&args[2]).unwrap();
    let answer = model.decide_json(&request).unwrap();
    assert!(model
        .decide_json("{\"state\":\"x\",\"questions\":{}}")
        .is_err());
    assert_eq!(model.decide_json(&request).unwrap(), answer);
    println!("{}", answer);
}
