// Temporary hosted consumer of the real safe Rust wrapper and shared library.
use crispembed::CrispEmbed;
use serde_json::{json, Value};
use std::{env, fs};
fn main() {
    let args: Vec<String> = env::args().collect();
    if args[2] == "--reject" {
        assert!(CrispEmbed::new(&args[1], 2).is_err());
        println!("{{\"rejected\":true}}");
        return;
    }
    let request: Value = serde_json::from_str(&fs::read_to_string(&args[2]).unwrap()).unwrap();
    let mut model = CrispEmbed::new(&args[1], 2).unwrap();
    let mut other = CrispEmbed::new(&args[1], 2).unwrap();
    let mut rows = Vec::new();
    for row in request.as_array().unwrap() {
        let text = row["text"].as_str().unwrap();
        let prefix = row["prefix"].as_str().unwrap();
        model.set_prefix(prefix);
        assert_eq!(model.prefix(), prefix);
        let dense = model.encode(text);
        let (ids, hidden) = model.encode_tokens_raw(text).unwrap();
        let normalized = model.encode_tokens(text);
        assert_eq!(hidden.len(), ids.len());
        assert_eq!(normalized.len(), ids.len());
        assert_eq!(model.encode(text), dense);
        let batch = model.encode_batch(&[text, text]);
        assert_eq!(batch, vec![dense.clone(), dense.clone()]);
        other.set_prefix("");
        assert_eq!(other.encode(&(prefix.to_owned() + text)), dense);
        assert!(model.encode("<|mask|>").is_empty());
        assert!(model.encode_tokens_raw("<|mask|>").is_err());
        assert!(model.encode_batch(&[text, "<|mask|>"]).is_empty());
        assert!(model.encode(&" x".repeat(513)).is_empty());
        assert!(model.encode(&"x".repeat(65537)).is_empty());
        assert_eq!(model.encode(text), dense);
        rows.push(json!({"id":row["id"],"ids":ids,"raw_cls":hidden[0],
                         "dense":dense,"token_normalized_cls":normalized[0].1}));
    }
    model.set_prefix("");
    let texts: Vec<&str> = request
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["prefix"] == "")
        .map(|r| r["text"].as_str().unwrap())
        .collect();
    let batch = model.encode_batch(&texts);
    assert_eq!(batch.len(), texts.len());
    for (actual, text) in batch.iter().zip(texts.iter()) {
        assert_eq!(*actual, model.encode(text));
    }
    println!(
        "{}",
        json!({"rows":rows,"contracts":{"repeat":true,"batch":true,
        "context_prefix":true,"rejection_recovery":true}})
    );
}
