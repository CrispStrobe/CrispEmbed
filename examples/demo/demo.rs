//! CrispEmbed Rust demo — text embedding, reranking, and RAG search.
//!
//! Build: cd examples/demo && cargo build --release
//! Run:   cargo run --release -- /path/to/model.gguf

use crispembed::CrispEmbed;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn main() {
    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        format!("{home}/.cache/crispembed/all-MiniLM-L6-v2.gguf")
    });

    println!("=== CrispEmbed Rust Demo ===\n");
    let mut model = CrispEmbed::new(&model_path, 4).expect("Failed to load model");
    println!("Model loaded: dim={}", model.dim());

    // 1. Dense encode
    let vec = model.encode("Hello world");
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("\n1. Dense encode: {} dims, norm={norm:.4}", vec.len());

    // 2. Batch encode
    let texts = &["Machine learning is amazing", "The weather is nice", "Deep learning uses neural networks"];
    let vecs = model.encode_batch(texts);
    println!("2. Batch encode: {} texts -> {} vectors", texts.len(), vecs.len());

    // 3. Similarity search
    let query = "What is deep learning?";
    let q_vec = model.encode(query);
    println!("\n3. Similarity search for: '{query}'");
    let mut scored: Vec<_> = texts.iter().zip(&vecs).map(|(t, v)| (cosine(&q_vec, v), *t)).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (sim, text) in &scored {
        println!("   {sim:.4}  {text}");
    }

    // 4. Bi-encoder reranking
    let ranked = model.rerank_biencoder(query, texts, Some(2));
    println!("\n4. Bi-encoder reranking (top 2):");
    for (idx, score) in &ranked {
        println!("   [{idx}] {score:.4}  {}", texts[*idx]);
    }

    // 5. Matryoshka
    model.set_dim(128);
    let vec128 = model.encode("Hello world");
    println!("\n5. Matryoshka: dim=128 -> {} dims", vec128.len());
    model.set_dim(0);

    // 6. Prefix
    model.set_prefix("query: ");
    let vec_pf = model.encode("Hello world");
    println!("6. With prefix '{}': {} dims", model.prefix(), vec_pf.len());
    model.set_prefix("");

    println!("\nAll Rust demo tests passed!");
}
