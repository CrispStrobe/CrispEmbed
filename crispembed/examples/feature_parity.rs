use crispembed::CrispEmbed;

fn assert_close(value: f32, target: f32, tol: f32, label: &str) {
    assert!(
        (value - target).abs() <= tol,
        "{label}: expected {target:.6}, got {value:.6}"
    );
}

fn l2_norm(vec: &[f32]) -> f32 {
    vec.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn main() {
    let mut args = std::env::args().skip(1);

    let dense_model = args.next().unwrap_or_else(|| {
        panic!("usage: cargo run -p crispembed --example feature_parity -- <dense.gguf> [retrieval.gguf] [reranker.gguf]")
    });
    let retrieval_model = args.next();
    let reranker_model = args.next();

    println!("[rust] dense model: {dense_model}");
    let mut model = CrispEmbed::new(&dense_model, 4).expect("failed to load dense model");

    let vec = model.encode("Hello world");
    assert!(!vec.is_empty(), "single encode returned no values");
    assert_close(l2_norm(&vec), 1.0, 1e-3, "single encode norm");

    let texts = ["query: crisp embeddings are fast", "dense retrieval with ggml", "batch inference should preserve order"];
    let batch = model.encode_batch(&texts);
    assert_eq!(batch.len(), texts.len(), "batch size mismatch");
    assert_eq!(batch[0].len(), vec.len(), "batch dim mismatch");
    let single_again = model.encode(texts[0]);
    assert!(
        batch[0]
            .iter()
            .zip(single_again.iter())
            .all(|(a, b)| (a - b).abs() <= 1e-5),
        "batch encode disagrees with single encode"
    );

    let trunc_dim = vec.len().min(128) as i32;
    model.set_dim(trunc_dim);
    let vec_trunc = model.encode("Hello world");
    assert_eq!(vec_trunc.len(), trunc_dim as usize, "set_dim did not truncate output");
    assert_close(l2_norm(&vec_trunc), 1.0, 1e-3, "truncated encode norm");
    model.set_dim(0);
    assert_eq!(model.encode("Hello world").len(), vec.len(), "set_dim(0) did not restore output");

    model.set_prefix("query: ");
    assert_eq!(model.prefix(), "query: ", "prefix getter mismatch");
    let prefixed = model.encode("hello");
    model.set_prefix("");
    let cleared = model.encode("hello");
    assert_eq!(model.prefix(), "", "prefix did not clear");
    assert_eq!(prefixed.len(), cleared.len(), "prefix changed output dim");
    assert!(
        prefixed
            .iter()
            .zip(cleared.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "prefix had no effect on embeddings"
    );

    let docs = [
        "Paris France capital city and Eiffel Tower.",
        "A bicycle uses two wheels and a chain.",
        "Berlin Germany capital city and Brandenburg Gate.",
    ];
    let ranked = model.rerank_biencoder("paris france capital", &docs, Some(2));
    assert_eq!(ranked.len(), 2, "rerank_biencoder top_n mismatch");
    assert_eq!(ranked[0].0, 0, "rerank_biencoder did not rank the relevant document first");
    assert!(ranked[0].1 >= ranked[1].1, "rerank_biencoder results are not sorted");

    println!("[rust] dense, batch, matryoshka, prefix, and bi-encoder rerank: PASS");

    if let Some(path) = retrieval_model {
        println!("[rust] retrieval model: {path}");
        let mut model = CrispEmbed::new(&path, 4).expect("failed to load retrieval model");
        assert!(model.has_sparse(), "retrieval model does not report sparse support");
        let sparse = model.encode_sparse("Paris is the capital of France.");
        assert!(!sparse.is_empty(), "encode_sparse returned no entries");
        assert!(
            sparse.iter().all(|(_, weight)| *weight > 0.0),
            "encode_sparse returned non-positive weights"
        );

        assert!(model.has_colbert(), "retrieval model does not report colbert support");
        let multi = model.encode_multivec("Paris is the capital of France.");
        assert!(!multi.is_empty(), "encode_multivec returned no token vectors");
        assert!(!multi[0].is_empty(), "encode_multivec returned zero-width token vectors");
        for token in &multi {
            assert_close(l2_norm(token), 1.0, 5e-3, "token vector norm");
        }
        println!("[rust] sparse and colbert retrieval: PASS");
    } else {
        println!("[rust] sparse and colbert retrieval: SKIP (no retrieval model)");
    }

    if let Some(path) = reranker_model {
        println!("[rust] reranker model: {path}");
        let mut model = CrispEmbed::new(&path, 4).expect("failed to load reranker model");
        assert!(model.is_reranker(), "reranker model does not report reranker support");
        let positive = model.rerank("capital of france", "Paris is the capital of France.");
        let negative = model.rerank("capital of france", "Bicycles have handlebars and pedals.");
        assert!(positive.is_finite() && negative.is_finite(), "rerank returned non-finite value");
        assert!(positive > negative, "reranker failed to score the relevant document higher");
        println!("[rust] cross-encoder rerank: PASS");
    } else {
        println!("[rust] cross-encoder rerank: SKIP (no reranker model)");
    }

    println!("[rust] feature parity script completed");
}
