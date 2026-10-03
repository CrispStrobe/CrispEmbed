use crispembed::CrispEmbed;

/// Run with CRISPEMBED_LFM2_ENCODER_MODEL pointing to the official F16 GGUF.
#[test]
#[ignore = "requires the official LFM2 encoder GGUF"]
fn native_encoder_features_and_mask_predictions() {
    let path = std::env::var("CRISPEMBED_LFM2_ENCODER_MODEL").expect("set model path");
    std::env::set_var("CRISPEMBED_LFM2_ENCODER", "1");
    let mut model = CrispEmbed::new(&path, 2).unwrap();
    assert!(model.has_masked_lm());
    let text = "The capital of France is <|mask|>.";
    let (ids, raw) = model.encode_tokens_raw(text).unwrap();
    assert_eq!(ids, vec![1, 1098, 5706, 803, 4481, 856, 730, 16, 523]);
    assert_eq!(raw.len(), 9);
    assert!(raw
        .iter()
        .all(|row| row.len() == 1024 && row.iter().all(|x| x.is_finite())));
    let normalized = model.encode_tokens(text);
    assert_eq!(normalized.len(), raw.len());
    for (row, (_, normed)) in raw.iter().zip(normalized) {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(row
            .iter()
            .zip(normed)
            .all(|(&x, y)| (x / norm - y).abs() < 2e-6));
    }
    let (positions, logits) = model.masked_logits(text).unwrap();
    assert_eq!(positions, vec![7]);
    assert_eq!(logits[0].len(), 65536);
    let filled = model
        .fill_mask("The capital of France is [MASK].", 5)
        .unwrap();
    assert_eq!(filled[0].position, 7);
    assert_eq!(filled[0].predictions[0].token_id, 5242);
    assert_eq!(filled[0].predictions[0].token, " Paris");
    assert_eq!(filled[0].predictions[0].logit, logits[0][5242]);
    assert!(filled[0].predictions[0].score > 0.0 && filled[0].predictions[0].score < 1.0);
    assert_eq!(model.masked_logits(text).unwrap().1, logits);
    assert!(model.masked_logits("no mask").is_err());
    assert!(model.fill_mask(text, 0).is_err());
    assert!(model.token_bytes(-1).is_err());
}
