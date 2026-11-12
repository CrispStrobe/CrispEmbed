use crispembed::{CrispOcrPipeline, OcrCleanupSpec, OcrStageSpec};
fn spec(model: &str) -> OcrStageSpec {
    OcrStageSpec { source_type: 0, engine: 14, model_a: model.into(), model_b: String::new(),
        cleanup: OcrCleanupSpec { enabled: false, ..Default::default() }, det_prob_threshold: 0.3,
        det_box_threshold: 0.5, det_target_short: 736, vlm_max_tokens: 0, vlm_prompt: String::new(),
        min_chars: 0, min_confidence: 0.0 }
}
fn pipeline(stages: &[OcrStageSpec]) -> CrispOcrPipeline {
    CrispOcrPipeline::from_stages(false, None, None, None, None, None, None, stages, 2).unwrap()
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = &args[1]; let hello = &args[2]; let real = &args[3];
    let mut missing = pipeline(&[spec("missing-owned-ocr-fixture.gguf")]);
    if args.get(4).is_some_and(|s| s == "older-runtime") {
        assert!(missing.run(hello).unwrap().full_text.is_empty());
        println!("older-immutable-runtime-link-positive; prior empty behavior retained"); return;
    }
    assert!(missing.run(hello).is_err());
    let mut legacy = spec(""); legacy.engine = 0;
    let mut mixed = pipeline(&[spec("missing-owned-ocr-fixture.gguf"), legacy.clone()]);
    assert!(mixed.run(hello).is_err());
    let mut unknown_only = pipeline(&[legacy]);
    assert!(unknown_only.run(hello).unwrap().full_text.is_empty());
    let mut valid = pipeline(&[spec(model)]);
    let owned = valid.run(hello).unwrap().full_text;
    assert_eq!(owned, "Hello OCR\n\n12345");
    assert!(valid.run(real).is_err());
    assert_eq!(owned, "Hello OCR\n\n12345");
    assert_eq!(valid.run(hello).unwrap().full_text, owned);
    let mut recovered = pipeline(&[spec("missing-owned-ocr-fixture.gguf"), spec(model)]);
    assert_eq!(recovered.run(hello).unwrap().full_text, owned);
    println!("rust-semantic-positive: failure, mixed unknown rejection, legacy blank compatibility, decoded text, owned result, next success, failed-first fallback");
}
