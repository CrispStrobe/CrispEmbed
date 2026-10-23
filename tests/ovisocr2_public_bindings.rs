use crispembed::{CrispOcrPipeline, OcrStageSpec, OcrCleanupSpec, OcrModel};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 7, "model raw.rgb width height image expected-text");
    let pixels = std::fs::read(&args[2]).unwrap();
    let w: i32 = args[3].parse().unwrap();
    let h: i32 = args[4].parse().unwrap();
    let expected = &args[6];
    let mut model = OcrModel::new(&args[1], 2).expect("Ovis model");
    model.set_max_tokens(12);
    assert_eq!(model.get_max_tokens(), 12);
    assert!(!model.set_prompt(Some("different instruction")).unwrap());
    assert!(model.set_prompt(None).unwrap());
    let text = model.recognize(&pixels, w, h).expect("RGB recognition");
    assert_eq!(&text, expected);
    let confidence = model.confidences();
    assert!(!confidence.is_empty() && confidence.len() <= 12);
    assert!(confidence.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
    assert!(model.mean_confidence().is_finite());
    println!("PASS Rust unified RGB cap=12 prompt-rejection text_bytes={} confidence_count={} mean={:.9}",
             text.len(), confidence.len(), model.mean_confidence());
    let stage = OcrStageSpec {
        source_type: 1, engine: 21, model_a: args[1].clone(), model_b: String::new(),
        cleanup: OcrCleanupSpec { enabled: false, ..Default::default() },
        det_prob_threshold: 0.0, det_box_threshold: 0.0, det_target_short: 0,
        vlm_max_tokens: 12, vlm_prompt: String::new(), min_chars: 1, min_confidence: 0.0,
    };
    let mut pipe = CrispOcrPipeline::from_stages(false, None, None, None, None, None, None, &[stage], 2)
        .expect("Ovis stage 21");
    assert_eq!(&pipe.run(&args[5]).expect("stage recognition").full_text, expected);
    println!("PASS Rust orchestrator engine=21");
}
