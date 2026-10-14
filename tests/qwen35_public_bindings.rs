use crispembed::{CrispOcrPipeline, OcrCleanupSpec, OcrModel, OcrStageSpec};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pixels = std::fs::read(&args[2]).unwrap();
    let w: i32 = args[3].parse().unwrap();
    let h: i32 = args[4].parse().unwrap();
    {
        let mut model = OcrModel::new(&args[1], 4).unwrap();
        model.set_max_tokens(12);
        assert_eq!(model.get_max_tokens(), 12);
        assert_eq!(
            model.recognize(&pixels, w, h).unwrap(),
            "Hello OCR  \n12345"
        );
        assert_eq!(model.confidences().len(), 9);
        assert!((model.mean_confidence() - 0.995560692).abs() < 1e-6);
        assert!(model.recognize(&pixels[..3], w, h).is_none());
        assert!(model.recognize(&pixels, -1, h).is_none());
        assert!(model.recognize_gray(&[], w, h).is_none());
        assert!(model.set_prompt(Some("x")).unwrap());
        assert!(model.set_prompt(None).unwrap());
        model.set_max_tokens(2);
        let gray: Vec<u8> = pixels.chunks_exact(3).map(|p| p[0]).collect();
        assert_eq!(model.recognize(&gray, w, h).unwrap(), "Hello OCR");
        assert_eq!(model.confidences().len(), 2);
        println!(
            "PASS Rust unified RGB/grayscale, buffer guards, prompt/reset, cap and confidence"
        );
    }
    let stage = OcrStageSpec {
        source_type: 1,
        engine: 20,
        model_a: args[1].clone(),
        model_b: String::new(),
        cleanup: OcrCleanupSpec {
            enabled: false,
            ..Default::default()
        },
        det_prob_threshold: 0.0,
        det_box_threshold: 0.0,
        det_target_short: 0,
        vlm_max_tokens: 12,
        vlm_prompt: String::new(),
        min_chars: 1,
        min_confidence: 0.0,
    };
    let mut pipe =
        CrispOcrPipeline::from_stages(false, None, None, None, None, None, None, &[stage], 4)
            .unwrap();
    assert_eq!(pipe.run(&args[5]).unwrap().full_text, "Hello OCR  \n12345");
    println!("PASS Rust orchestrator ID20");
}
