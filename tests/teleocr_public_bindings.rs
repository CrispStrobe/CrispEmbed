use crispembed::{CrispOcrPipeline, OcrCleanupSpec, OcrModel, OcrStageSpec, OCR_ENGINE_TELEOCR};

fn json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch < '\u{20}' => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        8,
        "model raw.rgb width height image expected-utf8-file max-tokens"
    );
    let pixels = std::fs::read(&args[2]).expect("raw RGB input");
    let width: i32 = args[3].parse().unwrap();
    let height: i32 = args[4].parse().unwrap();
    let expected = std::fs::read_to_string(&args[6]).expect("exact UTF-8 expected output");
    let cap: i32 = args[7].parse().unwrap();
    assert!(cap > 0 && !expected.is_empty());
    assert_eq!(OCR_ENGINE_TELEOCR, 24);
    assert_eq!(pixels.len(), width as usize * height as usize * 3);

    // Do not retain two full checkpoints at once: each surface owns and frees
    // its ordinary safe-wrapper context before the next surface starts.
    {
        let mut model = OcrModel::new(&args[1], 2).expect("TeleOCR native model");
        assert_eq!(model.get_max_tokens(), 4096);
        let defaults = model
            .generation_params()
            .expect("TeleOCR generation parameters");
        assert!((defaults.0 - 1.05).abs() < 1e-6);
        assert_eq!((defaults.1, defaults.2, defaults.3), (0.0, 0.0, 0));
        assert!(model.set_generation_params(1.0, 1.0, 0.05, 100));
        assert_eq!(model.generation_params(), Some((1.0, 1.0, 0.05, 100)));
        assert!(!model.set_generation_params(f32::NAN, 0.0, 0.0, 0));
        assert_eq!(model.generation_params(), Some((1.0, 1.0, 0.05, 100)));
        assert!(model.set_generation_params(defaults.0, defaults.1, defaults.2, defaults.3));
        model.set_max_tokens(cap);
        assert_eq!(model.get_max_tokens(), cap);
        assert!(model.set_prompt(Some("Read the image.")).unwrap());
        assert!(model.set_prompt(None).unwrap());
        let text = model
            .recognize(&pixels, width, height)
            .expect("RGB recognition");
        assert_eq!(text, expected);
        let confidences = model.confidences();
        assert!(!confidences.is_empty() && confidences.len() <= cap as usize);
        assert!(confidences
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
        assert!(model.mean_confidence().is_finite());
        println!(
            "{{\"surface\":\"rust-model\",\"passed\":true,\"text\":{},\"token_count\":{},\"confidence\":{},\"max_tokens\":{}}}",
            json_string(&text), confidences.len(), model.mean_confidence(), cap
        );
    }

    let stage = OcrStageSpec {
        source_type: 1,
        engine: OCR_ENGINE_TELEOCR,
        model_a: args[1].clone(),
        model_b: String::new(),
        cleanup: OcrCleanupSpec {
            enabled: false,
            ..Default::default()
        },
        det_prob_threshold: 0.0,
        det_box_threshold: 0.0,
        det_target_short: 0,
        vlm_max_tokens: cap,
        vlm_prompt: String::new(),
        min_chars: 1,
        min_confidence: 0.0,
    };
    let mut pipeline =
        CrispOcrPipeline::from_stages(false, None, None, None, None, None, None, &[stage], 2)
            .expect("TeleOCR stage 24");
    let result = pipeline.run(&args[5]).expect("stage recognition");
    assert_eq!(result.full_text, expected);
    assert_eq!(result.regions.len(), 1);
    assert_eq!(result.markdown, format!("{expected}\n\n"));
    println!(
        "{{\"surface\":\"rust-stage\",\"passed\":true,\"engine\":24,\"text\":{},\"markdown\":{},\"regions\":{}}}",
        json_string(&result.full_text), json_string(&result.markdown), result.regions.len()
    );
}
