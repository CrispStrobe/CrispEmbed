// MinerU2.5-Pro Rust wrapper consumer (crispembed::MinerU) against a real GGUF.
// Interface consistency, not reference parity: the expected text and token IDs are
// the Python/C anchor from the same native engine. Built (with serde_json) and run by
// tools/ci-heavy/ocr_wrappers.sh.
//
//   mineru25-public-probe model.gguf crop.rgb W H anchor.json max_tokens page.rgb PW PH
use crispembed::MinerU;
use serde_json::Value;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 10, "model crop.rgb width height anchor.json max-tokens page.rgb width height");
    let crop = std::fs::read(&args[2]).expect("crop RGB");
    let (width, height): (i32, i32) = (args[3].parse().unwrap(), args[4].parse().unwrap());
    let anchor: Value = serde_json::from_str(&std::fs::read_to_string(&args[5]).unwrap()).expect("anchor JSON");
    let max_tokens: i32 = args[6].parse().unwrap();
    let page = std::fs::read(&args[7]).expect("page RGB");
    let (pw, ph): (i32, i32) = (args[8].parse().unwrap(), args[9].parse().unwrap());

    let mut model = MinerU::new(&args[1], 2).expect("MinerU init");
    let region: Value = serde_json::from_str(
        &model.predict_rgb(&crop, width, height, "\nText Recognition:", max_tokens).expect("predict_rgb"),
    )
    .expect("region JSON");
    let predict_ok = region["text"] == anchor["text"] && region["token_ids"] == anchor["token_ids"];
    println!("{}", serde_json::json!({"surface": "rust-predict", "passed": predict_ok, "text": region["text"]}));

    let rejected = model.predict_rgb(&crop, width, height + 1, "\nText Recognition:", max_tokens).is_err();
    println!("{}", serde_json::json!({"surface": "rust-validation", "passed": rejected}));

    let page_json: Value =
        serde_json::from_str(&model.extract_native_basic(&page, pw, ph, false, 0).expect("extract_native_basic"))
            .expect("page JSON");
    let blocks = page_json["blocks"].as_array().map_or(0, |b| b.len());
    let markdown = page_json["markdown"].as_str().unwrap_or("").len();
    let page_ok = blocks > 0 && markdown > 0 && page_json["capped"] == Value::Bool(false);
    println!("{}", serde_json::json!({"surface": "rust-extract", "passed": page_ok, "blocks": blocks, "markdown_bytes": markdown}));
    std::process::exit(if predict_ok && rejected && page_ok { 0 } else { 1 });
}
