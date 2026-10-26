// Hosted consumer of unchanged safe Rust sources and an actual prebuilt library.
use crispembed::CrispEmbed;
use std::{env, fs, io::{self, Write}};

fn valid(vector: &[f32]) {
    assert_eq!(vector.len(), 384);
    assert!(vector.iter().all(|v| v.is_finite()));
    let norm: f64 = vector.iter().map(|v| (*v as f64).powi(2)).sum();
    assert!((norm - 1.0).abs() < 0.002);
}
fn vectors(rows: &[Vec<f32>]) {
    print!("[");
    for (i, row) in rows.iter().enumerate() {
        if i > 0 { print!(","); }
        print!("[");
        for (j, value) in row.iter().enumerate() {
            if j > 0 { print!(","); }
            print!("{}", value);
        }
        print!("]");
    }
    print!("]");
}
fn main() {
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 4);
    assert_eq!(env::var("CRISPEMBED_OFFLINE").unwrap(), "1");
    let model_path = fs::canonicalize(&args[1]).unwrap();
    let library = fs::canonicalize(&args[3]).unwrap();
    assert!(model_path.is_file() && library.is_file());
    let texts: Vec<String> = fs::read_to_string(&args[2]).unwrap().lines().map(str::to_string).collect();
    assert_eq!(texts.len(), 3);
    assert_eq!(CrispEmbed::resolve_model(model_path.to_str().unwrap(), Some(false)).unwrap(), model_path.to_str().unwrap());
    assert!(CrispEmbed::resolve_model(model_path.with_file_name("successor-deliberately-absent.gguf").to_str().unwrap(), Some(false)).is_err());
    let mut model = CrispEmbed::new(model_path.to_str().unwrap(), 2).unwrap();
    assert_eq!(model.dim(), 384);
    assert_eq!(model.prefix(), "");
    let scalar: Vec<Vec<f32>> = texts.iter().map(|text| model.encode(text)).collect();
    for vector in &scalar { valid(vector); }
    let repeat = model.encode(&texts[0]);
    valid(&repeat);
    assert!(repeat.iter().zip(&scalar[0]).all(|(a, b)| a.to_bits() == b.to_bits()));
    let batch = model.encode_batch(&[texts[2].as_str(), texts[0].as_str(), texts[1].as_str(), texts[0].as_str()]);
    assert_eq!(batch.len(), 4);
    for vector in &batch { valid(vector); }
    let mapped = fs::read_to_string("/proc/self/maps").unwrap();
    assert!(mapped.lines().any(|line| line.ends_with(library.to_str().unwrap())));
    // Free the actual wrapper context before bypassing process-global ggml teardown.
    drop(model);
    print!("{{\"scalar\":"); vectors(&scalar);
    print!(",\"repeat\":"); vectors(&[repeat]);
    print!(",\"batch\":"); vectors(&batch);
    println!(",\"contracts\":{{\"dimensions\":true,\"repeat_exact\":true,\"offline_cached_path\":true,\"missing_offline_rejected\":true,\"actual_prebuilt_mapped\":true}}}}");
    io::stdout().flush().unwrap();
    extern "C" { fn _Exit(status: i32) -> !; }
    unsafe { _Exit(0) }
}
