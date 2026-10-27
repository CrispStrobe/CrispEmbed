//! Hosted audit consumer: raw TSV requests, complete responses; no reference IDs.
use crispembed::SystemOne;
use std::{env, fs};
fn main() -> Result<(), String> {
    if env::var("GITHUB_ACTIONS").as_deref() != Ok("true") {
        return Err("Hosted model audit only".into());
    }
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        return Err("model.gguf requests.tsv".into());
    }
    let mut model = SystemOne::new_with_download(&args[1], 2, Some(false))?;
    if model.profile() != "official-pytorch-text-batch1-512-no-extra-temperature" {
        return Err("Profile differs".into());
    }
    for invalid in ["{", "{}", "x\0y"] {
        if model.run_json(invalid).is_ok() {
            return Err("Invalid request accepted".into());
        }
    }
    let text = fs::read_to_string(&args[2]).map_err(|e| e.to_string())?;
    let rows: Vec<_> = text.lines().collect();
    if rows.len() != 4 {
        return Err("Four frozen raw requests required".into());
    }
    let mut first = None;
    for line in rows {
        let (id, request) = line.split_once('\t').ok_or("Raw TSV record invalid")?;
        let result = model.run_json(request)?;
        if first.is_none() {
            first = Some((request.to_string(), result.clone()));
        }
        println!("{id}\t{result}");
    }
    if model.run_json("{").is_ok() {
        return Err("Post-success failure accepted".into());
    }
    let (request, result) = first.ok_or("Missing first record")?;
    if model.run_json(&request)? != result {
        return Err("Recovery/repeat differs".into());
    }
    Ok(())
}
