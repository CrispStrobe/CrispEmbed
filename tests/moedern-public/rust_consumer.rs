//! Actual Rust consumer through crispembed::MoEdern; hex TSV protocol.
use crispembed::moedern_text::{MoEdern, Role};
use std::{env, fs};
fn text(h: &str) -> String {
    if h == "-" { return String::new(); }
    let b: Vec<u8> = (0..h.len()).step_by(2).map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap()).collect();
    String::from_utf8(b).unwrap()
}
fn list<T: ToString>(v: &[T]) -> String { v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",") }
fn main() -> Result<(), String> {
    let a: Vec<String> = env::args().collect();
    let mut m = MoEdern::new(&a[1], 2, Some(false))?;
    for line in fs::read_to_string(&a[2]).map_err(|e| e.to_string())?.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let role = if f[1] == "query" { Role::Query } else { Role::Document };
        let r = m.encode(&text(f.get(2).copied().unwrap_or("-")), role)?;
        let fl = |v: &[f32]| v.iter().map(|x| format!("{:e}", x)).collect::<Vec<_>>().join(",");
        println!("{{\"kind\":\"case\",\"id\":\"{}\",\"role\":\"{}\",\"ids\":[{}],\"supplied_mask\":[{}],\"effective_mask\":[{}],\"scoring_mask\":[{}],\"expansion_positions\":[{}],\"all_vectors\":[{}],\"selected_vectors\":[{}]}}",
            f[0], f[1], list(&r.ids), list(&r.supplied_mask), list(&r.effective_mask), list(&r.scoring_mask),
            list(&r.expansion_positions), fl(&r.all_vectors), fl(&r.selected_vectors));
    }
    for line in fs::read_to_string(&a[3]).map_err(|e| e.to_string())?.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        println!("{{\"kind\":\"score\",\"id\":\"{}\",\"score\":{:e}}}", f[0], m.score(&text(f[1]), &text(f[2]))?);
    }
    if m.encode("x\0y", Role::Query).is_ok() { return Err("NUL accepted".into()); }
    Ok(())
}
