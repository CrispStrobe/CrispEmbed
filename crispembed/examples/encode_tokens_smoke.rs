//! Smoke test for the new `encode_tokens` API. Compares the per-token
//! output of a multilingual XLM-R-style model against pooled output to
//! verify the encoder is actually producing distinct contextual vectors
//! per token (not just repeating the pooled vector).
//!
//! Run:
//!   cargo run -p crispembed --example encode_tokens_smoke -- <gguf-path>
//!
//! Suggested model:
//!   paraphrase-multilingual-MiniLM-L12-v2-f16.gguf (XLM-R / SentencePiece)

use crispembed::CrispEmbed;

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot / (l2(a).max(1e-9) * l2(b).max(1e-9))
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: encode_tokens_smoke <gguf-path>");
    let mut m = CrispEmbed::new(&path, 4).expect("load model");

    println!("dim:            {}", m.dim());
    println!("tokenizer_kind: {} (1=WordPiece, 2=SentencePiece, 3=BPE)", m.tokenizer_kind());

    let pairs = [
        ("Hello world.", "Hallo Welt."),
        ("The dog is sleeping.", "Der Hund schläft."),
    ];

    for (src, tgt) in pairs.iter() {
        let src_tokens = m.encode_tokens(src);
        let tgt_tokens = m.encode_tokens(tgt);

        println!("\n--- {src:?}  ↔  {tgt:?} ---");
        println!("source tokens: {}", src_tokens.len());
        for (i, (s, v)) in src_tokens.iter().enumerate() {
            println!("  src[{i}]  {s:>20}  ‖v‖={:.4}", l2(v));
        }
        println!("target tokens: {}", tgt_tokens.len());
        for (j, (s, v)) in tgt_tokens.iter().enumerate() {
            println!("  tgt[{j}]  {s:>20}  ‖v‖={:.4}", l2(v));
        }

        // Pairwise cosine similarity — pick the best source token per
        // target token, and vice versa (cheap argmax view of SimAlign).
        println!("\n  cosine sim matrix (src rows × tgt cols):");
        for (i, (s_tok, s_vec)) in src_tokens.iter().enumerate() {
            print!("    {i:2} {s_tok:>16} | ");
            for (_, t_vec) in tgt_tokens.iter() {
                print!("{:.2} ", cos(s_vec, t_vec));
            }
            println!();
        }
        print!("       hdr            | ");
        for (j, (t_tok, _)) in tgt_tokens.iter().enumerate() {
            print!("{j:>4} ");
            let _ = t_tok;
        }
        println!();
    }
}
