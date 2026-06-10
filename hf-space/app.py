"""CrispEmbed Gradio Space — text embeddings, math OCR, semantic search, and more.

Wraps the CrispEmbed C++ HTTP server (running on :8090) with a Gradio UI
served on :7860 (the only port HF Spaces exposes).
"""

import json
import os
import tempfile
import traceback

import gradio as gr
import numpy as np
import requests

SERVER_URL = os.environ.get("CRISPEMBED_SERVER_URL", "http://127.0.0.1:8090")


def _post(endpoint: str, payload: dict, timeout: int = 120) -> dict:
    try:
        r = requests.post(f"{SERVER_URL}{endpoint}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _get(endpoint: str) -> dict:
    try:
        r = requests.get(f"{SERVER_URL}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def cosine_sim(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ─── Text Embedding + Similarity ───────────────────────────────────────────

def embed_texts(text_a: str, text_b: str) -> str:
    if not text_a.strip():
        return "Please enter at least one text."
    texts = [text_a.strip()]
    if text_b.strip():
        texts.append(text_b.strip())

    result = _post("/embed", {"texts": texts})
    if "error" in result:
        return f"Error: {result['error']}"

    embeddings = result.get("embeddings", [])
    dim = result.get("dim", 0)
    lines = [f"Dimension: {dim}"]

    for i, emb in enumerate(embeddings):
        preview = ", ".join(f"{v:.4f}" for v in emb[:8])
        lines.append(f"Text {i+1}: [{preview}, ...] (norm={np.linalg.norm(emb):.4f})")

    if len(embeddings) == 2:
        sim = cosine_sim(embeddings[0], embeddings[1])
        lines.append(f"\nCosine similarity: {sim:.6f}")

    return "\n".join(lines)


# ─── Semantic Search ────────────────────────────────────────────────────────

def semantic_search(query: str, corpus: str, top_k: int) -> str:
    if not query.strip() or not corpus.strip():
        return "Please enter a query and a corpus (one sentence per line)."

    docs = [line.strip() for line in corpus.strip().split("\n") if line.strip()]
    if not docs:
        return "Corpus is empty."

    all_texts = [query.strip()] + docs
    result = _post("/embed", {"texts": all_texts})
    if "error" in result:
        return f"Error: {result['error']}"

    embeddings = result.get("embeddings", [])
    if len(embeddings) < 2:
        return "Error: not enough embeddings returned."

    query_emb = embeddings[0]
    scores = []
    for i, doc_emb in enumerate(embeddings[1:]):
        scores.append((cosine_sim(query_emb, doc_emb), docs[i]))

    scores.sort(key=lambda x: -x[0])
    lines = [f"Query: {query.strip()}", f"Results (top {min(top_k, len(scores))}):", ""]
    for rank, (score, doc) in enumerate(scores[:top_k], 1):
        lines.append(f"  {rank}. [{score:.4f}] {doc}")

    return "\n".join(lines)


# ─── Math OCR ──────────────────────────────────────────────────────────────

def math_ocr(image) -> str:
    if image is None:
        return "Please upload or capture a math equation image."

    try:
        import PIL.Image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp") as f:
            if isinstance(image, np.ndarray):
                PIL.Image.fromarray(image).save(f.name)
            elif hasattr(image, "save"):
                image.save(f.name)
            else:
                return f"Error: unexpected image type: {type(image)}"
            tmp_path = f.name

        result = _post("/math/ocr", {"image": tmp_path})

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        if "error" in result:
            return f"Error: {result['error']}"

        latex = result.get("latex", "")
        ms = result.get("ms", 0)
        return f"LaTeX: {latex}\n\nInference time: {ms} ms"
    except Exception as e:
        return f"Error: {traceback.format_exc()}"


# ─── Batch Embed (OpenAI-compatible) ───────────────────────────────────────

def batch_embed(texts_raw: str, model_name: str) -> str:
    if not texts_raw.strip():
        return "Please enter texts (one per line)."

    texts = [line.strip() for line in texts_raw.strip().split("\n") if line.strip()]
    result = _post("/v1/embeddings", {"input": texts, "model": model_name or "default"})
    if "error" in result:
        return f"Error: {result['error']}"

    data = result.get("data", [])
    lines = [f"Model: {result.get('model', '?')}", f"Embeddings: {len(data)}", ""]
    for item in data:
        emb = item.get("embedding", [])
        preview = ", ".join(f"{v:.4f}" for v in emb[:6])
        lines.append(f"  [{item.get('index', '?')}] dim={len(emb)}: [{preview}, ...]")

    usage = result.get("usage", {})
    if usage:
        lines.append(f"\nTokens: {usage.get('total_tokens', '?')}")

    return "\n".join(lines)


# ─── Health ────────────────────────────────────────────────────────────────

def health_check() -> str:
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=10)
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return f"Error: {e}"


# ─── Build UI ──────────────────────────────────────────────────────────────

with gr.Blocks(title="CrispEmbed", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CrispEmbed — Text Embedding, Semantic Search & Math OCR")
    gr.Markdown(
        "Powered by [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed) "
        "— lightweight embedding inference via ggml. "
        "58 models, 10 architectures, no Python runtime."
    )

    with gr.Tab("Similarity"):
        gr.Markdown("Enter two texts to compute cosine similarity.")
        text_a = gr.Textbox(label="Text A", placeholder="The quick brown fox...")
        text_b = gr.Textbox(label="Text B",
                            placeholder="A fast auburn canine...")
        embed_btn = gr.Button("Compare", variant="primary")
        embed_out = gr.Textbox(label="Result", lines=8)
        embed_btn.click(embed_texts, inputs=[text_a, text_b], outputs=embed_out)

        gr.Examples(
            examples=[
                ["The weather is lovely today.", "It's a beautiful day outside."],
                ["Machine learning is a branch of AI.",
                 "Cooking is a culinary art."],
                ["The cat sat on the mat.", "Dogs are loyal companions."],
            ],
            inputs=[text_a, text_b],
        )

    with gr.Tab("Semantic Search"):
        gr.Markdown("Enter a query and a corpus (one sentence per line). "
                     "Returns the most similar sentences ranked by cosine similarity.")
        search_query = gr.Textbox(label="Query",
                                  placeholder="renewable energy sources")
        search_corpus = gr.Textbox(
            label="Corpus (one sentence per line)", lines=8,
            value="Solar panels convert sunlight into electricity.\n"
                  "Wind turbines generate power from moving air.\n"
                  "Coal is a fossil fuel used in power plants.\n"
                  "Electric vehicles reduce carbon emissions.\n"
                  "The stock market fluctuated today.\n"
                  "Photosynthesis is how plants make food.\n"
                  "Nuclear fusion could provide limitless energy.\n"
                  "The recipe calls for two cups of flour.",
        )
        search_k = gr.Slider(1, 20, value=5, step=1, label="Top K")
        search_btn = gr.Button("Search", variant="primary")
        search_out = gr.Textbox(label="Results", lines=10)
        search_btn.click(semantic_search,
                         inputs=[search_query, search_corpus, search_k],
                         outputs=search_out)

    with gr.Tab("Math OCR"):
        gr.Markdown("Upload an image of a math equation. "
                     "Returns LaTeX via on-device neural OCR (HMER).")
        image_in = gr.Image(label="Math equation image", type="numpy")
        ocr_btn = gr.Button("Recognize", variant="primary")
        ocr_out = gr.Textbox(label="Result", lines=4)
        ocr_btn.click(math_ocr, inputs=image_in, outputs=ocr_out)

    with gr.Tab("Batch Embed (OpenAI API)"):
        gr.Markdown("Batch-embed texts via the OpenAI-compatible `/v1/embeddings` endpoint. "
                     "One text per line.")
        batch_texts = gr.Textbox(label="Texts (one per line)", lines=6,
                                 placeholder="Hello world\nGoodbye world")
        batch_model = gr.Textbox(label="Model name (optional)",
                                 value="all-MiniLM-L6-v2")
        batch_btn = gr.Button("Embed Batch", variant="primary")
        batch_out = gr.Textbox(label="Result", lines=10)
        batch_btn.click(batch_embed, inputs=[batch_texts, batch_model],
                        outputs=batch_out)

    with gr.Tab("Health"):
        gr.Markdown("Server status and loaded model info.")
        health_btn = gr.Button("Check Health")
        health_out = gr.Textbox(label="Server response", lines=10)
        health_btn.click(health_check, outputs=health_out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
