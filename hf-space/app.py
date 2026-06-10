"""CrispEmbed Gradio Space — text embeddings, math OCR, and similarity search.

Wraps the CrispEmbed C++ HTTP server (running on :8090) with a Gradio UI
served on :7860 (the only port HF Spaces exposes).

Architecture follows the CrispASR Space pattern:
  start.sh → crispembed-server (background) → python3 app.py (Gradio + FastAPI)
"""

import json
import os
import tempfile
import traceback

import gradio as gr
import numpy as np
import requests

SERVER_URL = os.environ.get("CRISPEMBED_SERVER_URL", "http://127.0.0.1:8090")


def _post(endpoint: str, payload: dict) -> dict:
    """POST to the CrispEmbed server and return the JSON response."""
    try:
        r = requests.post(f"{SERVER_URL}{endpoint}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-9))


# ─── Tab 1: Text Embedding + Similarity ────────────────────────────────────

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
        sim = cosine_similarity(embeddings[0], embeddings[1])
        lines.append(f"\nCosine similarity: {sim:.6f}")

    return "\n".join(lines)


# ─── Tab 2: Math OCR ───────────────────────────────────────────────────────

def math_ocr(image) -> str:
    if image is None:
        return "Please upload or capture a math equation image."

    try:
        import PIL.Image
        # Save the image to a temp file accessible by the C++ server.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp") as f:
            if isinstance(image, np.ndarray):
                PIL.Image.fromarray(image).save(f.name)
            elif hasattr(image, "save"):
                image.save(f.name)
            else:
                return f"Error: unexpected image type: {type(image)}"
            tmp_path = f.name

        # The C++ server's /math/ocr accepts {"image": "/path/to/file.png"}
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


# ─── Tab 3: Server Health ──────────────────────────────────────────────────

def health_check() -> str:
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=10)
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return f"Error: {e}"


# ─── Build the Gradio app ──────────────────────────────────────────────────

with gr.Blocks(title="CrispEmbed", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CrispEmbed — Text Embedding & Math OCR")
    gr.Markdown(
        "Powered by [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed) "
        "— lightweight embedding inference via ggml."
    )

    with gr.Tab("Text Embeddings"):
        gr.Markdown("Enter one or two texts to embed. If two are provided, "
                     "cosine similarity is computed.")
        text_a = gr.Textbox(label="Text A", placeholder="The quick brown fox...")
        text_b = gr.Textbox(label="Text B (optional)",
                            placeholder="A fast auburn canine...")
        embed_btn = gr.Button("Embed", variant="primary")
        embed_out = gr.Textbox(label="Result", lines=8)
        embed_btn.click(embed_texts, inputs=[text_a, text_b], outputs=embed_out)

        gr.Examples(
            examples=[
                ["The weather is lovely today.", "It's a beautiful day outside."],
                ["Machine learning is a branch of AI.",
                 "Cooking is a culinary art."],
                ["x^2 + 2x + 1 = (x+1)^2", ""],
            ],
            inputs=[text_a, text_b],
        )

    with gr.Tab("Math OCR"):
        gr.Markdown("Upload or capture an image of a math equation. "
                     "Returns LaTeX via on-device neural OCR.")
        image_in = gr.Image(label="Math equation image", type="numpy")
        ocr_btn = gr.Button("Recognize", variant="primary")
        ocr_out = gr.Textbox(label="Result", lines=4)
        ocr_btn.click(math_ocr, inputs=image_in, outputs=ocr_out)

    with gr.Tab("Health"):
        gr.Markdown("Server status and loaded model info.")
        health_btn = gr.Button("Check Health")
        health_out = gr.Textbox(label="Server response", lines=10)
        health_btn.click(health_check, outputs=health_out)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
