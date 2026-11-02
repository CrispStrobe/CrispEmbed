#!/usr/bin/env python3
"""Original PyTorch/Transformers OCR on the real-scan set (hosted only: Kaggle / GitHub Actions).

usage: ref_worker.py {uocr2|minicpm} PAGES_JSON OUT_JSON DEADLINE_EPOCH
PAGES_JSON: [{"id", "path", "max_tokens"}] in run order. Pinned checkpoints and loaders come from
tools/unlimited-ocr2/reference.py and tools/minicpm-ocr/reference.py (same contracts as the parity work).
Writes after every page; per page: text, generated ids, seconds, process peak RSS so far.
"""
import importlib.util, json, os, resource, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
MINICPM_PROMPT = 'Text Recognition:'


def load(path):
    spec = importlib.util.spec_from_file_location('ref_' + path.parent.name.replace('-', '_'), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def rss_mb(): return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def main(kind, pages_json, out_json, deadline):
    pages = json.load(open(pages_json))
    rep = dict(schema='crispembed-ocr-real-scans-reference-v1', kind=kind, cases=[], completed=False)
    save = lambda: Path(out_json).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    torch.use_deterministic_algorithms(True, warn_only=True)
    model_dir = Path(out_json).parent / ('ckpt-' + kind)
    t0 = time.time()
    if kind == 'uocr2':
        R = load(ROOT / 'tools/unlimited-ocr2/reference.py')
        cfg = json.loads(R.CONTRACT.read_text()); R.fetch(cfg, model_dir)
        torch.Tensor.cuda = lambda self, *a, **k: self
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.float32).eval()
        mod = sys.modules[type(model).__module__]; build = mod._uocr_build_inputs

        def cast(x):
            if isinstance(x, torch.Tensor): return x.float() if x.dtype == torch.bfloat16 else x
            if isinstance(x, (list, tuple)): return type(x)(cast(v) for v in x)
            return x
        mod._uocr_build_inputs = lambda *a, **k: cast(build(*a, **k))

        def run(p):
            res = model.infer(tok, image_file=p['path'], eval_mode=True, return_dict=True,
                              max_new_tokens=int(p['max_tokens']), temperature=0.0)
            ids = None
            for k in ('ids', 'token_ids', 'output_ids'):
                if res.get(k) is not None:
                    v = res[k]; ids = v.tolist() if hasattr(v, 'tolist') else list(v); break
            return res.get('text'), ids
    else:
        R = load(ROOT / 'tools/minicpm-ocr/reference.py')
        cfg = json.loads(R.CONTRACT.read_text()); R.fetch(cfg, model_dir)
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from PIL import Image
        if transformers.__version__ != cfg['reference_runtime']['version']:
            raise SystemExit(f"transformers {transformers.__version__} != pinned {cfg['reference_runtime']['version']}")
        proc = AutoProcessor.from_pretrained(model_dir)
        model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float32).eval()

        def run(p):
            image = Image.open(p['path']).convert('RGB')
            msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': MINICPM_PROMPT}]}]
            prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inputs = proc(text=[prompt], images=[[image]], return_tensors='pt')
            gen = model.generate(**inputs, do_sample=False, max_new_tokens=int(p['max_tokens']))
            ids = gen[0, inputs['input_ids'].shape[1]:].tolist()
            return proc.tokenizer.decode(ids, skip_special_tokens=True), ids
    import transformers
    rep.update(revision=cfg['revision'], load_seconds=round(time.time() - t0, 1),
               versions=dict(torch=torch.__version__, transformers=transformers.__version__)); save()
    for p in pages:
        if time.time() > deadline:
            rep['cases'].append(dict(id=p['id'], skipped='time budget')); save(); continue
        t = time.time()
        try:
            with torch.inference_mode():
                text, ids = run(p)
            row = dict(id=p['id'], seconds=round(time.time() - t, 1), text=text, ids=ids, max_tokens=p['max_tokens'],
                       cap_hit=bool(ids) and len(ids) >= int(p['max_tokens']))
        except Exception as e:
            row = dict(id=p['id'], seconds=round(time.time() - t, 1), error=repr(e)[:2000])
        row['peak_rss_mb_so_far'] = rss_mb()
        rep['cases'].append(row); save(); print('reference', kind, p['id'], row.get('seconds'), flush=True)
    rep['completed'] = True; save()


if __name__ == '__main__':
    if os.environ.get('GITHUB_ACTIONS') != 'true' and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        sys.exit('hosted-only (Kaggle / GitHub Actions)')
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
