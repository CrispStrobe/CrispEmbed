#!/usr/bin/env python3
"""CrispEmbed — batch imatrix quantization + A/B + upload (Kaggle kernel).

Follows the standard CrispEmbed Kaggle regime (kaggle_harness = kh):
  * kh.init_progress()      — line-buffered I/O + JSONL progress, pushed to HF
  * kh.resolve_hf_token()   — env → Kaggle Secret → mounted DATASET (hf_token.txt)
  * kh.install_build_toolchain() + ccache warmed from the crispasr-ccache dataset
  * kh.build_heartbeat(...)  — 30 s heartbeat around every long step
Attach BOTH datasets in kernel-metadata.json:
    "dataset_sources": ["chr1str/crispasr-hf-token", "chr1str/crispasr-ccache"]

Processes a LIST of models in ONE kernel run (build once, then per model:
download source → calibrate → quantize (+imatrix) → A/B → upload → rm → next) —
the "rm, next" loop. Select with the MODELS env var (comma list) or DEFAULT_BATCH
below. Per-model failures are isolated (logged, skipped).

For each model the source is the existing full-precision GGUF ALREADY in its
cstr/<name>-GGUF repo (auto-detected: largest non-quant .gguf) — no HF
re-conversion, so LoRA models (jina-v5) and odd namings just work. Imatrix
outputs use DISTINCT names and NEVER overwrite the canonical q8_0/q4_k baselines.
"""
import os, re, sys, json, math, time, shutil, subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
if not WORK.exists():
    WORK = Path("/tmp/crisp-imatrix-work"); WORK.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
BRANCH   = os.environ.get("CRISP_BRANCH", "main")   # C1 is on main

# ── bootstrap kaggle_harness (kh): CrispASR clone, else bundled sibling copy ───
CRISPASR_DIR = WORK / "CrispASR"
if not CRISPASR_DIR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
            "https://github.com/CrispStrobe/CrispASR.git", str(CRISPASR_DIR)])
        sys.path.insert(0, str(CRISPASR_DIR / "tools" / "kaggle"))
    except Exception:
        pass
sys.path.insert(0, str(Path(__file__).resolve().parent))   # bundled fallback
import kaggle_harness as kh

# ── models to process (each maps to repo cstr/<name>-GGUF) ────────────────────
# The first three re-run to backfill their A/B summary (added after their initial
# run). The rest extend imatrix coverage across the embedding roster.
DEFAULT_BATCH = [
    # backfill summaries:
    "lfm2-embed", "jina-v5-nano", "bge-m3",
    # (e5-large, jina-v5-small already have summaries)
    # new coverage:
    "bge-large-en-v1.5", "bge-base-en-v1.5", "bge-small-en-v1.5",
    "mxbai-embed-large-v1", "multilingual-e5-base", "multilingual-e5-small",
    "nomic-embed-text-v1.5", "nomic-embed-text-v2-moe", "arctic-embed-l-v2",
    "gte-base-en-v1.5", "gte-large-en-v1.5", "octen-0.6b", "f2llm-v2-0.6b",
    "qwen3-embed-0.6b", "pixie-rune-v1",
    # group 2 (<=2.4GB, fit Kaggle CPU RAM):
    "all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2", "gte-small",
    "arctic-embed-xs", "snowflake-arctic-embed-m", "snowflake-arctic-embed-l",
    "paraphrase-multilingual-MiniLM-L12-v2", "harrier-270m", "harrier-0.6b",
    # embeddinggemma-300m re-enabled: the dense.* keep-F32 guard in
    # tools/quantize.cpp fixes the "tensor read out of bounds" load failure.
    "embeddinggemma-300m",
    # group 3 — large decoder embedders (f32 base 16-30GB). Handled by the big-base
    # path: calibrate/gold on the q8_0 (fits RAM), quantize from f32 (streaming),
    # stage in /tmp. 4B first to validate before the 30GB 8B downloads.
    "octen-4b", "qwen3-embed-4b", "octen-8b", "qwen3-embed-8b",
]
RUN = [m.strip() for m in os.environ.get("MODELS", "").split(",") if m.strip()] or DEFAULT_BATCH

# Optional per-model overrides: {name: {"hf_out":..., "quants":...}}.
# Default hf_out = cstr/<name>-GGUF; default quants = QSPECS.
OVERRIDES = {}

# ── reranker (cross-encoder) support ─────────────────────────────────────────
# Rerankers score (query, doc) pairs rather than producing a pooled embedding, so
# `mean_cos` is meaningless. A/B metric = Kendall-tau on the doc ranking (what a
# reranker is FOR) vs the q8/full-precision gold, with mean|dscore| as tiebreaker.
# Calibration runs the `--rerank` path — the imatrix collector fires on it exactly
# like the embed path (verified locally: ms-marco-MiniLM-L-6-v2 q4_k+im halves
# mean|dscore| 0.0102->0.0064; iq4_xs+im gives tau 1.0).
MODE = {m: "rerank" for m in (
    "bge-reranker-base", "bge-reranker-v2-m3", "jina-reranker-v2-base-multilingual",
    "ms-marco-MiniLM-L-6-v2", "ms-marco-MiniLM-L-12-v2",
    "mxbai-rerank-base-v1", "mxbai-rerank-xsmall-v1",
)}

# (query, [docs]) calibration pairs — mixed relevance across domains. imatrix
# quality tracks calibration relevance, so exercise realistic query/doc text.
RERANK_CALIB = [
    ("what causes rain", ["Rain forms when water vapor condenses into droplets.", "The stock market fell sharply today.", "Clouds release precipitation when saturated.", "Cats are popular pets worldwide."]),
    ("how do vaccines work", ["Vaccines train the immune system using antigens.", "A recipe for chocolate cake needs flour and eggs.", "Immunization prompts antibody production.", "The train departs at noon."]),
    ("best programming language for data science", ["Python is widely used for data analysis and ML.", "Football is played on a rectangular field.", "R is favored for statistical computing.", "The weather is cold in winter."]),
    ("symptoms of the flu", ["Influenza causes fever, cough, and body aches.", "Paris is the capital of France.", "Flu often brings fatigue and sore throat.", "Diamonds form under high pressure."]),
    ("how to reduce carbon emissions", ["Switching to renewable energy cuts emissions.", "The novel was published in 1925.", "Public transit reduces per-capita CO2.", "Guitars have six strings."]),
    ("history of the Roman Empire", ["Rome expanded across the Mediterranean by conquest.", "Photosynthesis converts sunlight to energy.", "The empire fell in the 5th century.", "Smartphones use lithium batteries."]),
    ("nutritional benefits of vegetables", ["Vegetables provide fiber, vitamins, and minerals.", "Jupiter is the largest planet.", "Leafy greens are rich in iron.", "The marathon is 42 kilometers."]),
    ("how neural networks learn", ["Networks adjust weights via backpropagation.", "Coffee contains caffeine.", "Gradient descent minimizes the loss.", "The Nile is a long river."]),
    ("effects of sleep deprivation", ["Lack of sleep impairs memory and focus.", "Mount Everest is the tallest mountain.", "Sleep loss weakens the immune system.", "Violins are string instruments."]),
    ("renewable energy sources", ["Solar and wind are clean energy sources.", "The recipe calls for two cups of sugar.", "Hydropower generates electricity from water.", "Penguins live in cold climates."]),
    # German (DE) calibration pairs
    ("Wie entsteht Regen", ["Regen entsteht, wenn Wasserdampf zu Tropfen kondensiert.", "Die Börse fiel heute stark.", "Wolken geben Niederschlag ab, wenn sie gesättigt sind.", "Hunde sind beliebte Haustiere."]),
    ("Wie wirken Impfstoffe", ["Impfstoffe trainieren das Immunsystem mit Antigenen.", "Ein Kuchenrezept braucht Mehl und Eier.", "Eine Impfung regt die Antikörperbildung an.", "Der Bus kommt um zwölf."]),
    ("Beste Programmiersprache für Datenanalyse", ["Python wird häufig für Datenanalyse und maschinelles Lernen genutzt.", "Fußball wird auf einem Rasenfeld gespielt.", "R eignet sich gut für statistische Berechnungen.", "Im Winter ist es kalt."]),
    ("Geschichte des Römischen Reiches", ["Rom dehnte sich durch Eroberungen im Mittelmeerraum aus.", "Photosynthese wandelt Sonnenlicht in Energie um.", "Das Reich fiel im fünften Jahrhundert.", "Smartphones nutzen Lithium-Akkus."]),
]
RERANK_EVAL = [
    ("treatment for headaches", ["Pain relievers like ibuprofen ease headaches.", "The bridge spans two kilometers.", "Rest and hydration reduce headache severity.", "Tomatoes are technically fruits."]),
    ("how do plants make food", ["Plants use photosynthesis to make glucose.", "The car engine has six cylinders.", "Chlorophyll captures light energy.", "Chess has sixty-four squares."]),
    ("causes of climate change", ["Greenhouse gas emissions drive warming.", "The museum opens at nine.", "Deforestation raises atmospheric CO2.", "Owls are nocturnal birds."]),
    ("benefits of regular exercise", ["Exercise strengthens the heart and muscles.", "The library has many books.", "Physical activity improves mood.", "Saturn has prominent rings."]),
    ("how computers store data", ["Data is stored as bits on drives and memory.", "Roses are often red.", "SSDs use flash memory cells.", "The concert starts at eight."]),
    # German (DE) — same relevance structure (docs 1 & 3 relevant, 2 & 4 off-topic)
    ("Wie funktioniert eine Solarzelle", ["Eine Solarzelle wandelt Sonnenlicht in elektrischen Strom um.", "Der Zug fährt um acht Uhr ab.", "Photovoltaik nutzt den photoelektrischen Effekt.", "Die Katze schläft auf dem Sofa."]),
    ("Symptome einer Erkältung", ["Eine Erkältung verursacht Husten und Schnupfen.", "Der Fluss ist zwei Kilometer lang.", "Halsschmerzen und Müdigkeit treten häufig auf.", "Tomaten sind botanisch Früchte."]),
    ("Ursachen des Klimawandels", ["Treibhausgase aus fossilen Brennstoffen erwärmen die Erde.", "Das Museum öffnet um neun Uhr.", "Abholzung erhöht den CO2-Gehalt der Luft.", "Eulen sind nachtaktive Vögel."]),
    ("Vorteile von regelmäßigem Sport", ["Sport stärkt Herz und Muskeln.", "Die Bibliothek hat viele Bücher.", "Bewegung verbessert die Stimmung.", "Der Saturn hat auffällige Ringe."]),
]

# ── NER (token-classification) support ───────────────────────────────────────
# Fixed-label NER (bert-base-NER, xlmr-ner-hrl) runs the BERT-NER path, whose
# encoder is a shared crispembed_context — so the imatrix collector fires on its
# sched with zero code change. A/B metric = micro span-F1 (exact (start,end,label)
# match) of the quant's entities vs the full-precision gold. (GLiNER models use a
# gallocr compute path with no eval-callback hook, so they're not covered here.)
# GLiNER (zero-shot span) also uses the --ner path; its compute now routes through a
# sched during calibration (see gliner_ner.cpp) so the collector fires. --ner without
# explicit labels uses the CLI default types, which the NER corpora exercise.
MODE.update({m: "ner" for m in ("bert-base-NER", "xlmr-ner-hrl", "gliner-deberta", "sauerkraut-gliner-lfm")})

NER_CALIB = [
    "Barack Obama met Angela Merkel in Berlin last Tuesday.",
    "Apple and Microsoft announced a partnership in California.",
    "The United Nations held a summit in Geneva about climate change.",
    "Elon Musk visited the Tesla factory near Berlin.",
    "Cristiano Ronaldo signed with a football club in Saudi Arabia.",
    "Amazon opened a new office in Toronto, Canada.",
    "Marie Curie was born in Warsaw and later worked in Paris.",
    "Google DeepMind is headquartered in London, England.",
    "The World Health Organization is based in Geneva, Switzerland.",
    "Nelson Mandela led South Africa after apartheid ended.",
    # German (DE) calibration — entity-rich
    "Ursula von der Leyen sprach in Brüssel mit Vertretern der NATO.",
    "Volkswagen und BMW stellten neue Modelle in Wolfsburg vor.",
    "Johann Wolfgang von Goethe wurde in Frankfurt geboren und lebte in Weimar.",
    "Die Vereinten Nationen hielten in Genf einen Gipfel zum Klima ab.",
    "Bayern München gewann das Spiel gegen Borussia Dortmund in München.",
]
NER_EVAL = [
    "Joe Biden spoke with Emmanuel Macron about NATO in Brussels.",
    "Samsung and Sony compete in the electronics market across Japan.",
    "The European Union met in Strasbourg to discuss the annual budget.",
    "Serena Williams won a tennis tournament in Melbourne, Australia.",
    "IBM and Oracle opened data centers in Texas and Virginia.",
    "Albert Einstein studied in Zurich before moving to Princeton.",
    # German (DE) — entity-rich (PER / ORG / LOC)
    "Olaf Scholz traf Emmanuel Macron in Berlin zu einem Gespräch.",
    "Siemens und Bosch eröffneten ein Werk in München.",
    "Angela Merkel wuchs in Hamburg auf und studierte in Leipzig.",
    "Die Europäische Union tagte in Straßburg über den Haushalt.",
]

# ── ColBERT (multi-vector) support ───────────────────────────────────────────
# lfm2-colbert emits per-token vectors via --colbert; its LFM2 backbone shares the
# instrumented lfm2 sched, so the collector fires unchanged. A/B metric = mean
# per-token cosine (same text -> aligned tokens) vs full-precision gold.
# (splade-pp sparse has its own MODE below, now that the MLM head is restored.)
MODE.update({m: "colbert" for m in ("lfm2-colbert",)})

COLBERT_CALIB = [
    "Machine learning transforms raw text into vector representations.",
    "The weather forecast predicts rain across the region tomorrow.",
    "Quantum computing research advances steadily each year.",
    "Financial markets reacted to the central bank announcement.",
    "Neural networks learn hierarchical features from data.",
    # German (DE)
    "Maschinelles Lernen wandelt Text in Vektoren um.",
    "Die Wettervorhersage sagt für morgen Regen voraus.",
    "Erneuerbare Energien werden weltweit immer wichtiger.",
    "Verteilte Systeme nutzen Konsensprotokolle für Konsistenz.",
    "Photosynthese wandelt Sonnenlicht in chemische Energie um.",
]
COLBERT_EVAL = [
    "Information retrieval systems rank documents by relevance.",
    "Climate change affects ecosystems around the globe.",
    "Deep learning models require large training datasets.",
    # German (DE)
    "Suchsysteme ordnen Dokumente nach Relevanz.",
    "Der Klimawandel betrifft Ökosysteme auf der ganzen Welt.",
    "Datenbanken indexieren Datensätze für schnelle Suche.",
]

# ── SPARSE (SPLADE) support ──────────────────────────────────────────────────
# splade-pp emits sparse term weights via --sparse (MLM head max-pooled over the
# vocab). The head runs through run_encoder_raw's ctx->sched, so the collector
# fires unchanged. A/B metric = cosine of the sparse term-weight vectors vs the
# full-precision gold (reuse the embed calib/eval text corpora).
MODE.update({m: "sparse" for m in ("splade-pp-en-v1",)})

# QSPECS: (qtype, use_imatrix, upload_name_template | None). upload_name None =
# A/B reference only (never uploaded — don't clobber baselines). imatrix variants
# get DISTINCT names: q4_k+imatrix -> *-q4_k-imatrix.gguf, iq4_xs -> *-iq4_xs.gguf.
QSPECS = [
    ("q8_0",   False, None),                          # A/B reference (baseline exists)
    ("q4_k",   False, None),                          # A/B baseline (no imatrix) — shows the delta
    ("q4_k",   True,  "{prefix}-q4_k-imatrix.gguf"),
    ("iq4_xs", True,  "{prefix}-iq4_xs.gguf"),
]

_CALIB_FB = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models transform raw text into dense vector representations.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Quarterly revenue grew 12% year over year, beating analyst expectations.",
    "Gravitational waves were first directly detected by LIGO in 2015.",
    "In distributed systems, consensus protocols like Raft ensure consistency.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "A binary search tree keeps keys in sorted order for O(log n) lookup.",
    "Neural networks learn hierarchical features through backpropagation.",
    "Cache invalidation is one of the hardest problems in computer science.",
]
_EVAL_FB = [
    "A large language model can summarize documents and answer questions.",
    "The stock market rallied after the earnings report was released.",
    "import numpy as np; a = np.zeros((3, 3)); a[1, 1] = 1.0",
    "A hash map provides average constant-time insertion and lookup.",
    "Our return policy allows exchanges within thirty days of purchase.",
]


def read_corpus(name, fallback):
    p = Path(__file__).resolve().parent / name
    if p.exists():
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return fallback


def embed(cli, model, texts):
    t0 = time.time()
    r = subprocess.run([str(cli), "-m", str(model), "--json", *texts],
                       capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"embed failed for {model}:\n{r.stderr[-1500:]}")
    data = json.loads(r.stdout)
    return [[float(x) for x in o["embedding"]] for o in data if o.get("embedding")], dt


def mean_cos(a, b):
    def cos(u, v):
        d = sum(x*y for x, y in zip(u, v))
        nu = math.sqrt(sum(x*x for x in u)); nv = math.sqrt(sum(y*y for y in v))
        return d/(nu*nv) if nu and nv else 0.0
    n = min(len(a), len(b))
    return (sum(cos(a[i], b[i]) for i in range(n)) / n if n else float("nan")), n


def rerank_scores(cli, model, query, docs):
    """Return {doc_index: score} from the cross-encoder --rerank path."""
    r = subprocess.run([str(cli), "-m", str(model), "--json", "--rerank", query, *docs],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"rerank failed for {model}:\n{r.stderr[-1500:]}")
    return {res["index"]: res["score"] for res in json.loads(r.stdout)["results"]}


def kendall_tau(a, b):
    """Kendall-tau between two {index: score} maps over their shared indices."""
    idx = sorted(set(a) & set(b)); con = dis = 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            x, y = idx[i], idx[j]
            sa, sb = a[x] - a[y], b[x] - b[y]
            if sa == 0 or sb == 0:
                continue
            con += (sa > 0) == (sb > 0); dis += (sa > 0) != (sb > 0)
    tot = con + dis
    return (con - dis) / tot if tot else 1.0


def rerank_ab(cli, model, gold):
    """Mean Kendall-tau + mean|dscore| of `model` vs gold scores over RERANK_EVAL."""
    taus, dabs = [], []
    for (q, docs), g in zip(RERANK_EVAL, gold):
        s = rerank_scores(cli, model, q, docs)
        taus.append(kendall_tau(g, s))
        shared = [i for i in g if i in s]
        dabs.append(sum(abs(g[i] - s[i]) for i in shared) / len(shared) if shared else 0.0)
    return sum(taus) / len(taus), sum(dabs) / len(dabs)


def ner_entities(cli, model, text):
    """Return the set of (start, end, label) spans from the --ner path."""
    r = subprocess.run([str(cli), "-m", str(model), "--json", "--ner", text],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ner failed for {model}:\n{r.stderr[-1500:]}")
    return {(e["start"], e["end"], e["label"]) for e in json.loads(r.stdout).get("entities", [])}


def ner_ab(cli, model, gold):
    """Micro span-F1 (exact start/end/label match) of `model` vs gold over NER_EVAL."""
    tp = fp = fn = 0
    for text, g in zip(NER_EVAL, gold):
        p = ner_entities(cli, model, text)
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def colbert_vecs(cli, model, text):
    """Return the list of per-token vectors from the --colbert path."""
    r = subprocess.run([str(cli), "-m", str(model), "--json", "--colbert", text],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"colbert failed for {model}:\n{r.stderr[-1500:]}")
    return json.loads(r.stdout)[0]["vectors"]


def colbert_ab(cli, model, gold):
    """Mean per-token cosine of `model`'s ColBERT vectors vs gold over COLBERT_EVAL."""
    def _cos(a, b):
        d = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
        return d / (na * nb) if na and nb else 0.0
    tot = n = 0
    for text, g in zip(COLBERT_EVAL, gold):
        pv = colbert_vecs(cli, model, text)
        for gt, pt in zip(g, pv):
            tot += _cos(gt, pt); n += 1
    return tot / n if n else float("nan")


def sparse_vec(cli, model, text):
    """Return {token_id: weight} sparse term-weight vector from the --sparse path."""
    r = subprocess.run([str(cli), "-m", str(model), "--json", "--sparse", text],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"sparse failed for {model}:\n{r.stderr[-1500:]}")
    return {e["token_id"]: e["weight"] for e in json.loads(r.stdout)[0]["sparse"]}


def sparse_ab(cli, model, gold):
    """Mean cosine of `model`'s sparse vectors vs gold over the eval texts."""
    def _scos(a, b):
        keys = set(a) | set(b)
        d = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
        na = math.sqrt(sum(v * v for v in a.values())); nb = math.sqrt(sum(v * v for v in b.values()))
        return d / (na * nb) if na and nb else 0.0
    vals = [_scos(sparse_vec(cli, model, t), g) for t, g in gold]
    return sum(vals) / len(vals) if vals else float("nan")


_QUANT_RE = re.compile(r'(^|[-.])(q\d|iq\d|q4_k|q5_k|q6_k|q8_0|q4_0|q5_0|q5_1|bf16|imatrix)', re.I)
# LoRA / task-adapter variants (jina-v5 ships these at the SAME size as the base
# retrieval model, so "largest non-quant" alone would wrongly pick one).
_TASK_RE  = re.compile(r'-(classification|clustering|text-matching|retrieval|separation|code|sts)$', re.I)

def pick_base_gguf(ggs, name):
    """Full-precision source, from a {filename: size} dict. Prefer the exact base
    name ({name}.gguf / -f16 / -f32); else the largest .gguf that is neither a
    quant nor a LoRA task-adapter variant. Returns (filename, prefix)."""
    if not ggs:
        raise RuntimeError(f"no .gguf for {name}")
    for cand in (f"{name}.gguf", f"{name}-f16.gguf", f"{name}-f32.gguf"):
        if cand in ggs:
            prefix = cand[:-5]
            for suf in ("-f16", "-f32"):
                if prefix.endswith(suf): prefix = prefix[:-len(suf)]
            return cand, prefix
    def ok(stem):
        return not _QUANT_RE.search(stem) and not _TASK_RE.search(stem)
    base = {f: sz for f, sz in ggs.items() if ok(f[:-5])}
    pool = base or {f: sz for f, sz in ggs.items() if not _QUANT_RE.search(f[:-5])} or ggs
    fn = max(pool, key=pool.get)
    prefix = fn[:-5]
    for suf in ("-f16", "-f32", ".f16", ".f32"):
        if prefix.endswith(suf):
            prefix = prefix[: -len(suf)]
    return fn, prefix


# Bases larger than this can't be loaded for inference on Kaggle's ~13GB RAM, so
# calibrate + A/B-gold run on the q8_0 (fits RAM; the imatrix is activation stats,
# ~identical on q8 vs f32) while quantization still reads the full-precision base
# (streaming, per-tensor). Big files stage in /tmp (~70GB) not /kaggle/working (~20GB).
BIG_BYTES = 10 * 1000**3

def process(name, cli, quant, api, calib, eval_):
    from huggingface_hub import hf_hub_download
    ov = OVERRIDES.get(name, {})
    hf_out = ov.get("hf_out", f"cstr/{name}-GGUF")
    quants = ov.get("quants", QSPECS)
    kh.step("model.start", model=name, repo=hf_out)

    ggs = {s.rfilename: (s.size or 0) for s in api.repo_info(hf_out, files_metadata=True).siblings
           if s.rfilename.endswith(".gguf")}
    base_fn, prefix = pick_base_gguf(ggs, name)
    base_sz = ggs.get(base_fn, 0)
    big = base_sz > BIG_BYTES
    stage = Path("/tmp/crisp-stage") if big else WORK
    srcdir = stage / "src" / name
    srcdir.mkdir(parents=True, exist_ok=True)

    # For big bases (>10GB) download + quantize + calibrate all from the q8_0
    # (4-8GB). crispembed-quantize dequantizes the q8 source before re-quantizing,
    # and q8_0 is ~lossless (cos ~0.9998) so q4-from-q8 ≈ q4-from-f32 — while
    # avoiding the 16-30GB f32 download that stalls on Kaggle nodes. Small models
    # still use the f32 base directly.
    q8_fn = f"{prefix}-q8_0.gguf"
    src_fn = q8_fn if (big and q8_fn in ggs) else base_fn
    with kh.build_heartbeat(f"{name}.download.src"):
        qsrc = Path(hf_hub_download(hf_out, src_fn, token=api.token, local_dir=str(srcdir)))
    csrc = qsrc
    goldlabel = "q8_0" if src_fn.endswith("-q8_0.gguf") else "full-precision"
    kh.step("model.source", model=name, src=src_fn, prefix=prefix,
            src_gb=round(ggs.get(src_fn, 0)/1e9, 2), big=big)

    mode = MODE.get(name, "embed")     # "embed" (pooled) / "rerank" (cross-encoder) / "ner"
    metric = {"rerank": "tau", "ner": "f1"}.get(mode, "cos")
    imat = stage / f"{prefix}.imatrix"; imat.unlink(missing_ok=True)
    env = dict(os.environ, CRISPEMBED_IMATRIX_OUT=str(imat))
    with kh.build_heartbeat(f"{name}.calibrate"):
        if mode == "rerank":
            # collector fires on the --rerank path too; run every calib pair
            outlen, err = 0, ""
            for q, docs in RERANK_CALIB:
                cal = subprocess.run([str(cli), "-m", str(csrc), "--json", "--rerank", q, *docs],
                                     env=env, capture_output=True, text=True)
                if cal.returncode != 0:
                    raise RuntimeError(f"rerank calibration rc={cal.returncode} for {name}; "
                                       f"stderr tail:\n{cal.stderr[-1200:]}")
                outlen += len(cal.stdout); err = cal.stderr
        elif mode == "ner":
            # BERT-NER encoder is a shared crispembed_context → collector fires on --ner
            outlen, err = 0, ""
            for t in NER_CALIB:
                cal = subprocess.run([str(cli), "-m", str(csrc), "--json", "--ner", t],
                                     env=env, capture_output=True, text=True)
                if cal.returncode != 0:
                    raise RuntimeError(f"ner calibration rc={cal.returncode} for {name}; "
                                       f"stderr tail:\n{cal.stderr[-1200:]}")
                outlen += len(cal.stdout); err = cal.stderr
        elif mode == "colbert":
            outlen, err = 0, ""
            for t in COLBERT_CALIB:
                cal = subprocess.run([str(cli), "-m", str(csrc), "--json", "--colbert", t],
                                     env=env, capture_output=True, text=True)
                if cal.returncode != 0:
                    raise RuntimeError(f"colbert calibration rc={cal.returncode} for {name}; "
                                       f"stderr tail:\n{cal.stderr[-1200:]}")
                outlen += len(cal.stdout); err = cal.stderr
        elif mode == "sparse":
            outlen, err = 0, ""
            for t in calib:
                cal = subprocess.run([str(cli), "-m", str(csrc), "--json", "--sparse", t],
                                     env=env, capture_output=True, text=True)
                if cal.returncode != 0:
                    raise RuntimeError(f"sparse calibration rc={cal.returncode} for {name}; "
                                       f"stderr tail:\n{cal.stderr[-1200:]}")
                outlen += len(cal.stdout); err = cal.stderr
        else:
            cal = subprocess.run([str(cli), "-m", str(csrc), "--json", *calib],
                                 env=env, capture_output=True, text=True)
            if cal.returncode != 0:
                raise RuntimeError(f"calibration rc={cal.returncode} for {name}; stderr tail:\n{cal.stderr[-1200:]}")
            outlen, err = len(cal.stdout), cal.stderr
    # Fail LOUDLY if calibration didn't produce the imatrix — otherwise the quantizer
    # silently falls back to NON-imatrix and uploads mislabeled "-imatrix" quants
    # (observed on qwen3-embed-8b: clean_exit skipped the atexit flush). Surface stderr.
    if not imat.exists() or imat.stat().st_size == 0:
        raise RuntimeError(f"calibration produced NO imatrix at {imat} for {name} "
                           f"(rc=0); stdout {outlen}B; stderr tail:\n{err[-1200:]}")

    # gold = calib source (q8_0 for big / full-precision base otherwise, ~lossless)
    if mode == "rerank":
        gold = [rerank_scores(cli, csrc, q, docs) for q, docs in RERANK_EVAL]
    elif mode == "ner":
        gold = [ner_entities(cli, csrc, t) for t in NER_EVAL]
    elif mode == "colbert":
        gold = [colbert_vecs(cli, csrc, t) for t in COLBERT_EVAL]
    elif mode == "sparse":
        gold = [(t, sparse_vec(cli, csrc, t)) for t in eval_]
    else:
        gold = embed(cli, csrc, eval_)[0]

    report = []
    for qtype, use_im, up_tmpl in quants:
        if big and qtype == "q8_0" and not use_im:
            continue  # q8_0 IS the gold for big models — no A/B needed
        tag = f"{qtype}{'-im' if use_im else ''}"
        out = stage / f"{prefix}-{tag}.gguf"
        cmd = [str(quant), str(qsrc), str(out), qtype] + (["--imatrix", str(imat)] if use_im else [])
        with kh.build_heartbeat(f"{name}.quant.{tag}"):
            subprocess.check_call(cmd)
        if mode == "rerank":
            val, dscore = rerank_ab(cli, out, gold)   # val = mean Kendall-tau
            extra = f" dscore={dscore:.4f}"
        elif mode == "ner":
            val = ner_ab(cli, out, gold)              # val = micro span-F1
            extra = ""
        elif mode == "colbert":
            val = colbert_ab(cli, out, gold)          # val = mean per-token cosine
            extra = ""
        elif mode == "sparse":
            val = sparse_ab(cli, out, gold)           # val = mean sparse-vector cosine
            extra = ""
        else:
            vecs, _ = embed(cli, out, eval_)
            val, _ = mean_cos(vecs, gold); extra = ""
        mb = out.stat().st_size / 1e6
        upname = up_tmpl.format(prefix=prefix) if up_tmpl else "(A/B only)"
        kh.step(f"{name}.ab.{tag}", imatrix=use_im, **{f"{metric}_vs_gold": round(val, 6)},
                gold=goldlabel, size_mb=round(mb, 1), upload=upname)
        report.append(f"{qtype:7s} imatrix={int(use_im)}  {metric}_vs_{goldlabel}={val:.6f}{extra}  {mb:7.1f}MB  -> {upname}")
        if up_tmpl:
            with kh.build_heartbeat(f"{name}.upload.{tag}"):
                api.upload_file(path_or_fileobj=str(out), path_in_repo=upname,
                    repo_id=hf_out, repo_type="model",
                    commit_message=f"{qtype} +imatrix ({metric}_vs_{goldlabel}={val:.4f})")
        out.unlink(missing_ok=True)

    calib_n = {"rerank": len(RERANK_CALIB), "ner": len(NER_CALIB), "colbert": len(COLBERT_CALIB)}.get(mode, len(calib))
    eval_n = {"rerank": len(RERANK_EVAL), "ner": len(NER_EVAL), "colbert": len(COLBERT_EVAL)}.get(mode, len(eval_))
    summary = (f"imatrix A/B — {name} ({hf_out}), {metric} vs {goldlabel} gold, "
               f"n={eval_n}, calib={calib_n}, quant_src={base_fn}\n" + "\n".join(report) + "\n")
    summ = stage / f"{prefix}-imatrix-ab.txt"; summ.write_text(summary)
    for p, msg in [(summ, f"A/B summary ({metric} vs gold)"), (imat, "importance matrix (calibration)")]:
        with kh.build_heartbeat(f"{name}.upload.meta"):
            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                repo_id=hf_out, repo_type="model", commit_message=msg)
    imat.unlink(missing_ok=True); summ.unlink(missing_ok=True)
    shutil.rmtree(srcdir, ignore_errors=True)   # free the multi-GB source(s)
    kh.step("model.done", model=name)
    return name, report


def main():
    kh.init_progress()
    token = kh.resolve_hf_token()
    kh.step("harness_ready", n_models=len(RUN), hf_token_ok=bool(token))
    calib = read_corpus("calib_corpus.txt", _CALIB_FB)
    eval_ = read_corpus("eval_corpus.txt", _EVAL_FB)

    # build crispembed-cli + crispembed-quantize (CPU; GPU attached only for internet)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "huggingface_hub", "hf_transfer", "gguf"])
    repo = WORK / "CrispEmbed"
    if not repo.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(repo)])
        subprocess.check_call(["git", "-C", str(repo), "submodule", "update", "--init", "--recursive"])
    kh.install_build_toolchain()
    build = repo / "build"; build.mkdir(exist_ok=True)
    GPU = os.environ.get("CRISP_GPU", "0") != "0"
    flags = (kh.cuda_build_flags(kh.detect_cuda_arch()) if GPU else ["-DGGML_CUDA=OFF"]) + kh.cache_and_link_flags()
    kh.sh_with_progress(f"cmake -S {repo} -B {build} -DCMAKE_BUILD_TYPE=Release " + " ".join(flags))
    with kh.build_heartbeat("cmake.build"):
        kh.sh_with_progress(f"cmake --build {build} --target crispembed-cli crispembed-quantize "
                            f"-j{kh.safe_build_jobs(gpu=GPU)}")
    cli, quant = build / "crispembed", build / "crispembed-quantize"
    kh.step("built")

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    # Idempotent: skip models whose repo already has imatrix quants (unless FORCE=1),
    # so re-running the batch only processes newly-added models.
    force = os.environ.get("FORCE", "0") != "0"
    def is_done(hf_out):
        try:
            fs = api.list_repo_files(hf_out)
        except Exception:
            return False
        return (any(f.endswith("-iq4_xs.gguf") for f in fs)
                and any(f.endswith("-imatrix-ab.txt") for f in fs))

    import traceback
    results, failures, skipped = [], [], []
    for name in RUN:
        hf_out = OVERRIDES.get(name, {}).get("hf_out", f"cstr/{name}-GGUF")
        if not force and is_done(hf_out):
            print(f"[skip] {name} — already has imatrix quants", flush=True)
            kh.step("model.skip", model=name); skipped.append(name); continue
        try:
            results.append(process(name, cli, quant, api, calib, eval_))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {name}: {err}\n{traceback.format_exc()[-1500:]}", flush=True)
            kh.step("model.fail", model=name, error=err[:300])
            failures.append((name, err))

    # Downloadable batch summary (kernels stdout is not captured; kaggle_usage #15).
    lines = ["===== BATCH SUMMARY ====="]
    for name, rep in results:
        lines.append(f"\n## {name}")
        lines += ["  " + r for r in rep]
    if failures:
        lines.append("\nFAILED:")
        lines += [f"  {n}: {e}" for n, e in failures]
    text = "\n".join(lines) + "\n"
    (WORK / "batch_summary.txt").write_text(text)   # kept in working dir → downloadable
    print("\n" + text, flush=True)
    kh.step("all_done", ok=len(results), skipped=len(skipped), failed=len(failures),
            failures=",".join(n for n, _ in failures))
    print(f"[DONE] ok={len(results)} skipped={len(skipped)} failed={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
