#!/usr/bin/env python3
"""Controlled A/B — does bilingual (EN+DE) imatrix calibration improve German
quantization vs English-only calibration?

For a few multilingual models, quantize q4_k twice — once with an imatrix
calibrated on ENGLISH-only text, once on the SAME text + German — then evaluate
BOTH against the full-precision (q8) model on a GERMAN eval set (English for
contrast). Reports the German delta; UPLOADS NO model quants and overwrites
nothing. Only a report file is uploaded.
"""
import os, sys, subprocess, json, math
from pathlib import Path

WORK = Path("/kaggle/working")
STAGE = Path("/tmp/calib-ab"); STAGE.mkdir(parents=True, exist_ok=True)  # /tmp per kaggle_usage #18

CRISPASR = WORK / "CrispASR"
if not CRISPASR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
            "https://github.com/CrispStrobe/CrispASR.git", str(CRISPASR)])
        sys.path.insert(0, str(CRISPASR / "tools" / "kaggle"))
    except Exception:
        pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh
kh.init_progress()
token = kh.resolve_hf_token()
kh.step("harness_ready", hf_token_ok=bool(token))

# ── corpora: EN-only vs (EN + DE); eval split by language ────────────────────
EN_CALIB = [
    "Machine learning models turn raw text into dense numerical vectors.",
    "The central bank raised interest rates to slow down inflation.",
    "Photosynthesis lets green plants convert sunlight into chemical energy.",
    "Regular exercise strengthens the heart and improves overall mood.",
    "Renewable energy sources like wind and solar keep getting cheaper.",
    "The spacecraft entered orbit after a journey of several months.",
    "Distributed databases replicate records so that lookups stay fast.",
    "Ancient trade routes connected distant cities across several continents.",
    "Heavy rain and strong winds are expected across the region tomorrow.",
    "Learning a second language improves memory and opens new opportunities.",
]
DE_CALIB = [
    "Modelle des maschinellen Lernens wandeln Text in dichte Zahlenvektoren um.",
    "Die Zentralbank erhöhte die Zinsen, um die Inflation zu bremsen.",
    "Die Photosynthese wandelt Sonnenlicht in chemische Energie um.",
    "Regelmäßiger Sport stärkt das Herz und verbessert die Stimmung.",
    "Erneuerbare Energien wie Wind und Sonne werden immer günstiger.",
    "Die Raumsonde trat nach mehreren Monaten in die Umlaufbahn ein.",
    "Verteilte Datenbanken replizieren Datensätze für schnelle Abfragen.",
    "Alte Handelswege verbanden weit entfernte Städte über Kontinente hinweg.",
    "Für morgen werden in der Region starker Regen und Wind erwartet.",
    "Das Erlernen einer zweiten Sprache verbessert das Gedächtnis.",
]
EN_EVAL = [
    "Doctors recommend rest and fluids to recover from a mild fever.",
    "Cutting carbon emissions requires a shift toward cleaner transport.",
    "Small class sizes let teachers give students more attention.",
    "Migratory birds travel thousands of kilometres between seasons.",
    "The bridge was designed to withstand strong earthquakes.",
    "Deep ocean currents move heat slowly around the planet.",
]
DE_EVAL = [
    "Ärzte empfehlen Ruhe und Flüssigkeit bei leichtem Fieber.",
    "Weniger CO2-Ausstoß erfordert einen Wechsel zu saubererem Verkehr.",
    "Kleine Klassen ermöglichen mehr individuelle Betreuung der Schüler.",
    "Zugvögel legen zwischen den Jahreszeiten Tausende Kilometer zurück.",
    "Die Brücke wurde so gebaut, dass sie starken Erdbeben standhält.",
    "Tiefe Meeresströmungen transportieren Wärme um den ganzen Planeten.",
]
# reranker (query, [docs]) — docs 1&3 relevant, 2&4 off-topic
EN_RR_CALIB = [
    ("what causes rain", ["Rain forms when vapor condenses into droplets.", "Stocks fell today.", "Clouds release precipitation when saturated.", "Cats are common pets."]),
    ("how do vaccines work", ["Vaccines train the immune system with antigens.", "A cake needs eggs.", "Immunization prompts antibody production.", "The bus arrives at noon."]),
    ("benefits of exercise", ["Exercise strengthens the heart.", "The library has books.", "Activity improves mood.", "Saturn has rings."]),
    ("how neural networks learn", ["Networks adjust weights via backpropagation.", "Coffee has caffeine.", "Gradient descent minimizes loss.", "The Nile is long."]),
]
DE_RR_CALIB = [
    ("Wie entsteht Regen", ["Regen entsteht, wenn Wasserdampf kondensiert.", "Die Börse fiel heute.", "Wolken geben Niederschlag ab.", "Katzen sind Haustiere."]),
    ("Wie wirken Impfstoffe", ["Impfstoffe trainieren das Immunsystem.", "Ein Kuchen braucht Eier.", "Eine Impfung regt Antikörper an.", "Der Bus kommt um zwölf."]),
    ("Vorteile von Sport", ["Sport stärkt das Herz.", "Die Bibliothek hat Bücher.", "Bewegung hebt die Stimmung.", "Saturn hat Ringe."]),
    ("Wie lernen neuronale Netze", ["Netze passen Gewichte per Backpropagation an.", "Kaffee enthält Koffein.", "Gradientenabstieg minimiert den Verlust.", "Der Nil ist lang."]),
]
EN_RR_EVAL = [
    ("treatment for headaches", ["Pain relievers ease headaches.", "The bridge is long.", "Rest reduces headache severity.", "Tomatoes are fruits."]),
    ("causes of climate change", ["Greenhouse gases drive warming.", "The museum opens at nine.", "Deforestation raises CO2.", "Owls are nocturnal."]),
    ("how computers store data", ["Data is stored as bits.", "Roses are red.", "SSDs use flash memory.", "The concert is at eight."]),
]
DE_RR_EVAL = [
    ("Behandlung von Kopfschmerzen", ["Schmerzmittel lindern Kopfschmerzen.", "Die Brücke ist lang.", "Ruhe verringert die Beschwerden.", "Tomaten sind Früchte."]),
    ("Ursachen des Klimawandels", ["Treibhausgase treiben die Erwärmung.", "Das Museum öffnet um neun.", "Abholzung erhöht das CO2.", "Eulen sind nachtaktiv."]),
    ("Wie Computer Daten speichern", ["Daten werden als Bits gespeichert.", "Rosen sind rot.", "SSDs nutzen Flash-Speicher.", "Das Konzert ist um acht."]),
]
# NER — entity-rich
EN_NER_CALIB = [
    "Barack Obama met Angela Merkel in Berlin.",
    "Apple and Microsoft announced a deal in California.",
    "Marie Curie was born in Warsaw and worked in Paris.",
    "Google DeepMind is based in London.",
    "Amazon opened an office in Toronto.",
]
DE_NER_CALIB = [
    "Olaf Scholz traf Emmanuel Macron in Berlin.",
    "Siemens und Bosch eröffneten ein Werk in München.",
    "Angela Merkel wuchs in Hamburg auf und studierte in Leipzig.",
    "Volkswagen stellte ein Modell in Wolfsburg vor.",
    "Die Vereinten Nationen tagten in Genf.",
]
EN_NER_EVAL = [
    "Joe Biden spoke with Emmanuel Macron in Brussels.",
    "Samsung and Sony compete in Japan.",
    "Albert Einstein studied in Zurich and Princeton.",
]
DE_NER_EVAL = [
    "Ursula von der Leyen sprach in Brüssel mit der NATO.",
    "BMW und Audi eröffneten ein Zentrum in Ingolstadt.",
    "Johann Wolfgang von Goethe lebte in Weimar und Frankfurt.",
]
NER_LABELS = "person,organization,location"

MODELS = [("bge-m3", "embed"),
          ("jina-reranker-v2-base-multilingual", "rerank"),
          ("xlmr-ner-hrl", "ner")]

# ── metrics ──────────────────────────────────────────────────────────────────
def _cos(a, b):
    d = sum(x*y for x, y in zip(a, b)); na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return d/(na*nb) if na and nb else 0.0

def embed(cli, m, texts):
    r = subprocess.run([str(cli), "-m", str(m), "--json", *texts], capture_output=True, text=True, check=True)
    return [o["embedding"] for o in json.loads(r.stdout)]

def rr_scores(cli, m, q, docs):
    r = subprocess.run([str(cli), "-m", str(m), "--json", "--rerank", q, *docs], capture_output=True, text=True, check=True)
    return {e["index"]: e["score"] for e in json.loads(r.stdout)["results"]}

def kendall(a, b):
    idx = sorted(set(a) & set(b)); con = dis = 0
    for i in range(len(idx)):
        for j in range(i+1, len(idx)):
            sa, sb = a[idx[i]]-a[idx[j]], b[idx[i]]-b[idx[j]]
            if sa == 0 or sb == 0: continue
            con += (sa > 0) == (sb > 0); dis += (sa > 0) != (sb > 0)
    return (con-dis)/(con+dis) if (con+dis) else 1.0

def ner_ents(cli, m, t):
    r = subprocess.run([str(cli), "-m", str(m), "--json", "--ner", t, "--ner-labels", NER_LABELS], capture_output=True, text=True, check=True)
    return {(e["start"], e["end"], e["label"]) for e in json.loads(r.stdout).get("entities", [])}

def eval_vs_gold(cli, model, mode, eval_, gold):
    if mode == "embed":
        v = embed(cli, model, eval_)
        return sum(_cos(a, b) for a, b in zip(v, gold)) / len(v)
    if mode == "rerank":
        ts = [kendall(gold[i], rr_scores(cli, model, q, docs)) for i, (q, docs) in enumerate(eval_)]
        return sum(ts) / len(ts)
    if mode == "ner":
        tp = fp = fn = 0
        for (t, g) in zip(eval_, gold):
            p = ner_ents(cli, model, t); tp += len(g & p); fp += len(p - g); fn += len(g - p)
        prec = tp/(tp+fp) if tp+fp else 0.0; rec = tp/(tp+fn) if tp+fn else 0.0
        return 2*prec*rec/(prec+rec) if prec+rec else 0.0

def gold_of(cli, q8, mode, eval_):
    if mode == "embed":  return embed(cli, q8, eval_)
    if mode == "rerank": return [rr_scores(cli, q8, q, docs) for q, docs in eval_]
    if mode == "ner":    return [ner_ents(cli, q8, t) for t in eval_]

def calibrate(cli, csrc, mode, calib, imat):
    env = dict(os.environ, CRISPEMBED_IMATRIX_OUT=str(imat))
    if mode == "embed":
        subprocess.run([str(cli), "-m", str(csrc), "--json", *calib], env=env, capture_output=True, text=True, check=True)
    elif mode == "rerank":
        for q, docs in calib:
            subprocess.run([str(cli), "-m", str(csrc), "--json", "--rerank", q, *docs], env=env, capture_output=True, text=True, check=True)
    elif mode == "ner":
        for t in calib:
            subprocess.run([str(cli), "-m", str(csrc), "--json", "--ner", t, "--ner-labels", NER_LABELS], env=env, capture_output=True, text=True, check=True)
    if not imat.exists() or imat.stat().st_size == 0:
        raise RuntimeError(f"empty imatrix for {csrc}")

CORP = {
    "embed":  (EN_CALIB, DE_CALIB, EN_EVAL, DE_EVAL),
    "rerank": (EN_RR_CALIB, DE_RR_CALIB, EN_RR_EVAL, DE_RR_EVAL),
    "ner":    (EN_NER_CALIB, DE_NER_CALIB, EN_NER_EVAL, DE_NER_EVAL),
}

# ── build CrispEmbed @ main ──────────────────────────────────────────────────
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub", "hf_transfer", "gguf"])
REPO = WORK / "CrispEmbed"
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", "main", "https://github.com/CrispStrobe/CrispEmbed.git", str(REPO)])
    subprocess.check_call(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive"])
kh.install_build_toolchain()
BUILD = REPO / "build"; BUILD.mkdir(exist_ok=True)
kh.sh_with_progress(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF " + " ".join(kh.cache_and_link_flags()))
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"cmake --build {BUILD} --target crispembed-cli crispembed-quantize -j{kh.safe_build_jobs(gpu=False)}")
CLI = BUILD / "crispembed"; QUANT = BUILD / "crispembed-quantize"
kh.step("built")

from huggingface_hub import hf_hub_download, HfApi, list_repo_files
import re
api = HfApi(token=token)
_QRE = re.compile(r'(^|[-.])(q\d|iq\d|q4_k|q5_k|q6_k|q8_0|q4_0|bf16|imatrix)', re.I)

report = ["# imatrix calibration A/B — English-only vs bilingual(EN+DE), evaluated on German",
          "# metric: embed=cosine, rerank=Kendall-tau, ner=span-F1 (vs q8 gold). higher=better.",
          f"# {'model':40s} {'lang':4s} {'en-calib':>9s} {'ende-calib':>11s} {'delta':>8s}"]
for name, mode in MODELS:
    try:
        repo = f"cstr/{name}-GGUF"
        sib = {s.rfilename: (s.size or 0) for s in api.repo_info(repo, files_metadata=True).siblings
               if s.rfilename.endswith(".gguf")}
        nonq = {f: sz for f, sz in sib.items() if not _QRE.search(f) and "ref" not in f}
        base = f"{name}.gguf" if f"{name}.gguf" in nonq else max(nonq, key=nonq.get)
        # q8 gold: find the repo's *-q8_0.gguf directly (its prefix may differ from the
        # base's, e.g. base xlmr-ner-hrl-f32.gguf vs quant xlmr-ner-hrl-q8_0.gguf)
        q8n = next(f for f in sib if f.endswith("-q8_0.gguf"))
        d = STAGE / name; d.mkdir(exist_ok=True)
        with kh.build_heartbeat(f"{name}.download"):
            basep = Path(hf_hub_download(repo, base, token=token, local_dir=str(d)))
            q8p = Path(hf_hub_download(repo, q8n, token=token, local_dir=str(d)))
        en_c, de_c, en_e, de_e = CORP[mode]
        imat_en, imat_ende = d/"en.imatrix", d/"ende.imatrix"
        with kh.build_heartbeat(f"{name}.calibrate"):
            calibrate(CLI, basep, mode, en_c, imat_en)
            calibrate(CLI, basep, mode, en_c + de_c, imat_ende)
        q4_en, q4_ende = d/"q4_en.gguf", d/"q4_ende.gguf"
        subprocess.check_call([str(QUANT), str(basep), str(q4_en), "q4_k", "--imatrix", str(imat_en)])
        subprocess.check_call([str(QUANT), str(basep), str(q4_ende), "q4_k", "--imatrix", str(imat_ende)])
        for lang, ev in (("DE", de_e), ("EN", en_e)):
            gold = gold_of(CLI, q8p, mode, ev)
            m_en = eval_vs_gold(CLI, q4_en, mode, ev, gold)
            m_ende = eval_vs_gold(CLI, q4_ende, mode, ev, gold)
            report.append(f"  {name:40s} {lang:4s} {m_en:9.4f} {m_ende:11.4f} {m_ende-m_en:+8.4f}")
            kh.step(f"{name}.{lang}", en_calib=round(m_en, 5), ende_calib=round(m_ende, 5), delta=round(m_ende-m_en, 5))
        for p in (basep, q8p, q4_en, q4_ende):
            p.unlink(missing_ok=True)
    except Exception as e:
        import traceback
        report.append(f"  {name:40s} FAILED: {type(e).__name__}: {e}")
        kh.step(f"{name}.fail", err=str(e)[:200]); print(traceback.format_exc()[-1200:], flush=True)

text = "\n".join(report) + "\n"
(WORK / "calib_ab_report.txt").write_text(text)
print("\n" + text, flush=True)
if token:
    try:
        api.create_repo("cstr/crispembed-calib-ab", repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=str(WORK / "calib_ab_report.txt"), path_in_repo="calib_ab_report.txt",
                        repo_id="cstr/crispembed-calib-ab", repo_type="dataset",
                        commit_message="EN-only vs bilingual imatrix calibration, DE eval")
    except Exception as e:
        print("upload failed:", e, flush=True)
kh.step("all_done")
print("[DONE]", flush=True)
