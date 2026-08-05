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
    ("treatment for migraine headaches", [
        "Triptans are prescription drugs that abort acute migraine attacks.",
        "Over-the-counter ibuprofen can relieve mild headache pain.",
        "Staying hydrated and resting in a dark room eases migraine symptoms.",
        "Regular sleep schedules reduce the frequency of migraines.",
        "The suspension bridge spans nearly two kilometers across the bay.",
        "Ripe tomatoes are botanically classified as fruits."]),
    ("how do plants convert sunlight into energy", [
        "Photosynthesis converts sunlight, water and CO2 into glucose.",
        "Chlorophyll in the chloroplasts absorbs light energy.",
        "The light-dependent reactions occur in the thylakoid membranes.",
        "Plants also respire, consuming oxygen at night.",
        "The six-cylinder engine produces three hundred horsepower.",
        "A standard chessboard has sixty-four alternating squares."]),
    ("main causes of global climate change", [
        "Burning fossil fuels releases carbon dioxide that traps heat.",
        "Methane from agriculture is a potent greenhouse gas.",
        "Deforestation reduces the planet's capacity to absorb CO2.",
        "Industrial emissions have risen sharply since 1950.",
        "The art museum opens to visitors at nine each morning.",
        "Owls are nocturnal birds with excellent night vision."]),
    ("health benefits of regular aerobic exercise", [
        "Aerobic exercise strengthens the heart and improves circulation.",
        "Regular activity lowers blood pressure and resting heart rate.",
        "Exercise releases endorphins that improve mood and reduce stress.",
        "Walking thirty minutes a day supports long-term health.",
        "The public library houses over a million printed volumes.",
        "Saturn is encircled by a bright system of icy rings."]),
    ("how computers store digital data", [
        "Data is encoded as binary bits on storage media.",
        "Solid-state drives store data in flash memory cells.",
        "Hard disk drives write bits magnetically on spinning platters.",
        "RAM holds data temporarily while programs run.",
        "Red roses are a traditional symbol of romance.",
        "The orchestra concert is scheduled to begin at eight."]),
    ("what makes a good night of sleep", [
        "Deep sleep stages restore the body and consolidate memory.",
        "A cool, dark, quiet room promotes uninterrupted sleep.",
        "Avoiding caffeine late in the day improves sleep quality.",
        "Consistent bedtimes help regulate the circadian rhythm.",
        "The quarterly earnings report exceeded analyst expectations.",
        "Basalt is a common volcanic rock formed from lava."]),
    ("how vaccines protect against disease", [
        "Vaccines train the immune system to recognize a pathogen.",
        "They introduce a harmless antigen that triggers antibodies.",
        "Immune memory cells enable a fast response to real infection.",
        "Booster doses maintain protection as immunity wanes.",
        "The ferry departs from the northern pier every hour.",
        "Maple syrup is harvested from the sap of maple trees."]),
    ("why is the sky blue during the day", [
        "Sunlight scatters off air molecules, and blue scatters most.",
        "Rayleigh scattering favors shorter blue wavelengths.",
        "At sunset the light travels farther, so reds dominate.",
        "The atmosphere's composition shapes the scattering effect.",
        "The championship match went into extra time on Sunday.",
        "Sourdough bread rises through natural wild-yeast fermentation."]),
    ("Wie funktioniert eine Solarzelle", [
        "Eine Solarzelle wandelt Sonnenlicht direkt in elektrischen Strom um.",
        "Der photoelektrische Effekt löst Elektronen im Halbleiter aus.",
        "Silizium ist das häufigste Material für Photovoltaikzellen.",
        "Mehrere Zellen werden zu einem Solarmodul zusammengeschaltet.",
        "Der Zug nach München fährt pünktlich um acht Uhr ab.",
        "Die Katze schläft den ganzen Nachmittag auf dem Sofa."]),
    ("Symptome und Verlauf einer Erkältung", [
        "Eine Erkältung beginnt oft mit Halskratzen und Schnupfen.",
        "Husten und leichtes Fieber können in den ersten Tagen auftreten.",
        "Ausreichend Ruhe und Flüssigkeit lindern die Beschwerden.",
        "Die meisten Erkältungen klingen nach einer Woche ab.",
        "Der Fluss schlängelt sich zwei Kilometer durch das Tal.",
        "Tomaten gelten botanisch gesehen als Früchte."]),
    ("Hauptursachen des globalen Klimawandels", [
        "Die Verbrennung fossiler Brennstoffe setzt viel CO2 frei.",
        "Methan aus der Landwirtschaft verstärkt den Treibhauseffekt.",
        "Abholzung verringert die Aufnahme von Kohlendioxid.",
        "Industrieabgase sind seit 1950 stark angestiegen.",
        "Das Museum öffnet werktags erst um neun Uhr.",
        "Eulen sind nachtaktive Vögel mit sehr gutem Gehör."]),
    ("Vorteile von regelmäßigem Ausdauersport", [
        "Ausdauersport stärkt das Herz und die Blutgefäße.",
        "Regelmäßige Bewegung senkt den Ruhepuls und den Blutdruck.",
        "Sport schüttet Endorphine aus und hebt die Stimmung.",
        "Schon dreißig Minuten Gehen pro Tag fördern die Gesundheit.",
        "Die Stadtbibliothek besitzt über eine Million Bücher.",
        "Der Saturn ist von einem hellen Ringsystem umgeben."]),
    ("Wie speichern Computer digitale Daten", [
        "Daten werden als binäre Bits auf Datenträgern gespeichert.",
        "SSDs speichern Informationen in Flash-Speicherzellen.",
        "Festplatten schreiben Bits magnetisch auf rotierende Scheiben.",
        "Der Arbeitsspeicher hält Daten nur vorübergehend.",
        "Rote Rosen gelten als Symbol der Romantik.",
        "Das Konzert des Orchesters beginnt um acht Uhr abends."]),
    ("Was sorgt für erholsamen Schlaf", [
        "Tiefschlafphasen erholen den Körper und festigen Erinnerungen.",
        "Ein kühles, dunkles und ruhiges Zimmer fördert den Schlaf.",
        "Koffein am Abend zu meiden verbessert die Schlafqualität.",
        "Feste Schlafenszeiten stabilisieren den Biorhythmus.",
        "Der Quartalsbericht übertraf die Erwartungen der Analysten.",
        "Basalt ist ein häufiges vulkanisches Gestein aus Lava."]),
    ("Wie schützen Impfungen vor Krankheiten", [
        "Impfungen trainieren das Immunsystem, einen Erreger zu erkennen.",
        "Ein harmloses Antigen regt die Bildung von Antikörpern an.",
        "Gedächtniszellen ermöglichen eine schnelle spätere Abwehr.",
        "Auffrischungsimpfungen erhalten den Schutz über die Zeit.",
        "Die Fähre legt jede Stunde am Nordpier ab.",
        "Ahornsirup wird aus dem Saft von Ahornbäumen gewonnen."]),
    ("Warum ist der Himmel tagsüber blau", [
        "Sonnenlicht wird an Luftmolekülen gestreut, Blau am stärksten.",
        "Die Rayleigh-Streuung bevorzugt kurze blaue Wellenlängen.",
        "Bei Sonnenuntergang dominieren wegen des langen Wegs die Rottöne.",
        "Die Zusammensetzung der Atmosphäre prägt die Streuung.",
        "Das Endspiel ging am Sonntag in die Verlängerung.",
        "Sauerteigbrot geht durch wilde Hefegärung auf."]),
    # ── expansion 2026-07-12: +7 EN / +7 DE distinct topics (self-authored CC0) ──
    ("how antibiotics fight bacterial infections", [
        "Antibiotics kill bacteria or stop them from multiplying.",
        "Penicillin disrupts the bacterial cell wall until the cell bursts.",
        "Broad-spectrum antibiotics act against many bacterial species.",
        "Finishing the full course helps prevent resistant strains.",
        "The lighthouse beam sweeps the harbor every ten seconds.",
        "Marble is a metamorphic rock prized by sculptors."]),
    ("what causes ocean tides", [
        "The Moon's gravity pulls ocean water into tidal bulges.",
        "High and low tides alternate roughly every six hours.",
        "The Sun's gravity adds to spring and neap tide variation.",
        "Coastlines and sea depth shape local tide heights.",
        "The violinist tuned her strings before the recital.",
        "Cacti store water in their thick fleshy stems."]),
    ("how a four-stroke engine works", [
        "A four-stroke engine cycles intake, compression, power and exhaust.",
        "The spark plug ignites the compressed fuel-air mixture.",
        "The piston's motion turns the crankshaft to drive the wheels.",
        "Valves open and close to admit air and expel exhaust gases.",
        "The bakery sells fresh croissants every morning.",
        "Emperor penguins huddle together to survive the Antarctic cold."]),
    ("benefits of a balanced diet", [
        "A balanced diet supplies the nutrients the body needs to function.",
        "Fruits and vegetables provide vitamins, fiber and antioxidants.",
        "Adequate protein supports muscle repair and immune function.",
        "Limiting added sugar lowers the risk of metabolic disease.",
        "The comet will next be visible from Earth in seventy years.",
        "Gothic cathedrals feature pointed arches and flying buttresses."]),
    ("how earthquakes happen", [
        "Earthquakes occur when tectonic plates slip along a fault.",
        "Stress builds until the rock suddenly ruptures, releasing energy.",
        "Seismic waves radiate outward and shake the ground.",
        "The Richter scale measures an earthquake's released energy.",
        "The chef garnished the plate with a sprig of basil.",
        "Honeybees communicate food locations with a waggle dance."]),
    ("why we dream during sleep", [
        "The most vivid dreams occur during REM sleep.",
        "During REM the brain is highly active while the body stays still.",
        "Dreams may help process emotions and consolidate memory.",
        "Sleep cycles alternate between REM and non-REM stages.",
        "The train departs from platform four at noon.",
        "Quartz is one of the most abundant minerals in the crust."]),
    ("how a rainbow forms", [
        "A rainbow forms when sunlight refracts and reflects inside raindrops.",
        "Each droplet splits white light into its component colors.",
        "Red appears on the outer arc and violet on the inner arc.",
        "Rainbows appear opposite the Sun, low after a shower.",
        "The stock exchange closed higher on Friday afternoon.",
        "Camels can go many days without drinking water."]),
    ("Wie wirken Antibiotika gegen Bakterien", [
        "Antibiotika töten Bakterien ab oder hemmen ihre Vermehrung.",
        "Penicillin zerstört die Zellwand der Bakterien.",
        "Breitbandantibiotika wirken gegen viele Bakterienarten.",
        "Die vollständige Einnahme beugt resistenten Stämmen vor.",
        "Der Leuchtturm bestreicht den Hafen alle zehn Sekunden.",
        "Marmor ist ein bei Bildhauern beliebtes Gestein."]),
    ("Was verursacht Ebbe und Flut", [
        "Die Anziehungskraft des Mondes formt die Gezeitenberge.",
        "Flut und Ebbe wechseln etwa alle sechs Stunden.",
        "Auch die Sonne verstärkt Spring- und Nipptiden.",
        "Küstenform und Wassertiefe bestimmen die örtliche Tidenhöhe.",
        "Die Geigerin stimmte vor dem Konzert ihre Saiten.",
        "Kakteen speichern Wasser in ihren dicken Stämmen."]),
    ("Wie funktioniert ein Viertaktmotor", [
        "Ein Viertaktmotor durchläuft Ansaugen, Verdichten, Arbeiten und Ausstoßen.",
        "Die Zündkerze entzündet das verdichtete Kraftstoffgemisch.",
        "Die Kolbenbewegung treibt über die Kurbelwelle die Räder an.",
        "Ventile lassen Luft ein und leiten die Abgase ab.",
        "Die Bäckerei verkauft jeden Morgen frische Croissants.",
        "Kaiserpinguine drängen sich gegen die Kälte zusammen."]),
    ("Vorteile einer ausgewogenen Ernährung", [
        "Eine ausgewogene Ernährung liefert alle nötigen Nährstoffe.",
        "Obst und Gemüse liefern Vitamine, Ballaststoffe und Antioxidantien.",
        "Ausreichend Eiweiß unterstützt Muskeln und Immunsystem.",
        "Weniger Zucker senkt das Risiko für Stoffwechselerkrankungen.",
        "Der Komet ist erst in siebzig Jahren wieder sichtbar.",
        "Gotische Kathedralen haben Spitzbögen und Strebepfeiler."]),
    ("Wie entstehen Erdbeben", [
        "Erdbeben entstehen, wenn tektonische Platten an einer Verwerfung abrutschen.",
        "Spannung baut sich auf, bis das Gestein plötzlich bricht.",
        "Seismische Wellen breiten sich aus und erschüttern den Boden.",
        "Die Richterskala misst die freigesetzte Energie eines Bebens.",
        "Der Koch garnierte den Teller mit einem Basilikumzweig.",
        "Honigbienen zeigen Futterquellen mit einem Schwänzeltanz."]),
    ("Warum träumen wir im Schlaf", [
        "Die lebhaftesten Träume treten im REM-Schlaf auf.",
        "Im REM-Schlaf ist das Gehirn aktiv, während der Körper ruht.",
        "Träume helfen, Gefühle zu verarbeiten und Gelerntes zu festigen.",
        "Schlafzyklen wechseln zwischen REM und Non-REM.",
        "Der Zug fährt mittags von Gleis vier ab.",
        "Quarz zählt zu den häufigsten Mineralen der Erdkruste."]),
    ("Wie entsteht ein Regenbogen", [
        "Ein Regenbogen entsteht durch Brechung und Reflexion in Regentropfen.",
        "Jeder Tropfen zerlegt weißes Licht in seine Farben.",
        "Rot liegt außen, Violett innen am Bogen.",
        "Regenbögen erscheinen der Sonne gegenüber nach einem Schauer.",
        "Die Börse schloss am Freitag höher.",
        "Kamele kommen viele Tage ohne Wasser aus."]),
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

# q8_0 is normally only an A/B reference, but for the F2LLM-v2 family it is the
# SHIPPED default and 0.6B's is known-soft (T19-E1: cos 0.9909 with +3.8% norm
# inflation) — so that family also gets a q8_0+imatrix arm.
QSPECS_Q8IM = [
    ("q8_0",   False, None),
    ("q8_0",   True,  None),
    ("q4_k",   False, None),
    ("q4_k",   True,  "{prefix}-q4_k-imatrix.gguf"),
    ("iq4_xs", True,  "{prefix}-iq4_xs.gguf"),
]
for _m in ("f2llm-v2-80m", "f2llm-v2-160m", "f2llm-v2-330m"):
    OVERRIDES[_m] = {"quants": QSPECS_Q8IM}

# f2llm-v2-0.6b ALREADY ships -q4_k-imatrix.gguf / -iq4_xs.gguf from the earlier
# batch — and `examples/cli/model_hashes.h` PINS their SHA256. Overwriting them
# would break the pin for every existing user, so this re-calibration uploads
# under "-c2" names ("c2" = second-generation calibration corpus: German + the
# model's own query prompt + newline-bearing text, vs the 10 English one-liners
# the first run silently fell back to). Promotion is the coordinator's call.
OVERRIDES["f2llm-v2-0.6b"] = {
    "meta_prefix": "f2llm-v2-0.6b-c2",
    "quants": [
        ("q8_0",   False, None),
        ("q8_0",   True,  "{prefix}-q8_0-imatrix.gguf"),
        ("q4_k",   False, None),
        ("q4_k",   True,  "{prefix}-q4_k-imatrix-c2.gguf"),
        ("iq4_xs", True,  "{prefix}-iq4_xs-c2.gguf"),
    ],
}


# Corpus loading. A Kaggle *script* kernel ships ONLY its code_file — bundled
# sibling files are NOT readable at runtime (kaggle_usage #26/#19), so the old
# `Path(__file__).parent / "calib_corpus.txt"` lookup ALWAYS missed on Kaggle and
# every imatrix quant shipped so far was silently calibrated on the 10-sentence
# English `_CALIB_FB` fallback (visible as "calib=10" in the uploaded A/B
# summaries). Corpora are therefore read from the CLONED repo and a miss is FATAL.
CORPUS_DIR = None      # set by main() to <clone>/tools/kaggle/crispembed-imatrix-quant

def load_corpus(stem):
    """Load `<stem>.jsonl` (rows {"role","text"}) from the cloned repo; fall back
    to the legacy line-per-text `<stem>.txt` (role "doc"). Raises if neither
    exists — a silent fallback is what produced the mis-calibrated batch."""
    if CORPUS_DIR is None:
        raise RuntimeError("CORPUS_DIR not set — call after the repo clone")
    j = CORPUS_DIR / f"{stem}.jsonl"
    if j.exists():
        rows = [json.loads(l) for l in j.read_text(encoding="utf-8").splitlines() if l.strip()]
        return [(r.get("role", "doc"), r["text"]) for r in rows if r.get("text")]
    t = CORPUS_DIR / f"{stem}.txt"
    if t.exists():
        return [("doc", l.strip()) for l in t.read_text(encoding="utf-8").splitlines() if l.strip()]
    raise RuntimeError(f"no corpus {stem}.jsonl/.txt under {CORPUS_DIR} — refusing "
                       f"to calibrate on a fallback (that is the bug this fixes)")


def split_roles(rows):
    """(queries, docs). Query rows are embedded through the model's OWN query
    prompt (the CLI auto-prefix: F2LLM's instruction prompt, arctic-v2's
    "query: "); everything else is embedded with the prefix explicitly OFF,
    because documents take no prefix in every family we ship."""
    q = [t for role, t in rows if role.startswith("query")]
    d = [t for role, t in rows if not role.startswith("query")]
    return q, d


def embed(cli, model, texts, as_query=False):
    """Embed `texts`. as_query=False forces `--prefix ""` (auto-prefix OFF);
    as_query=True leaves the CLI's model-derived query prefix in place."""
    if not texts:
        return [], 0.0
    args = [str(cli), "-m", str(model), "--json"]
    if not as_query:
        args += ["--prefix", ""]
    t0 = time.time()
    r = subprocess.run(args + list(texts), capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"embed failed for {model}:\n{r.stderr[-1500:]}")
    data = json.loads(r.stdout)
    return [[float(x) for x in o["embedding"]] for o in data if o.get("embedding")], dt


def _norm(u):
    return math.sqrt(sum(x * x for x in u))


def _cos(u, v):
    d = sum(x * y for x, y in zip(u, v))
    nu, nv = _norm(u), _norm(v)
    return d / (nu * nv) if nu and nv else 0.0


def cos_stats(a, b):
    """CONTINUOUS metrics, per HARD RULE #2b + the imatrix lesson that a
    thresholded/mean-only score cannot see quant quality: per-text cosine
    min/mean/median AND the |quant|/|gold| norm-ratio distribution (cosine is
    scale-blind — the f2llm-0.6b q8_0 +3.8% norm inflation is invisible to it)."""
    n = min(len(a), len(b))
    if not n:
        return {"n": 0, "cos_min": float("nan"), "cos_mean": float("nan"),
                "cos_med": float("nan"), "nr_mean": float("nan"),
                "nr_min": float("nan"), "nr_max": float("nan")}
    cs = sorted(_cos(a[i], b[i]) for i in range(n))
    nrs = sorted((_norm(a[i]) / _norm(b[i])) if _norm(b[i]) else float("nan")
                 for i in range(n))
    med = cs[n // 2] if n % 2 else 0.5 * (cs[n // 2 - 1] + cs[n // 2])
    return {"n": n, "cos_min": cs[0], "cos_mean": sum(cs) / n, "cos_med": med,
            "nr_mean": sum(nrs) / n, "nr_min": nrs[0], "nr_max": nrs[-1]}


def mean_cos(a, b):
    s = cos_stats(a, b)
    return s["cos_mean"], s["n"]


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


def imatrix_coverage(path):
    """Digest of a collected .imatrix: per-shape tensor-key counts. `leaf_N` keys
    are ggml auto-names for an UNNAMED graph leaf — the F7 defect signature (the
    pre-merged BERT QKV weight), and they match nothing at quantize time."""
    try:
        from gguf import GGUFReader
        import collections
        names = [t.name for t in GGUFReader(str(path)).tensors]
        c = collections.Counter(re.sub(r"\d+", "N", n) for n in names)
        leaf = sum(v for k, v in c.items() if k.startswith("leaf_"))
        head = f"    {len(names)} keys, leaf_N={leaf} ({'DEFECT' if leaf else 'OK — no unnamed leaves'})"
        return "\n".join([head] + [f"    {v:4d}  {k}" for k, v in sorted(c.items())])
    except Exception as e:                                  # never fail a run on a digest
        return f"    (coverage digest unavailable: {type(e).__name__}: {e})"


def process(name, cli, quant, api, calib, eval_):
    from huggingface_hub import hf_hub_download
    ov = OVERRIDES.get(name, {})
    hf_out = ov.get("hf_out", f"cstr/{name}-GGUF")
    quants = ov.get("quants", QSPECS)
    kh.step("model.start", model=name, repo=hf_out)

    ggs = {s.rfilename: (s.size or 0) for s in api.repo_info(hf_out, files_metadata=True).siblings
           if s.rfilename.endswith(".gguf")}
    base_fn, prefix = pick_base_gguf(ggs, name)
    # `base_file` pins the full-precision source explicitly. pick_base_gguf prefers
    # the exact `<name>.gguf`, which is WRONG whenever a repo carries a corrected
    # re-conversion beside the original (ms-marco: `<name>-g7c.gguf` is the artifact
    # with the BertPooler stage; `<name>.gguf` is the superseded 1-layer-head file).
    # `prefix` then keeps the intermediate/upload stems on the canonical model name
    # instead of inheriting the correction suffix.
    if ov.get("base_file"):
        base_fn = ov["base_file"]
        if base_fn not in ggs:
            raise RuntimeError(f"base_file {base_fn} not in {hf_out} (have {sorted(ggs)})")
        prefix = ov.get("prefix", base_fn[:-5])
    elif ov.get("prefix"):
        prefix = ov["prefix"]
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
    # Corpora arrive as (role, text); the non-embed modes want plain text lists.
    calib_q, calib_d = split_roles(calib)
    eval_q, eval_d = split_roles(eval_)
    calib_txt = [t for _, t in calib]
    eval_txt = [t for _, t in eval_]

    def embed_eval(model):
        """Docs (prefix OFF) then queries (model's own query prompt) — one fixed
        order so every arm's vectors line up index-for-index."""
        return embed(cli, model, eval_d, as_query=False)[0] + \
               embed(cli, model, eval_q, as_query=True)[0]
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
            for t in calib_txt:
                cal = subprocess.run([str(cli), "-m", str(csrc), "--json", "--sparse", t],
                                     env=env, capture_output=True, text=True)
                if cal.returncode != 0:
                    raise RuntimeError(f"sparse calibration rc={cal.returncode} for {name}; "
                                       f"stderr tail:\n{cal.stderr[-1200:]}")
                outlen += len(cal.stdout); err = cal.stderr
        else:
            # TWO passes so the collected activations cover both halves of a real
            # retrieval workload: documents with the prefix OFF, queries through
            # the model's own query prompt. Both write to the same
            # CRISPEMBED_IMATRIX_OUT — the collector merges an existing file on
            # flush, so statistics accumulate across invocations.
            outlen, err = 0, ""
            for texts, as_q in ((calib_d, False), (calib_q, True)):
                if not texts:
                    continue
                args = [str(cli), "-m", str(csrc), "--json"]
                if not as_q:
                    args += ["--prefix", ""]
                cal = subprocess.run(args + texts, env=env, capture_output=True, text=True)
                if cal.returncode != 0:
                    raise RuntimeError(f"calibration (query={as_q}) rc={cal.returncode} for "
                                       f"{name}; stderr tail:\n{cal.stderr[-1200:]}")
                outlen += len(cal.stdout); err = cal.stderr
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
        gold = [(t, sparse_vec(cli, csrc, t)) for t in eval_txt]
    else:
        gold = embed_eval(csrc)

    report = []
    scale = []
    if mode == "rerank":
        scale.append(f'  RAW SCORES, query "{RERANK_EVAL[0][0]}" x {len(RERANK_EVAL[0][1])} docs')
        scale.append("  " + f"{goldlabel:9s} " + " ".join(
            f"[{i}]{s:+8.3f}" for i, s in sorted(gold[0].items())))
    for qtype, use_im, up_tmpl in quants:
        if big and qtype == "q8_0" and not use_im:
            continue  # q8_0 IS the gold for big models — no A/B needed
        tag = f"{qtype}{'-im' if use_im else ''}"
        stats = {}
        out = stage / f"{prefix}-{tag}.gguf"
        cmd = [str(quant), str(qsrc), str(out), qtype] + (["--imatrix", str(imat)] if use_im else [])
        with kh.build_heartbeat(f"{name}.quant.{tag}"):
            qp = subprocess.run(cmd, capture_output=True, text=True)
        # Judge coverage by the QUANTIZER'S OWN stdout, never by the exit code: a
        # name mismatch between the collected imatrix and the artifact's tensors
        # (crisp-mode vs ollama-mode conversion, or the pre-F7 `leaf_N` defect)
        # exits 0 and silently ships an "-imatrix" file with NO importance.
        qout = (qp.stdout or "") + (qp.stderr or "")
        print(qout, flush=True)
        if qp.returncode != 0:
            raise RuntimeError(f"quantize {tag} rc={qp.returncode} for {name}:\n{qout[-2000:]}")
        m_cov = re.search(r"(\d+) quantized, (\d+) kept(?:, (\d+) with imatrix)?", qout)
        m_load = re.search(r"imatrix: loaded importance vectors for (\d+) tensors", qout)
        cov_line = m_cov.group(0) if m_cov else "(no quantizer coverage line!)"
        n_cov = int(m_cov.group(3)) if (m_cov and m_cov.group(3)) else 0
        n_load = int(m_load.group(1)) if m_load else 0
        if use_im and n_cov == 0:
            raise RuntimeError(
                f"{name} {tag}: quantizer reported '{cov_line}' (imatrix file loaded "
                f"{n_load} vectors) — ZERO tensors took importance; refusing to upload "
                f"a mislabeled -imatrix artifact")
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
            st = cos_stats(embed_eval(out), gold)
            val = st["cos_mean"]
            # min + median + norm-ratio alongside the mean: a mean-only score
            # hides both the worst text and any uniform magnitude drift.
            extra = (f" min={st['cos_min']:.6f} med={st['cos_med']:.6f}"
                     f" normratio={st['nr_mean']:.4f} [{st['nr_min']:.4f},{st['nr_max']:.4f}]")
            stats = st
        mb = out.stat().st_size / 1e6
        upname = up_tmpl.format(prefix=prefix) if up_tmpl else "(A/B only)"
        kh.step(f"{name}.ab.{tag}", imatrix=use_im, **{f"{metric}_vs_gold": round(val, 6)},
                gold=goldlabel, size_mb=round(mb, 1), upload=upname, quant_coverage=cov_line,
                imatrix_vectors_loaded=n_load, n_with_imatrix=n_cov,
                **({k: round(v, 6) for k, v in stats.items()} if mode == "embed" else {}))
        report.append(f"{qtype:7s} imatrix={int(use_im)}  {metric}_vs_{goldlabel}={val:.6f}{extra}  "
                      f"{mb:7.1f}MB  [{cov_line}]  -> {upname}")
        if mode == "rerank":
            # RAW decoded scores for one eval pair, per arm. A reranker's absolute
            # scale is the thing tau/dscore cannot see (G7c: a structurally broken
            # head still ranked, but scored +-0.2 instead of +-11), so the summary
            # carries the actual logits alongside the gold's.
            scale.append(f"  {tag:9s} " + " ".join(
                f"[{i}]{s:+8.3f}" for i, s in sorted(rerank_scores(cli, out, *RERANK_EVAL[0]).items())))
        if up_tmpl:
            with kh.build_heartbeat(f"{name}.upload.{tag}"):
                api.upload_file(path_or_fileobj=str(out), path_in_repo=upname,
                    repo_id=hf_out, repo_type="model",
                    commit_message=f"{qtype} +imatrix ({metric}_vs_{goldlabel}={val:.4f})")
        out.unlink(missing_ok=True)

    calib_n = {"rerank": len(RERANK_CALIB), "ner": len(NER_CALIB), "colbert": len(COLBERT_CALIB)}.get(mode, len(calib))
    eval_n = {"rerank": len(RERANK_EVAL), "ner": len(NER_EVAL), "colbert": len(COLBERT_EVAL)}.get(mode, len(eval_))
    mix = (f" (calib {len(calib_d)} doc + {len(calib_q)} query-prompted; "
           f"eval {len(eval_d)} doc + {len(eval_q)} query-prompted)") if mode == "embed" else ""
    summary = (f"imatrix A/B — {name} ({hf_out}), {metric} vs {goldlabel} gold, "
               f"n={eval_n}, calib={calib_n}{mix}, quant_src={base_fn}\n" + "\n".join(report) + "\n")
    summary += "\nimatrix tensor coverage (collector keys; any `leaf_N` = uncovered matmul):\n"
    summary += imatrix_coverage(imat) + "\n"
    if scale:
        summary += "\n" + "\n".join(scale) + "\n"
    # meta_prefix lets a re-calibration publish its .imatrix / A/B summary beside
    # (not on top of) an earlier run's — the GGUF SHAs are pinned in model_hashes.h.
    mprefix = ov.get("meta_prefix", prefix)
    summ = stage / f"{mprefix}-imatrix-ab.txt"; summ.write_text(summary)
    if mprefix != prefix:
        imat2 = stage / f"{mprefix}.imatrix"; imat.rename(imat2); imat = imat2
    for p, msg in [(summ, f"A/B summary ({metric} vs gold)"), (imat, "importance matrix (calibration)")]:
        with kh.build_heartbeat(f"{name}.upload.meta"):
            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                repo_id=hf_out, repo_type="model", commit_message=msg)
    imat.unlink(missing_ok=True); summ.unlink(missing_ok=True)
    shutil.rmtree(srcdir, ignore_errors=True)   # free the multi-GB source(s)
    kh.step("model.done", model=name)
    return name, report


def main():
    global CORPUS_DIR
    kh.init_progress()
    token = kh.resolve_hf_token(require=True)  # upload-bearing: fail fast before any compute (F9b)
    kh.step("harness_ready", n_models=len(RUN), hf_token_ok=bool(token))

    # build crispembed-cli + crispembed-quantize (CPU; GPU attached only for internet)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "huggingface_hub", "hf_transfer", "gguf"])
    repo = WORK / "CrispEmbed"
    if not repo.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(repo)])
        subprocess.check_call(["git", "-C", str(repo), "submodule", "update", "--init", "--recursive"])

    # Corpora come from the CLONE, never from the push dir (kaggle_usage #26).
    CORPUS_DIR = repo / "tools" / "kaggle" / "crispembed-imatrix-quant"
    calib = load_corpus("calib_corpus")
    eval_ = load_corpus("eval_corpus")
    cq, cd = split_roles(calib)
    kh.step("corpora", calib=len(calib), eval=len(eval_), calib_query=len(cq), calib_doc=len(cd),
            newline_bearing=sum("\n" in t for _, t in calib), src=str(CORPUS_DIR))
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
