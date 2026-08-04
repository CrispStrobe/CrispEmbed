#!/usr/bin/env python3
"""Generate the imatrix calibration + evaluation corpora (JSONL).

Why this exists
---------------
imatrix quality scales with how well the calibration activations resemble the
production ones. The retrieval embedders this repo ships are used mostly on
**German** and English text (see PLAN.md T19), each family applies its own
**query prompt prefix**, and the T19-E1 tokenizer bug proved that
**newline-bearing** text exercises a genuinely different code path than the
newline-free one-liners the old 30-line `calib_corpus.txt` contained.

So the corpus deliberately mixes six roles/domains:

  doc.de      German retrieval-style passages (the T19 quality focus)
  doc.en      English retrieval-style passages
  query.de    short German questions — emitted through the model's OWN query
              prompt (the CLI auto-prefix; F2LLM's is the instruction prompt
              "Instruct: ...\\nQuery: ", arctic-v2's is "query: ")
  query.en    short English questions, same treatment
  code        multi-line source code (newlines, tabs, punctuation runs)
  struct      newline-heavy structured prose: markdown, lists, tables, logs

`role` drives HOW the text is embedded during calibration, not just what:
`query.*` rows run with the CLI's auto query prefix ON, every other role runs
with `--prefix ""` (documents take no prefix in every family we ship). That
makes the collected activations cover both halves of a real retrieval workload.

Calib/eval split is deterministic and DISJOINT (every 3rd item of each category
goes to eval), so the A/B is never measured on the texts that were calibrated on.

Usage:  python make_corpora.py          # rewrites calib_corpus.jsonl + eval_corpus.jsonl
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── German retrieval passages ────────────────────────────────────────────────
DOC_DE = [
    "Die Europäische Zentralbank hat den Leitzins erneut angehoben, um die hartnäckige Inflation im Euroraum zu dämpfen.",
    "Das Bundesverfassungsgericht in Karlsruhe prüft, ob das Gesetz mit dem Grundgesetz vereinbar ist.",
    "Bei der Photosynthese wandeln grüne Pflanzen Sonnenlicht, Wasser und Kohlendioxid in Traubenzucker und Sauerstoff um.",
    "Die Deutsche Bahn hat angekündigt, das Streckennetz in den kommenden Jahren umfassend zu sanieren.",
    "Eine ausgewogene Ernährung mit Gemüse, Hülsenfrüchten und Vollkornprodukten senkt das Risiko für Herz-Kreislauf-Erkrankungen.",
    "Der Betriebsrat muss bei Kündigungen angehört werden; unterbleibt die Anhörung, ist die Kündigung unwirksam.",
    "Die Alpen erstrecken sich über acht Staaten und bilden die höchste Gebirgskette Mitteleuropas.",
    "Künstliche neuronale Netze lernen, indem sie ihre Gewichte über den Fehlerrückfluss schrittweise anpassen.",
    "Im Mittelalter verbanden Handelswege wie die Via Regia entfernte Städte über mehrere Landesgrenzen hinweg.",
    "Die Mietpreisbremse begrenzt, wie stark die Miete bei einer Neuvermietung angehoben werden darf.",
    "Regelmäßiger Ausdauersport stärkt das Herz, senkt den Ruhepuls und verbessert die allgemeine Stimmung.",
    "Erneuerbare Energiequellen wie Wind- und Sonnenkraft werden pro Kilowattstunde stetig günstiger.",
    "Der Wasserstoff wird durch Elektrolyse gewonnen und kann in der Stahlindustrie Kokskohle ersetzen.",
    "Die Anmeldung eines Gewerbes erfolgt beim Gewerbeamt der Gemeinde, in der der Betrieb seinen Sitz hat.",
    "Ein Sparplan auf einen breit gestreuten Indexfonds gilt als kostengünstiger Einstieg in den Kapitalmarkt.",
    "Die Bundesnetzagentur überwacht den Wettbewerb in den Bereichen Strom, Gas, Telekommunikation und Post.",
    "Kinder haben ab dem vollendeten ersten Lebensjahr einen Rechtsanspruch auf einen Betreuungsplatz.",
    "Der Nationalpark Wattenmeer gehört seit 2009 zum Weltnaturerbe der UNESCO.",
    "Bei einem Wohnungsbrand sollte man die Wohnungstür schließen, das Haus verlassen und die 112 wählen.",
    "Die Promotion setzt in der Regel ein abgeschlossenes wissenschaftliches Hochschulstudium voraus.",
    "Der Datenschutzbeauftragte überwacht die Einhaltung der Datenschutz-Grundverordnung im Unternehmen.",
    "Weizenmehl, Wasser, Salz und ein Sauerteigansatz genügen, um ein kräftiges Bauernbrot zu backen.",
    "Die Schwarzwaldbahn überwindet auf ihrer Strecke einen Höhenunterschied von fast sechshundert Metern.",
    "Eine Photovoltaikanlage auf dem Dach eines Einfamilienhauses amortisiert sich meist nach zehn bis zwölf Jahren.",
    "Antibiotika wirken ausschließlich gegen Bakterien und sind bei viralen Infekten wirkungslos.",
    "Der Deutsche Bundestag wird für vier Jahre gewählt; die Abgeordneten sind nur ihrem Gewissen unterworfen.",
    "Die Elbe entspringt im Riesengebirge und mündet bei Cuxhaven in die Nordsee.",
    "Ein Testament kann handschriftlich verfasst oder notariell beurkundet werden.",
    "Beim Fasten sinkt der Insulinspiegel, und der Körper greift verstärkt auf seine Fettreserven zurück.",
    "Die Hanse war ein Bund von Kaufleuten und Städten, der den Ostseehandel jahrhundertelang prägte.",
    "Moderne Wärmepumpen erreichen bei milden Außentemperaturen eine Jahresarbeitszahl von über vier.",
    "Der Betriebsübergang nach Paragraf 613a BGB lässt die Arbeitsverhältnisse unverändert auf den Erwerber übergehen.",
    "Die Kartoffel gelangte im sechzehnten Jahrhundert aus Südamerika nach Europa und wurde zum Grundnahrungsmittel.",
    "Ein Algorithmus zur Sortierung großer Datenmengen sollte im Mittel in logarithmischer Zeit skalieren.",
    "Die gesetzliche Rentenversicherung finanziert sich im Umlageverfahren aus den Beiträgen der Erwerbstätigen.",
    "Bei Gewitter sollte man freistehende Bäume meiden und Schutz in einem Gebäude oder Fahrzeug suchen.",
    "Der Rhein ist die verkehrsreichste Wasserstraße Europas und verbindet die Schweiz mit der Nordsee.",
    "Eine Mietkaution darf höchstens drei Nettokaltmieten betragen und muss verzinst angelegt werden.",
    "Sprachmodelle zerlegen Text zunächst in Token und bilden diese anschließend auf dichte Vektoren ab.",
    "Die Weimarer Republik scheiterte an wirtschaftlicher Not, politischer Gewalt und schwachen Institutionen.",
    "Kalk im Wasserkocher lässt sich mit verdünnter Zitronensäure schonend entfernen.",
    "Der Mindestlohn wird von einer unabhängigen Kommission regelmäßig überprüft und angepasst.",
    "Ein Bausparvertrag verbindet eine Ansparphase mit dem späteren Anspruch auf ein zinsgünstiges Darlehen.",
    "Die Ostsee ist ein Brackwassermeer mit deutlich geringerem Salzgehalt als der Atlantik.",
    "Impfstoffe trainieren das Immunsystem, indem sie ein harmloses Antigen präsentieren.",
    "Der Wolf ist nach über hundert Jahren wieder in mehreren deutschen Bundesländern heimisch geworden.",
    "Bei der Steuererklärung lassen sich Werbungskosten, Sonderausgaben und außergewöhnliche Belastungen absetzen.",
    "Ein Kühlschrank arbeitet nach dem Prinzip der Verdampfung und Kondensation eines Kältemittels.",
    "Die Bundesautobahn 7 ist mit fast tausend Kilometern die längste Autobahn Deutschlands.",
    "Zur Gründung einer GmbH ist ein Stammkapital von mindestens fünfundzwanzigtausend Euro erforderlich.",
    "Der Schwarzwald ist bekannt für seine Tannenwälder, Uhrmachertradition und die Kirschtorte.",
    "Beim maschinellen Übersetzen erzeugt das Modell die Zielsprache Token für Token aus dem Kontext.",
    "Eine Patientenverfügung legt fest, welche medizinischen Maßnahmen im Ernstfall gewünscht sind.",
    "Die Zugspitze ist mit 2962 Metern der höchste Berg Deutschlands.",
]

# ── English retrieval passages ───────────────────────────────────────────────
DOC_EN = [
    "The central bank raised interest rates again in an effort to slow persistent inflation.",
    "Photosynthesis lets green plants convert sunlight, water and carbon dioxide into glucose and oxygen.",
    "A balanced diet rich in vegetables, legumes and whole grains lowers cardiovascular risk.",
    "Neural networks learn hierarchical representations by adjusting weights through backpropagation.",
    "The supreme court will decide whether the statute conflicts with constitutional guarantees.",
    "Renewable sources such as wind and solar keep getting cheaper per kilowatt hour.",
    "Ancient trade routes connected distant cities across several continents and shaped early economies.",
    "Vaccines train the immune system by presenting a harmless antigen that triggers antibody production.",
    "In distributed systems, consensus protocols like Raft keep replicated logs consistent under failure.",
    "The mountain range stretches for hundreds of kilometres along the western coast.",
    "Quarterly revenue grew twelve percent year over year, comfortably beating analyst expectations.",
    "Gravitational waves were directly detected for the first time by the LIGO observatories in 2015.",
    "A binary search tree keeps keys in sorted order, giving logarithmic lookup on balanced inputs.",
    "Regular aerobic exercise strengthens the heart, lowers resting heart rate and improves mood.",
    "Antibiotics act only against bacteria and have no effect on viral infections.",
    "The company disclosed a material weakness in its internal controls over financial reporting.",
    "Heat pumps move thermal energy rather than generating it, which is why efficiency exceeds unity.",
    "Cache invalidation and naming things remain the two genuinely hard problems in computing.",
    "The tenant may withhold part of the rent when a defect substantially impairs the use of the flat.",
    "Sea level rise threatens low-lying coastal cities through both flooding and saltwater intrusion.",
    "A well-diversified index fund is generally considered a low-cost entry point to equity markets.",
    "Transformers replaced recurrent architectures because self-attention parallelises across the sequence.",
    "Sleep deprivation impairs memory consolidation, attention and immune response.",
    "The Hanseatic League was a confederation of merchant guilds that dominated Baltic trade for centuries.",
    "Electrolysis splits water into hydrogen and oxygen and can decarbonise steel production.",
    "Employment contracts must state the notice period, the place of work and the agreed remuneration.",
    "Rayleigh scattering favours shorter wavelengths, which is why the daytime sky appears blue.",
    "A hash map offers average constant-time insertion and lookup but degrades under adversarial keys.",
    "The archipelago consists of more than three hundred islands, only forty of which are inhabited.",
    "Continuous integration catches regressions early by building and testing every pushed commit.",
    "Sourdough bread rises through fermentation by naturally occurring wild yeast and lactic bacteria.",
    "The treaty obliges signatories to report their greenhouse gas inventories annually.",
    "Solid-state drives store data in flash cells and have no moving parts to fail mechanically.",
    "Insulin resistance develops gradually and often precedes a diagnosis of type two diabetes by years.",
    "Public transit reduces per-capita carbon emissions more effectively than vehicle electrification alone.",
    "The manuscript was rejected because the control group was too small to support the conclusion.",
    "Retrieval-augmented generation grounds a language model's answer in documents fetched at query time.",
    "Deforestation reduces the planet's capacity to absorb carbon dioxide from the atmosphere.",
    "The refund policy allows exchanges within thirty days of purchase with the original receipt.",
    "Quantum computers exploit superposition and entanglement to explore many states simultaneously.",
    "Migraine attacks are often preceded by visual aura and aggravated by light and sound.",
    "The bridge spans nearly two kilometres and carries both rail and road traffic across the bay.",
]

# ── German queries (embedded WITH the model's own query prompt) ──────────────
QUERY_DE = [
    "Wie hoch ist der aktuelle Leitzins der Europäischen Zentralbank?",
    "Wie entsteht Regen?",
    "Wie wirken Impfstoffe im Körper?",
    "Welche Unterlagen brauche ich für die Gewerbeanmeldung?",
    "Wie viel Mietkaution darf ein Vermieter verlangen?",
    "Was ist der Unterschied zwischen Brutto und Netto?",
    "Wie funktioniert eine Wärmepumpe im Altbau?",
    "Welche Kosten kann ich in der Steuererklärung absetzen?",
    "Wann verjährt eine Forderung nach deutschem Recht?",
    "Wie lange dauert die Kündigungsfrist im Arbeitsvertrag?",
    "Welcher Berg ist der höchste in Deutschland?",
    "Wie backe ich ein Brot mit Sauerteig?",
    "Was hilft gegen Kalk im Wasserkocher?",
    "Wie melde ich mein Auto ab?",
    "Warum ist der Himmel blau?",
    "Wie viel Stammkapital braucht eine GmbH?",
    "Welche Symptome hat eine Grippe?",
    "Wie beantrage ich Elterngeld?",
    "Was bedeutet Jahresarbeitszahl bei einer Wärmepumpe?",
    "Wie funktioniert ein Sparplan auf einen ETF?",
    "Wo entspringt die Elbe?",
    "Wie schütze ich mich bei einem Gewitter im Freien?",
    "Was ist die Mietpreisbremse?",
    "Wie lange muss ich Kontoauszüge aufbewahren?",
    "Welche Impfungen werden für Erwachsene empfohlen?",
    "Wie schreibe ich ein handschriftliches Testament?",
    "Was kostet eine Photovoltaikanlage für ein Einfamilienhaus?",
    "Wie lerne ich am besten eine neue Sprache?",
    "Warum steigen die Strompreise?",
    "Wie funktioniert maschinelles Lernen einfach erklärt?",
    "Welche Rechte habe ich bei einer verspäteten Bahnfahrt?",
    "Wie viele Bundesländer hat Deutschland?",
    "Was ist der Unterschied zwischen Bakterien und Viren?",
    "Wie beantrage ich einen neuen Personalausweis?",
    "Wann ist die beste Zeit, Tomaten zu pflanzen?",
    "Wie hoch ist der gesetzliche Mindestlohn?",
    "Welche Versicherungen sind wirklich notwendig?",
    "Wie funktioniert die gesetzliche Rentenversicherung?",
]

# ── English queries (embedded WITH the model's own query prompt) ─────────────
QUERY_EN = [
    "What causes inflation to persist for several years?",
    "How do vaccines protect against disease?",
    "What is the best programming language for data science?",
    "How do plants convert sunlight into energy?",
    "What are the main causes of climate change?",
    "How does a heat pump work in cold weather?",
    "What are the symptoms of the flu?",
    "How do I reduce my household carbon emissions?",
    "What is retrieval-augmented generation?",
    "How does backpropagation train a neural network?",
    "What is the difference between a list and a tuple in Python?",
    "How much should I save for retirement each month?",
    "Why is the sky blue during the day?",
    "What makes a good night of sleep?",
    "How does a solid-state drive store data?",
    "What is the capital of Australia?",
    "How do I file an amended tax return?",
    "What causes migraine headaches?",
    "How does consensus work in distributed databases?",
    "What is the recommended daily intake of fibre?",
    "How do I make sourdough starter from scratch?",
    "What is the difference between HTTP and HTTPS?",
    "How long does it take to learn to touch type?",
    "What are the health benefits of aerobic exercise?",
    "How does an index fund differ from an actively managed fund?",
    "What is the boiling point of water at high altitude?",
    "How do I debug a segmentation fault in C++?",
    "What is the purpose of a load balancer?",
    "How does encryption keep messages private?",
    "What should I look for when buying a used car?",
]

# ── multi-line code (newline + indentation + punctuation-run coverage) ───────
CODE = [
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "import numpy as np\n\na = np.zeros((3, 3))\na[1, 1] = 1.0\nprint(a.sum(axis=0))\n",
    "SELECT customer_id, SUM(amount) AS total\nFROM orders\nWHERE created_at >= '2026-01-01'\nGROUP BY customer_id\nHAVING SUM(amount) > 1000\nORDER BY total DESC;",
    "#include <vector>\n\nstd::vector<float> normalize(std::vector<float> v) {\n    float n = 0.0f;\n    for (float x : v) n += x * x;\n    n = std::sqrt(n);\n    for (float & x : v) x /= n;\n    return v;\n}",
    "func main() {\n\tch := make(chan int, 4)\n\tgo func() {\n\t\tfor i := 0; i < 4; i++ {\n\t\t\tch <- i * i\n\t\t}\n\t\tclose(ch)\n\t}()\n\tfor v := range ch {\n\t\tfmt.Println(v)\n\t}\n}",
    "async function fetchAll(urls) {\n  const results = await Promise.all(\n    urls.map((u) => fetch(u).then((r) => r.json()))\n  );\n  return results.filter(Boolean);\n}",
    "class Rectangle:\n    def __init__(self, w: float, h: float) -> None:\n        self.w, self.h = w, h\n\n    @property\n    def area(self) -> float:\n        return self.w * self.h\n",
    "# .github/workflows/ci.yml\nname: CI\non:\n  push:\n    branches: [main]\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - run: cmake -B build && cmake --build build -j4\n",
    "impl Iterator for Counter {\n    type Item = u32;\n\n    fn next(&mut self) -> Option<u32> {\n        if self.count < 5 {\n            self.count += 1;\n            Some(self.count)\n        } else {\n            None\n        }\n    }\n}",
    "try:\n    cfg = json.loads(path.read_text())\nexcept json.JSONDecodeError as exc:\n    logger.error(\"bad config at %s: %s\", path, exc)\n    raise SystemExit(1)\n",
    "$ git rebase -i origin/main\n$ cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n$ cmake --build build -j8\n[100/100] Linking CXX executable crispembed\n",
    "def query_prefix(model: str) -> str | None:\n    if \"f2llm-v2\" in model:\n        return (\"Instruct: Given a question, retrieve passages \"\n                \"that can help answer the question.\\nQuery: \")\n    if \"arctic-embed\" in model and \"-v2\" in model:\n        return \"query: \"\n    return None\n",
    "<config>\n  <threads>8</threads>\n  <backend name=\"metal\" enabled=\"true\"/>\n  <paths>\n    <model>/models/embed.gguf</model>\n  </paths>\n</config>",
    "for i in $(seq 1 5); do\n  echo \"run $i\"\n  ./crispembed -m model.gguf --json \"Beispieltext\" >> out.jsonl\ndone\n",
    "type Result<T> =\n  | { ok: true; value: T }\n  | { ok: false; error: string };\n\nconst unwrap = <T,>(r: Result<T>): T => {\n  if (!r.ok) throw new Error(r.error);\n  return r.value;\n};",
    "MATCH (a:Person)-[:KNOWS]->(b:Person)\nWHERE a.city = 'Berlin'\nRETURN a.name, count(b) AS friends\nORDER BY friends DESC\nLIMIT 10;",
    "public record Point(double x, double y) {\n    double distanceTo(Point o) {\n        return Math.hypot(x - o.x, y - o.y);\n    }\n}",
    "// Kommentar auf Deutsch: berechnet den gleitenden Mittelwert\nfloat moving_average(const float * x, int n, int w) {\n    float s = 0.0f;\n    for (int i = n - w; i < n; ++i) s += x[i];\n    return s / (float) w;\n}",
]

# ── newline-heavy structured prose (markdown, tables, logs, addresses) ───────
STRUCT = [
    "# Quartalsbericht 2026\n\n## Umsatz\n\n- Deutschland: 12,4 Mio. EUR\n- Österreich: 3,1 Mio. EUR\n- Schweiz: 2,8 Mio. EUR\n\n## Ausblick\n\nDas Management erwartet ein moderates Wachstum im zweiten Halbjahr.",
    "| Modell | Größe | Cosine |\n|--------|-------|--------|\n| q8_0   | 330MB | 0.9990 |\n| q4_k   | 274MB | 0.9540 |\n| iq4_xs | 265MB | 0.9480 |",
    "Sehr geehrte Damen und Herren,\n\nhiermit kündige ich meinen Mietvertrag für die Wohnung in der\nBeispielstraße 12, 10115 Berlin, fristgerecht zum 30. September.\n\nMit freundlichen Grüßen\nMax Mustermann",
    "Zutaten:\n- 500 g Weizenmehl Type 550\n- 350 ml lauwarmes Wasser\n- 10 g Salz\n- 100 g Sauerteigansatz\n\nZubereitung:\n1. Alle Zutaten verkneten.\n2. 30 Minuten ruhen lassen.\n3. Dreimal im Abstand von 30 Minuten dehnen und falten.\n4. Über Nacht im Kühlschrank gehen lassen.",
    "2026-08-04 11:22:03 INFO  loaded model f2llm-v2-0.6b.gguf (2390 MB)\n2026-08-04 11:22:04 WARN  metal: falling back to CPU for op CPY\n2026-08-04 11:22:09 INFO  embedded 64 texts in 5.13 s\n2026-08-04 11:22:09 ERROR upload failed: connection reset by peer\n",
    "Checklist before release\n\n[x] unit tests green\n[x] clang-format clean\n[ ] PERFORMANCE.md updated\n[ ] registry pin refreshed\n[ ] CHANGELOG entry written\n",
    "FAQ\n\nQ: Wie lange dauert der Versand?\nA: In der Regel zwei bis drei Werktage innerhalb Deutschlands.\n\nQ: Kann ich die Ware zurückgeben?\nA: Ja, innerhalb von 30 Tagen im Originalzustand.\n",
    "Anschrift:\nMusterfirma GmbH\nAbteilung Einkauf\nIndustriestraße 45a\n70565 Stuttgart\nDeutschland\n\nUSt-IdNr.: DE123456789",
    "Agenda\n\n09:00  Begrüßung\n09:15  Stand der Migration\n10:00  Pause\n10:15  Diskussion: Quantisierung und Qualität\n11:30  Beschlüsse\n12:00  Ende\n",
    "## Installation\n\n```bash\npip install crispembed\n```\n\n## Usage\n\n```python\nfrom crispembed import Session\ns = Session(\"arctic-embed-m-v2\")\nprint(s.embed([\"Hallo Welt\"]))\n```\n",
    "Fehlermeldung:\n\n    Traceback (most recent call last):\n      File \"run.py\", line 42, in <module>\n        main()\n      File \"run.py\", line 31, in main\n        cfg = load(path)\n    FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'\n",
    "Article 5\n\n1. Personal data shall be processed lawfully, fairly and transparently.\n2. Data shall be collected for specified, explicit and legitimate purposes.\n3. Data shall be adequate, relevant and limited to what is necessary.\n",
    "Wetterbericht Norddeutschland\n\nHeute:   stark bewölkt, 17 °C, Wind aus Nordwest\nMorgen:  Schauer, 15 °C\nÜbermorgen: heiter, 19 °C\n",
    "TODO\n====\n* fix the tokenizer whitespace collapse\n* add the German retrieval fixture\n* re-run the A/B with the larger corpus\n* update PLAN.md and push\n",
    "Rechnung Nr. 2026-0815\n\nPos.  Beschreibung              Menge   Einzelpreis   Gesamt\n1     Beratungsleistung          8 h      120,00 €    960,00 €\n2     Reisekosten                1        215,50 €    215,50 €\n\nZwischensumme: 1.175,50 €\nUmsatzsteuer 19 %: 223,35 €\nGesamtbetrag: 1.398,85 €\n",
    "Bedienungsanleitung\n\nSchritt 1: Gerät auspacken und auf einen ebenen Untergrund stellen.\nSchritt 2: Netzkabel anschließen.\nSchritt 3: Ein-/Aus-Schalter auf der Rückseite betätigen.\n\nAchtung: Das Gerät darf nicht in der Nähe von Wasser betrieben werden!\n",
    "commit 19f0f0eb0a1b2c3d\nAuthor: Beispiel <mail@example.org>\nDate:   Tue Aug 4 21:03:11 2026 +0200\n\n    fix(embed): apply the query prefix only to queries\n\n    Documents take no prefix in every family we ship.\n",
]


def build():
    cats = [("doc.de", DOC_DE), ("doc.en", DOC_EN), ("query.de", QUERY_DE),
            ("query.en", QUERY_EN), ("code", CODE), ("struct", STRUCT)]
    calib, evalr = [], []
    for role, items in cats:
        for i, t in enumerate(items):
            (evalr if i % 3 == 2 else calib).append({"role": role, "text": t})
    return calib, evalr


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


if __name__ == "__main__":
    calib, evalr = build()
    n1 = write(HERE / "calib_corpus.jsonl", calib)
    n2 = write(HERE / "eval_corpus.jsonl", evalr)
    from collections import Counter
    print(f"calib {n1}  {dict(Counter(r['role'] for r in calib))}")
    print(f"eval  {n2}  {dict(Counter(r['role'] for r in evalr))}")
