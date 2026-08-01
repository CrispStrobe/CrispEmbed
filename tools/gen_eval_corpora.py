#!/usr/bin/env python3
"""Generate the bilingual (EN+DE) imatrix calibration/eval corpora.

Writes tools/kaggle/crispembed-imatrix-quant/{calib_corpus.txt,eval_corpus.txt},
which the harness auto-reads for the embed / sparse / ColBERT A/B (read_corpus()).
The A/B compares a quant vs the full-precision model on the SAME text, so no labels
are needed — only realistic, diverse text.

The text below is **self-authored** for this project and released into the public
domain (CC0 / MIT — no attribution required), so the corpora carry no third-party
content license (avoids CC-BY / Wikipedia-CC-BY-SA). Each entry is an EN sentence
and its DE translation, so the A/B can surface EN-vs-DE quantization differences on
identical content across a spread of domains.

Usage: python tools/gen_eval_corpora.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "kaggle" / "crispembed-imatrix-quant"

# (EN, DE) parallel pairs — self-authored, CC0. Diverse domains.
CALIB_PAIRS = [
    ("Machine learning models turn raw text into dense numerical vectors.",
     "Modelle des maschinellen Lernens wandeln rohen Text in dichte Zahlenvektoren um."),
    ("The central bank raised interest rates to slow down inflation.",
     "Die Zentralbank erhöhte die Zinsen, um die Inflation zu bremsen."),
    ("Photosynthesis lets green plants convert sunlight into chemical energy.",
     "Die Photosynthese ermöglicht es grünen Pflanzen, Sonnenlicht in chemische Energie umzuwandeln."),
    ("A balanced diet with vegetables and whole grains supports good health.",
     "Eine ausgewogene Ernährung mit Gemüse und Vollkorn fördert die Gesundheit."),
    ("The mountain range stretches for hundreds of kilometres along the coast.",
     "Das Gebirge erstreckt sich über Hunderte von Kilometern entlang der Küste."),
    ("Ancient trade routes connected distant cities across several continents.",
     "Alte Handelswege verbanden weit entfernte Städte über mehrere Kontinente hinweg."),
    ("Regular exercise strengthens the heart and improves overall mood.",
     "Regelmäßiger Sport stärkt das Herz und verbessert die allgemeine Stimmung."),
    ("She added two cups of flour and a pinch of salt to the dough.",
     "Sie gab zwei Tassen Mehl und eine Prise Salz zum Teig hinzu."),
    ("The orchestra rehearsed the symphony for several hours before the concert.",
     "Das Orchester probte die Sinfonie mehrere Stunden lang vor dem Konzert."),
    ("Renewable energy sources like wind and solar keep getting cheaper.",
     "Erneuerbare Energiequellen wie Wind und Sonne werden immer günstiger."),
    ("The spacecraft entered orbit after a journey of several months.",
     "Die Raumsonde trat nach einer Reise von mehreren Monaten in die Umlaufbahn ein."),
    ("Learning a second language improves memory and opens new opportunities.",
     "Das Erlernen einer zweiten Sprache verbessert das Gedächtnis und eröffnet neue Chancen."),
    ("Heavy rain and strong winds are expected across the region tomorrow.",
     "Für morgen werden in der Region starker Regen und kräftiger Wind erwartet."),
    ("The new railway line cut the travel time between the two cities in half.",
     "Die neue Bahnstrecke halbierte die Reisezeit zwischen den beiden Städten."),
    ("Distributed databases replicate records so that lookups stay fast.",
     "Verteilte Datenbanken replizieren Datensätze, damit Abfragen schnell bleiben."),
]

EVAL_PAIRS = [
    ("Doctors recommend rest and fluids to recover from a mild fever.",
     "Ärzte empfehlen Ruhe und Flüssigkeit, um sich von leichtem Fieber zu erholen."),
    ("The function returns an error when the input list is empty.",
     "Die Funktion gibt einen Fehler zurück, wenn die Eingabeliste leer ist."),
    ("Cutting carbon emissions requires a shift toward cleaner transport.",
     "Die Senkung der CO2-Emissionen erfordert einen Wechsel zu saubererem Verkehr."),
    ("The painting uses warm colours to convey a sense of calm.",
     "Das Gemälde verwendet warme Farben, um ein Gefühl der Ruhe zu vermitteln."),
    ("Small class sizes let teachers give students more individual attention.",
     "Kleine Klassen ermöglichen es Lehrern, den Schülern mehr individuelle Aufmerksamkeit zu geben."),
    ("Migratory birds travel thousands of kilometres between seasons.",
     "Zugvögel legen zwischen den Jahreszeiten Tausende von Kilometern zurück."),
    ("The bridge was designed to withstand strong earthquakes.",
     "Die Brücke wurde so entworfen, dass sie starken Erdbeben standhält."),
    ("Deep ocean currents move heat slowly around the entire planet.",
     "Tiefe Meeresströmungen transportieren Wärme langsam um den gesamten Planeten."),
]


def _write(path, pairs):
    lines = [x for en, de in pairs for x in (en, de)]
    (OUT / path).write_text("\n".join(lines) + "\n")
    return len(lines)


def main():
    n1 = _write("calib_corpus.txt", CALIB_PAIRS)
    n2 = _write("eval_corpus.txt", EVAL_PAIRS)
    print(f"wrote {n1} calib + {n2} eval lines (EN+DE, self-authored/CC0) to {OUT}")


if __name__ == "__main__":
    main()
