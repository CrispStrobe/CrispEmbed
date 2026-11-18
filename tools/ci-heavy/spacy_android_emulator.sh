#!/usr/bin/env bash
# Runs inside reactivecircus/android-emulator-runner: push the x86_64 build and
# model files, run spacy_smoke on the emulator, pull its stdout byte-exactly.
set -euo pipefail
run_dir="$1"
source "$run_dir/hashes.env"
remote=/data/local/tmp/spacy
adb shell "rm -rf $remote && mkdir -p $remote"
adb push "$run_dir/x86_64/libcrispembed-spacy.so" "$run_dir/x86_64/spacy-smoke" \
  "$run_dir/model.gguf" "$run_dir/resources.json" "$run_dir/ner.gguf" "$run_dir/texts.txt" "$remote/"
adb shell "chmod 755 $remote/spacy-smoke"
adb exec-out "cd $remote && LD_LIBRARY_PATH=$remote ./spacy-smoke model.gguf $MODEL_SHA resources.json $RESOURCES_SHA texts.txt ner.gguf $NER_SHA" \
  > "$run_dir/emulator-smoke-x86_64.jsonl"
adb shell getprop ro.build.version.sdk | tr -d '\r' > "$run_dir/emulator-api.txt"
test -s "$run_dir/emulator-smoke-x86_64.jsonl"
