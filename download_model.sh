#!/usr/bin/env bash
# Downloads (or resumes) the Kinyarwanda Whisper CTranslate2 model.
# Safe to re-run: each curl uses -C - to resume partial downloads.
set -e

REPO="leophill/whisper-large-v3-turbo-sw-kinyarwanda-ct2"
DEST="models/whisper-large-v3-turbo-kinyarwanda-ct2"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$DEST"
cd "$DEST"

for f in config.json preprocessor_config.json tokenizer.json vocabulary.json README.md model.bin; do
    echo "Fetching $f ..."
    curl -L -C - --retry 50 --retry-delay 5 --retry-all-errors --connect-timeout 30 -o "$f" "$BASE_URL/$f"
done

echo "Done. Model files in $DEST"
