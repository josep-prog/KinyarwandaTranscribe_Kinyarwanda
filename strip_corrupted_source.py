"""
One-off cleanup: remove lines in lm_corpus.txt that were injected from
mbazaNLP/common-voice-kinyarwanda-english-dataset, which turned out to be
systemically corrupted (rows are unrelated Kinyarwanda + English fragments
glued together, not real sentences).

Re-derives the exact sentence set that extract_hf_text.py's
extract_mbaza_common_voice() would have produced (same filters, deterministic
given the same dataset state), then drops any corpus line matching it
(case-insensitive, matching the original dedup logic).
"""

from pathlib import Path
from extract_hf_text import extract_mbaza_common_voice

CORPUS_PATH = Path(__file__).parent / "lm_corpus.txt"
BACKUP_PATH = Path(__file__).parent / "lm_corpus.before_strip.txt"


def main():
    print("Re-deriving mbaza Common Voice sentence set to identify for removal...")
    bad_sentences = set(s.lower() for s in extract_mbaza_common_voice())
    print(f"Identified {len(bad_sentences):,} corrupted sentences to strip.")

    lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    print(f"Corpus before: {len(lines):,} lines")

    CORPUS_PATH.rename(BACKUP_PATH)
    print(f"Backed up original corpus to {BACKUP_PATH}")

    kept = [line for line in lines if line.strip().lower() not in bad_sentences]
    removed = len(lines) - len(kept)
    print(f"Removed {removed:,} lines, kept {len(kept):,} lines")

    CORPUS_PATH.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"Wrote cleaned corpus to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
