"""
KenLM-backed n-gram language model rescorer for Kinyarwanda ASR output.

Replaces the original pure-Python Stupid Backoff model with a compiled
KenLM model (modified Kneser-Ney smoothing), which handles this corpus's
size/diversity far better and queries ~1000x faster.

Pipeline:
  1. Build a 5-gram KenLM model from lm_corpus.txt   -> python lm_rescore.py build
  2. Word-level correction on ASR hypothesis          -> python lm_rescore.py test
  3. Integration into evaluate.py                     -> import correct_words

Requires the kenlm Python package and the lmplz/build_binary binaries in
kenlm_bin/ (built from https://github.com/kpu/kenlm; see kenlm_build.log
for how they were built on this machine).
"""

import subprocess
import sys
from collections import Counter
from pathlib import Path

import kenlm

CORPUS_PATH  = Path(__file__).parent / "lm_corpus.txt"
ARPA_PATH    = Path(__file__).parent / "kenlm_model.arpa"
BINARY_PATH  = Path(__file__).parent / "kenlm_model.binary"
LMPLZ        = Path(__file__).parent / "kenlm_bin" / "lmplz"
BUILD_BINARY = Path(__file__).parent / "kenlm_bin" / "build_binary"

NGRAM_ORDER = 5
# KenLM scores are log10 probabilities (vs. natural log in the old Stupid
# Backoff model) -- this threshold is calibrated for that scale, not nats.
MIN_IMPROVE = 1.5
TOP_VOCAB   = 5_000      # words considered "known" -- skip correction for these
MAX_CANDS   = 8          # max candidates to score per unknown word


# ── Model + vocab loading ─────────────────────────────────────────────────────

_model: kenlm.Model | None = None
_top_vocab: list[str] | None = None


def load_model() -> kenlm.Model:
    global _model
    if _model is not None:
        return _model
    if not BINARY_PATH.exists():
        raise FileNotFoundError(
            f"LM not built yet. Run: python lm_rescore.py build"
        )
    _model = kenlm.Model(str(BINARY_PATH))
    return _model


def load_top_vocab() -> list[str]:
    """Most frequent words in the corpus, used as the 'known word' set.

    KenLM's Python bindings don't expose vocabulary enumeration, so this is
    derived directly from the training corpus rather than from the model.
    """
    global _top_vocab
    if _top_vocab is not None:
        return _top_vocab
    counts: Counter = Counter()
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        counts.update(line.split())
    _top_vocab = [w for w, _ in counts.most_common(TOP_VOCAB)]
    return _top_vocab


def build():
    if not LMPLZ.exists() or not BUILD_BINARY.exists():
        sys.exit(f"lmplz/build_binary not found in {LMPLZ.parent}. "
                  f"Build KenLM's command-line tools first (see kenlm_build.log).")

    print(f"Training {NGRAM_ORDER}-gram KenLM model from {CORPUS_PATH} ...")
    with open(CORPUS_PATH, "rb") as corpus, open(ARPA_PATH, "wb") as arpa_out:
        subprocess.run(
            [str(LMPLZ), "-o", str(NGRAM_ORDER)],
            stdin=corpus, stdout=arpa_out, check=True,
        )
    subprocess.run([str(BUILD_BINARY), str(ARPA_PATH), str(BINARY_PATH)], check=True)

    sz = BINARY_PATH.stat().st_size / 1e6
    print(f"Model saved to {BINARY_PATH} ({sz:.1f} MB)")


# ── Candidate generation (edit distance <= 1) ─────────────────────────────────

def _edits1(word: str, vocab: list[str]) -> list[str]:
    """Words in vocab within edit-distance 1 of word."""
    results = []
    wl = len(word)
    for v in vocab:
        vl = len(v)
        if abs(vl - wl) > 1:
            continue
        # quick prefix filter
        if word[:2] != v[:2] and word[-2:] != v[-2:]:
            continue
        # Levenshtein <= 1
        if wl == vl:
            diffs = sum(a != b for a, b in zip(word, v))
            if diffs <= 1:
                results.append(v)
        else:
            shorter, longer = (word, v) if wl < vl else (v, word)
            # check if longer is shorter + one insertion
            found = False
            for skip in range(len(longer)):
                if shorter == longer[:skip] + longer[skip+1:]:
                    found = True
                    break
            if found:
                results.append(v)
        if len(results) >= MAX_CANDS * 4:
            break
    return results[:MAX_CANDS]


# ── Word-level correction ─────────────────────────────────────────────────────

def correct_words(text: str) -> str:
    """
    For each word not in the top vocabulary, try edit-distance-1 neighbours
    and substitute if the LM score in context improves meaningfully.

    This is a light pass -- it won't rearrange words or invent new ones.
    It fixes acoustic confusions like misili->misiri, merongo->mirongo.
    """
    model = load_model()
    vocab_list = load_top_vocab()
    top_vocab_set = set(vocab_list)

    words = text.split()
    result = list(words)

    for i, word in enumerate(words):
        # skip short words and common words
        if len(word) <= 3 or word in top_vocab_set:
            continue

        # context window for scoring
        lo = max(0, i - NGRAM_ORDER + 1)
        hi = min(len(words), i + NGRAM_ORDER)

        def score_at(w: str) -> float:
            ctx = result[lo:i] + [w] + words[i + 1:hi]
            return model.score(" ".join(ctx), bos=False, eos=False)

        base = score_at(word)
        best_w = word
        best_s = base

        for cand in _edits1(word, vocab_list):
            s = score_at(cand)
            if s > best_s + MIN_IMPROVE:
                best_s = s
                best_w = cand

        result[i] = best_w

    return " ".join(result)


# ── CLI ───────────────────────────────────────────────────────────────────────

def test():
    model = load_model()
    # Score some known-good vs known-bad pairs
    pairs = [
        ("yezu kristo w'i nazareti uwo mwabambye ku musaraba imana ikamuzura",
         "yeze kiristo w'i nazareti uwo mwabambye ku musaraba imana ikamuzura"),
        ("petero na yohani bagiye mu rugo rw'ingoro y'imana",
         "peter na yohani bagiye mu rugo rw'ingoro y'imana"),
        ("bose buzuzwa mwuka muziranenge batangira kuvuga izindi ndimi",
         "bose buzuzwa amwuka muziranenge batangira kuvuga izindi ndimi"),
    ]
    for good, bad in pairs:
        sg = model.score(good, bos=True, eos=True)
        sb = model.score(bad, bos=True, eos=True)
        print(f"\nGOOD ({sg:.3f}): {good}")
        print(f"BAD  ({sb:.3f}): {bad}")
        print(f"Delta = {sg - sb:.3f}  ({'correct' if sg > sb else 'WRONG DIRECTION'})")

    print("\n--- Word correction demo ---")
    test_hyp = "peter na yohani bagiye mu rugo rw'ingoro y'imana hariho umuntu wavutsa ari ikirema"
    corrected = correct_words(test_hyp)
    print(f"Input:     {test_hyp}")
    print(f"Corrected: {corrected}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "build":
        build()
    elif cmd == "test":
        test()
    else:
        print("Usage: python lm_rescore.py build|test")
