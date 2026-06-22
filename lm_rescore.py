"""
Pure-Python n-gram language model rescorer for Kinyarwanda ASR output.

No compiled dependencies (kenlm / boost not required).
Uses Stupid Backoff (Brants et al. 2007) — fast, good enough for rescoring.

Pipeline:
  1. Build a 5-gram model from lm_corpus.txt     → python lm_rescore.py build
  2. Word-level correction on ASR hypothesis      → python lm_rescore.py test
  3. Integration into evaluate.py                → import correct_words

Architecture:
  - Stores n-gram counts in memory (~60–80 MB for 658k-word corpus)
  - Stupid Backoff: P(w|context) = count(context,w)/count(context)
    backing off through shorter contexts with penalty 0.4
  - For each "unusual" word (not in top-5k vocab), tries edit-distance-1
    neighbours from the corpus vocabulary and replaces if LM score improves
"""

import re
import math
import pickle
import sys
from collections import defaultdict, Counter
from pathlib import Path

CORPUS_PATH  = Path(__file__).parent / "lm_corpus.txt"
MODEL_PATH   = Path(__file__).parent / "lm_model.pkl"
NGRAM_ORDER  = 5
BACKOFF_ALPHA = 0.4        # Stupid Backoff penalty per level
MIN_IMPROVE  = 0.5         # minimum log-prob gain (nats) to accept correction
TOP_VOCAB    = 5_000       # words considered "known" — skip correction for these
MAX_CANDS    = 8           # max candidates to score per unknown word


# ── N-gram model ─────────────────────────────────────────────────────────────

class StupidBackoffLM:
    """Stupid Backoff n-gram language model (Brants et al. 2007)."""

    def __init__(self, order: int = NGRAM_ORDER):
        self.order = order
        # counts[(w1,...,wn)] = int
        self.counts: dict[tuple, int] = {}
        self.vocab: set[str] = set()
        self.top_vocab: list[str] = []  # most frequent words

    def train(self, corpus_path: Path):
        print(f"Training {self.order}-gram LM from {corpus_path} ...")
        unigram_counts: Counter = Counter()
        ngram_counts: Counter = Counter()

        lines = corpus_path.read_text(encoding="utf-8").splitlines()
        total_lines = len(lines)

        for i, line in enumerate(lines):
            if i % 10_000 == 0:
                print(f"  {i:,}/{total_lines:,} lines", end="\r")
            tokens = ["<s>"] + line.split() + ["</s>"]
            unigram_counts.update(tokens)
            for n in range(1, self.order + 1):
                for j in range(len(tokens) - n + 1):
                    ngram_counts[tuple(tokens[j:j+n])] += 1

        print(f"\nVocab size: {len(unigram_counts):,}  N-grams: {len(ngram_counts):,}")
        self.counts = dict(ngram_counts)
        self.vocab = set(unigram_counts.keys()) - {"<s>", "</s>"}
        self.top_vocab = [w for w, _ in unigram_counts.most_common(TOP_VOCAB)
                          if w not in {"<s>", "</s>"}]

    def score(self, ngram: tuple) -> float:
        """Stupid Backoff log-probability for an n-gram (natural log)."""
        if len(ngram) == 1:
            c = self.counts.get(ngram, 0)
            total = self.counts.get(("<s>",), 1)
            return math.log((c + 1e-8) / (total + len(self.vocab)))
        c_ngram = self.counts.get(ngram, 0)
        c_ctx   = self.counts.get(ngram[:-1], 0)
        if c_ngram > 0 and c_ctx > 0:
            return math.log(c_ngram / c_ctx)
        # back off
        return math.log(BACKOFF_ALPHA) + self.score(ngram[1:])

    def sentence_score(self, tokens: list[str]) -> float:
        """Sum of log-probs for all n-grams in the sentence."""
        padded = ["<s>"] * (self.order - 1) + tokens + ["</s>"]
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            ngram = tuple(padded[i - self.order + 1: i + 1])
            total += self.score(ngram)
        return total

    def score_per_word(self, text: str) -> float:
        tokens = text.split()
        if not tokens:
            return -1e9
        return self.sentence_score(tokens) / len(tokens)


# ── Model persistence ─────────────────────────────────────────────────────────

_model: StupidBackoffLM | None = None

def load_model() -> StupidBackoffLM:
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LM not built yet. Run: python lm_rescore.py build"
        )
    print("Loading LM ...", end=" ", flush=True)
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    _model = StupidBackoffLM(order=data["order"])
    _model.counts    = data["counts"]
    _model.vocab     = data["vocab"]
    _model.top_vocab = data["top_vocab"]
    print("done.")
    return _model


def build():
    lm = StupidBackoffLM(order=NGRAM_ORDER)
    lm.train(CORPUS_PATH)
    # Save as plain dict so pickle doesn't depend on __main__ class resolution
    data = {"order": lm.order, "counts": lm.counts,
            "vocab": lm.vocab, "top_vocab": lm.top_vocab}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    sz = MODEL_PATH.stat().st_size / 1e6
    print(f"Model saved to {MODEL_PATH} ({sz:.1f} MB)")


# ── Candidate generation (edit distance ≤ 1) ─────────────────────────────────

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
        # Levenshtein ≤ 1
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

    This is a light pass — it won't rearrange words or invent new ones.
    It fixes acoustic confusions like misili→misiri, merongo→mirongo.
    """
    lm = load_model()
    top_vocab_set = set(lm.top_vocab)
    vocab_list    = lm.top_vocab  # ordered by frequency → faster early exit

    words = text.split()
    result = list(words)
    order = lm.order

    for i, word in enumerate(words):
        # skip short words and common words
        if len(word) <= 3 or word in top_vocab_set:
            continue

        # context window for scoring
        lo = max(0, i - order + 1)
        hi = min(len(words), i + order)

        def score_at(w: str) -> float:
            ctx = result[lo:i] + [w] + words[i+1:hi]
            return lm.sentence_score(ctx)

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
    lm = load_model()
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
        sg = lm.score_per_word(good)
        sb = lm.score_per_word(bad)
        print(f"\nGOOD ({sg:.3f}): {good}")
        print(f"BAD  ({sb:.3f}): {bad}")
        print(f"Δ = {sg - sb:.3f}  ({'✓ correct' if sg > sb else '✗ wrong direction'})")

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
