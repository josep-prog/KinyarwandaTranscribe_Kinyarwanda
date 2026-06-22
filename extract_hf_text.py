"""
Extract Kinyarwanda sentence text from HuggingFace datasets and merge into lm_corpus.txt.

Sources:
  1. DigitalUmuganda/kinyarwanda-tts-dataset (3,991 sentences)
  2. mbazaNLP/common-voice-kinyarwanda-english-dataset (700k+ rows)
  3. mbazaNLP/fleurs-kinyarwanda (FLEURS sentences)
"""

from datasets import load_dataset
from pathlib import Path
import re
import sys

CORPUS_PATH = Path(__file__).parent / "lm_corpus.txt"

MIN_SENTENCE_LEN = 15    # minimum characters for a valid sentence
MAX_SENTENCE_LEN = 500   # maximum characters for a valid sentence

# ── Helpers ──────────────────────────────────────────────────────────────────

def is_mostly_kinyarwanda(text: str) -> bool:
    """Check if text looks like genuine Kinyarwanda (has enough Bantu-language
    characteristics: vowel-final words, common prefixes)."""
    if not text:
        return False
    # Filter out obviously non-Kinyarwanda content
    english_words = {"the", "and", "for", "are", "but", "not", "you", "all",
                     "can", "had", "her", "was", "one", "our", "out", "has",
                     "have", "been", "were", "said", "they", "that", "with",
                     "this", "from", "which", "what", "their", "will", "would"}
    words = text.lower().split()
    if not words:
        return False
    english_count = sum(1 for w in words if w in english_words)
    if english_count > len(words) * 0.3:
        return False
    return True


def clean_sentence(text: str) -> str | None:
    """Clean and validate a single sentence."""
    text = text.strip()
    # Remove leading/trailing quotes
    text = text.strip('"\'«»“”')
    # Skip if too short or too long
    if len(text) < MIN_SENTENCE_LEN or len(text) > MAX_SENTENCE_LEN:
        return None
    # Skip if it looks like an ID or code
    if re.match(r'^[A-Z0-9_\-/\\. ]+$', text):
        return None
    # Skip if it's just numbers/punctuation
    if sum(1 for c in text if c.isalpha()) < 5:
        return None
    # Skip if it contains HTML/XML
    if '<' in text and '>' in text:
        return None
    # Normalize whitespace
    text = ' '.join(text.split())
    # Ensure ends with sentence-ending punctuation
    if text and text[-1] not in '.!?':
        text += '.'
    return text


# ── Source 1: DigitalUmuganda TTS ───────────────────────────────────────────

def extract_tts_dataset() -> list[str]:
    """Extract sentences from DigitalUmuganda/kinyarwanda-tts-dataset."""
    print("📥 Loading DigitalUmuganda TTS dataset...")
    sentences = []
    try:
        dataset = load_dataset('DigitalUmuganda/kinyarwanda-tts-dataset',
                               split='train', streaming=True)
        cols = list(dataset.features.keys())
        # The text column is the one that's NOT 'TTS 1_2' and NOT 'Unnamed: 2'
        text_col = None
        for col in cols:
            if col != 'Unnamed: 2' and 'TTS' not in col:
                text_col = col
                break

        if not text_col:
            # Fallback: try the second column
            text_col = cols[1] if len(cols) > 1 else None

        print(f"  Using column: {text_col}")

        for i, example in enumerate(dataset):
            val = example.get(text_col) if text_col else None
            if val and isinstance(val, str) and val.strip():
                cleaned = clean_sentence(val.strip())
                if cleaned and is_mostly_kinyarwanda(cleaned):
                    sentences.append(cleaned)
            if i % 500 == 0:
                print(f"  Processed {i} rows, collected {len(sentences)} sentences...")

    except Exception as e:
        print(f"  ⚠ TTS dataset error: {e}")

    print(f"  ✅ Extracted {len(sentences)} sentences from TTS dataset")
    return sentences


# ── Source 2: mbazaNLP Common Voice Kinyarwanda ────────────────────────────

def extract_mbaza_common_voice() -> list[str]:
    """Extract Kinyarwanda sentences from mbazaNLP/common-voice-kinyarwanda-english-dataset."""
    print("📥 Loading mbazaNLP Common Voice Kinyarwanda-English dataset...")
    sentences = []
    try:
        dataset = load_dataset(
            'mbazaNLP/common-voice-kinyarwanda-english-dataset',
            split='train',
            streaming=True
        )
        cols = list(dataset.features.keys())
        print(f"  Columns: {cols}")

        # Try common column names for Kinyarwanda text
        text_col_candidates = ['sentence', 'text', 'transcript',
                               'transcription', 'Kinyarwanda', 'rw_sentence']
        text_col = None
        for c in text_col_candidates:
            if c in cols:
                text_col = c
                break
        if not text_col:
            # Pick the first text-like column that isn't audio/English
            for c in cols:
                if c not in ('audio', 'path', 'client_id', 'up_votes',
                             'down_votes', 'age', 'gender', 'accent',
                             'locale', 'segment', 'id', 'English', 'en'):
                    text_col = c
                    break

        print(f"  Using column: {text_col}")

        for i, example in enumerate(dataset):
            val = example.get(text_col) if text_col else None
            if val and isinstance(val, str) and val.strip():
                cleaned = clean_sentence(val.strip())
                if cleaned and is_mostly_kinyarwanda(cleaned):
                    sentences.append(cleaned)
            if i % 5000 == 0:
                print(f"  Processed {i} rows, collected {len(sentences)} sentences...")

    except Exception as e:
        print(f"  ⚠ mbazaNLP Common Voice error: {e}")

    print(f"  ✅ Extracted {len(sentences)} sentences from mbazaNLP Common Voice")
    return sentences


# ── Source 3: mbazaNLP FLEURS Kinyarwanda ──────────────────────────────────

def extract_fleurs() -> list[str]:
    """Extract Kinyarwanda sentences from mbazaNLP/fleurs-kinyarwanda."""
    print("📥 Loading mbazaNLP FLEURS Kinyarwanda dataset...")
    sentences = []
    try:
        dataset = load_dataset(
            'mbazaNLP/fleurs-kinyarwanda',
            split='train',
            streaming=True
        )
        cols = list(dataset.features.keys())
        print(f"  Columns: {cols}")

        text_col = None
        for c in ('transcription', 'sentence', 'text', 'raw_transcription'):
            if c in cols:
                text_col = c
                break

        for i, example in enumerate(dataset):
            val = example.get(text_col) if text_col else None
            if val and isinstance(val, str) and val.strip():
                cleaned = clean_sentence(val.strip())
                if cleaned and is_mostly_kinyarwanda(cleaned):
                    sentences.append(cleaned)
            if i % 1000 == 0:
                print(f"  Processed {i} rows, collected {len(sentences)} sentences...")

    except Exception as e:
        print(f"  ⚠ FLEURS error: {e}")

    print(f"  ✅ Extracted {len(sentences)} sentences from FLEURS")
    return sentences


# ── Merge ───────────────────────────────────────────────────────────────────

def main():
    all_new = []

    # Source 1
    all_new.extend(extract_tts_dataset())

    # Source 2
    all_new.extend(extract_mbaza_common_voice())

    # Source 3
    all_new.extend(extract_fleurs())

    # Deduplicate
    unique = list(dict.fromkeys(all_new))  # preserves order
    print(f"\n📊 Total collected: {len(all_new)} unique: {len(unique)}")

    if not unique:
        print("No new sentences found. Exiting.")
        return

    # Read existing corpus
    existing = set()
    if CORPUS_PATH.exists():
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(line)

    print(f"  Existing corpus: {len(existing)} lines")

    # Add only truly new sentences (case-insensitive dedup)
    existing_lower = {s.lower() for s in existing}
    truly_new = [s for s in unique if s.lower() not in existing_lower]

    if not truly_new:
        print("  No new sentences to add.")
        # We can still rebuild
        return

    print(f"  Adding {len(truly_new)} new sentences...")

    # Append to corpus
    with open(CORPUS_PATH, 'a', encoding='utf-8') as f:
        for s in truly_new:
            f.write(s + '\n')

    print(f"✅ Done! Corpus expanded by {len(truly_new)} sentences.")
    print(f"   Total corpus size: {len(existing) + len(truly_new)} lines")


if __name__ == '__main__':
    main()
