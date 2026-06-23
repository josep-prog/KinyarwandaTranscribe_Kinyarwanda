#!/usr/bin/env python3
"""
Pseudo-label audio files for future fine-tuning: transcribe with the current
(KenLM-corrected) pipeline and save the result plus per-segment confidence
metrics, so a later filtering step can discard low-confidence pseudo-labels
before they're used as training data.

Idempotent: skips files that already have an output JSON, so this can be
re-run as more source audio is added over time.

Usage:
    python pseudo_label.py --source-dir ~/Music
    python pseudo_label.py --source-dir ~/Music --limit 1   # quick pilot
"""

import argparse
import json
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

from normalize import normalize_hypothesis
from lm_rescore import correct_words

MODEL_DIR = Path(__file__).parent / "models" / "whisper-large-v3-turbo-kinyarwanda-ct2"
OUTPUT_DIR = Path(__file__).parent / "pseudo_labels"
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm")


def output_path_for(audio_path: Path) -> Path:
    # Use a stable id derived from the filename so re-runs are idempotent
    # even though these filenames contain spaces/punctuation.
    safe_stem = "".join(c if c.isalnum() else "_" for c in audio_path.stem)[:80]
    return OUTPUT_DIR / f"{safe_stem}.json"


def process_file(model: WhisperModel, audio_path: Path) -> dict:
    start = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        language="sw",
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        compression_ratio_threshold=2.4,
    )

    seg_records = []
    raw_text_parts = []
    for seg in segments:
        text = seg.text.strip()
        raw_text_parts.append(text)
        seg_records.append({
            "start": seg.start,
            "end": seg.end,
            "text": text,
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
            "compression_ratio": seg.compression_ratio,
        })

    raw_text = " ".join(raw_text_parts)
    corrected_text = correct_words(normalize_hypothesis(raw_text))
    elapsed = time.time() - start

    avg_logprobs = [s["avg_logprob"] for s in seg_records]
    no_speech_probs = [s["no_speech_prob"] for s in seg_records]

    return {
        "source_path": str(audio_path),
        "duration_s": info.duration,
        "transcribe_time_s": elapsed,
        "detected_language": info.language,
        "language_probability": info.language_probability,
        "raw_text": raw_text,
        "corrected_text": corrected_text,
        "segments": seg_records,
        # Quality signals for a later filtering pass -- NOT yet filtered here.
        "mean_avg_logprob": sum(avg_logprobs) / len(avg_logprobs) if avg_logprobs else None,
        "mean_no_speech_prob": sum(no_speech_probs) / len(no_speech_probs) if no_speech_probs else None,
        "num_segments": len(seg_records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, help="Directory of source audio files")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N unprocessed files (for a quick pilot)")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser()
    if not source_dir.is_dir():
        sys.exit(f"Not a directory: {source_dir}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    all_files = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    todo = [p for p in all_files if not output_path_for(p).exists()]

    print(f"Found {len(all_files)} audio files in {source_dir}, {len(todo)} not yet pseudo-labeled.")
    if args.limit:
        todo = todo[:args.limit]
        print(f"Limiting to first {len(todo)} for this run.")

    if not todo:
        print("Nothing to do.")
        return

    print("Loading model...", file=sys.stderr)
    model = WhisperModel(str(MODEL_DIR), device="cpu", compute_type="int8")

    for i, audio_path in enumerate(todo, 1):
        out_path = output_path_for(audio_path)
        print(f"[{i}/{len(todo)}] {audio_path.name}")
        try:
            record = process_file(model, audio_path)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  duration={record['duration_s']:.1f}s transcribe_time={record['transcribe_time_s']:.1f}s "
              f"mean_avg_logprob={record['mean_avg_logprob']:.3f} "
              f"mean_no_speech_prob={record['mean_no_speech_prob']:.3f}")

    print(f"\nDone. Pseudo-labels in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
