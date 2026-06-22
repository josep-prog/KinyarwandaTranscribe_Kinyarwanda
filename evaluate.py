#!/usr/bin/env python3
"""Evaluate transcribe.py output against hand-typed ground-truth transcripts.

For each <testdir>/<name>.<audio ext>, reads the matching <testdir>/<name>.docx
or <name>.txt as the reference transcript, runs the Kinyarwanda model on the
audio, and reports Word Error Rate (WER) and Character Error Rate (CER)
between the two.

By default runs against testing_data/. Pass --testdir to run against a
different set (e.g. a held-out general-speech eval set), which writes its
report to evaluation_report_<testdir name>.txt instead of overwriting the
default report.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import docx
import jiwer
from faster_whisper import WhisperModel
from normalize import normalize_hypothesis
from lm_rescore import correct_words

MODEL_DIR = Path(__file__).parent / "models" / "whisper-large-v3-turbo-kinyarwanda-ct2"
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


def get_reference_text(ref_path: Path) -> str:
    if ref_path.suffix == ".txt":
        return ref_path.read_text(encoding="utf-8").strip()
    document = docx.Document(str(ref_path))
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    return " ".join(paragraphs)


def normalize(text: str) -> str:
    text = text.lower()
    # unify curly apostrophes/quotes to straight ones
    text = text.replace("’", "'").replace("‘", "'")
    # drop literary quotation markers («», <<>>, "")
    text = re.sub(r"[<>«»“”\"]", " ", text)
    # drop punctuation except apostrophes (meaningful for Kinyarwanda elisions)
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_reference(audio_path: Path) -> Path | None:
    for ext in (".docx", ".txt"):
        ref_path = audio_path.with_suffix(ext)
        if ref_path.exists():
            return ref_path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testdir", default="testing_data",
                         help="Directory of audio files + matching .docx/.txt references")
    args = parser.parse_args()

    testing_dir = Path(__file__).parent / args.testdir
    report_path = (Path(__file__).parent / "evaluation_report.txt" if args.testdir == "testing_data"
                   else Path(__file__).parent / f"evaluation_report_{testing_dir.name}.txt")

    pairs = []
    audio_paths = [p for p in testing_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS]
    for audio_path in sorted(audio_paths):
        ref_path = find_reference(audio_path)
        if ref_path:
            pairs.append((audio_path, ref_path))
        else:
            print(f"Skipping {audio_path.name}: no matching .docx/.txt reference", file=sys.stderr)

    pairs.sort(key=lambda p: int(re.search(r"(\d+)", p[0].stem).group()))

    print("Loading model...", file=sys.stderr)
    model = WhisperModel(str(MODEL_DIR), device="cpu", compute_type="int8")

    report_lines = []
    total_wer = 0.0
    total_cer = 0.0
    n = 0

    for audio_path, ref_path in pairs:
        name = audio_path.stem
        reference = get_reference_text(ref_path)
        ref_norm = normalize(reference)

        start = time.time()
        segments, info = model.transcribe(
            str(audio_path),
            # The model repurposes Whisper's "sw" (Swahili) language token for Kinyarwanda.
            language="sw",
            task="transcribe",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            # Prevent hallucination loops
            condition_on_previous_text=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            compression_ratio_threshold=2.4,
        )
        hypothesis = " ".join(seg.text.strip() for seg in segments)
        elapsed = time.time() - start
        hypothesis = normalize_hypothesis(hypothesis)
        hypothesis = correct_words(hypothesis)
        hyp_norm = normalize(hypothesis)

        wer = jiwer.wer(ref_norm, hyp_norm)
        cer = jiwer.cer(ref_norm, hyp_norm)
        total_wer += wer
        total_cer += cer
        n += 1

        print(f"[{name}] duration={info.duration:.1f}s transcribe_time={elapsed:.1f}s "
              f"WER={wer:.1%} CER={cer:.1%}")

        report_lines.append(f"===== {name} =====")
        report_lines.append(f"Audio duration: {info.duration:.1f}s | Transcribe time: {elapsed:.1f}s")
        report_lines.append(f"WER: {wer:.2%} | CER: {cer:.2%}")
        report_lines.append("")
        report_lines.append("--- REFERENCE (normalized) ---")
        report_lines.append(ref_norm)
        report_lines.append("")
        report_lines.append("--- HYPOTHESIS (normalized) ---")
        report_lines.append(hyp_norm)
        report_lines.append("")
        report_lines.append("--- HYPOTHESIS (raw model output) ---")
        report_lines.append(hypothesis)
        report_lines.append("")
        report_lines.append("")

    avg_wer = total_wer / n if n else 0.0
    avg_cer = total_cer / n if n else 0.0

    summary = f"\n===== OVERALL ({n} files) =====\nAverage WER: {avg_wer:.2%}\nAverage CER: {avg_cer:.2%}\n"
    print(summary)
    report_lines.insert(0, summary)

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Full report written to {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
