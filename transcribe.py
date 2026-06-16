#!/usr/bin/env python3
"""Kinyarwanda speech-to-text CLI using a CTranslate2-optimized Whisper model."""

import argparse
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

MODEL_DIR = Path(__file__).parent / "models" / "whisper-large-v3-turbo-kinyarwanda-ct2"


def format_timestamp(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def main():
    parser = argparse.ArgumentParser(description="Transcribe Kinyarwanda speech from an audio file.")
    parser.add_argument("audio", help="Path to the audio file (wav, mp3, m4a, ogg, flac, ...)")
    parser.add_argument("-o", "--output", help="Write transcript to this file instead of stdout")
    parser.add_argument("--timestamps", action="store_true", help="Include per-segment timestamps")
    parser.add_argument("--no-vad", action="store_true", help="Disable voice-activity-detection filtering")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding (default: 5)")
    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=["int8", "int8_float32", "float32"],
        help="Quantization/compute type (default: int8, fastest on CPU)",
    )
    parser.add_argument("--cpu-threads", type=int, default=0, help="Number of CPU threads (0 = auto)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if not (MODEL_DIR / "model.bin").exists():
        print(f"Error: model not found at {MODEL_DIR}. The model download may still be in progress.", file=sys.stderr)
        sys.exit(1)

    model = WhisperModel(
        str(MODEL_DIR),
        device="cpu",
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
    )

    start = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        # The model repurposes Whisper's "sw" (Swahili) language token for Kinyarwanda.
        language="sw",
        task="transcribe",
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        vad_parameters=dict(min_silence_duration_ms=300),
        # Prevent hallucination loops
        condition_on_previous_text=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        compression_ratio_threshold=2.4,
    )

    lines = []
    for seg in segments:
        text = seg.text.strip()
        if args.timestamps:
            lines.append(f"[{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}] {text}")
        else:
            lines.append(text)

    elapsed = time.time() - start
    output_text = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(output_text + "\n", encoding="utf-8")
        print(f"Transcript written to {args.output}", file=sys.stderr)
    else:
        print(output_text)

    print(
        f"\n--- audio duration: {info.duration:.1f}s | transcribed in {elapsed:.1f}s "
        f"| detected language: {info.language} (p={info.language_probability:.2f}) ---",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
