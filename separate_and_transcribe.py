"""
Audio source separation + transcription comparison for music-affected files.

For each target audio file:
  1. Run demucs htdemucs to isolate the vocals track
  2. Transcribe both original and vocals-only with the Kinyarwanda model
  3. Compute WER against the ground-truth docx for both
  4. Report the difference — this tells us how much music hurts

Usage:
    python separate_and_transcribe.py [file1 file2 ...]
    # defaults to the two worst files: ibyakozwe7 ibyakozwe8
"""

import re
import sys
import time
import subprocess
from pathlib import Path

import docx
import jiwer
from faster_whisper import WhisperModel

from normalize import normalize_hypothesis
from lm_rescore import correct_words

MODEL_DIR   = Path(__file__).parent / "models" / "whisper-large-v3-turbo-kinyarwanda-ct2"
TESTING_DIR = Path(__file__).parent / "testing_data"
SEP_DIR     = Path(__file__).parent / "separated"   # demucs output lands here
NR_DIR      = SEP_DIR / "noisereduce"               # noisereduce output lands here
SEP_DIR.mkdir(exist_ok=True)
NR_DIR.mkdir(exist_ok=True)

DEFAULT_FILES = ["ibyakozwe7", "ibyakozwe8", "ibyakozwe10"]


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[<>«»“”\"]", " ", text)
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def get_reference(docx_path: Path) -> str:
    document = docx.Document(str(docx_path))
    return " ".join(p.text.strip() for p in document.paragraphs if p.text.strip())


def transcribe(model: WhisperModel, audio_path: Path) -> str:
    segments, _ = model.transcribe(
        str(audio_path),
        language="sw",
        task="transcribe",
        beam_size=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=4,
        compression_ratio_threshold=2.4,
    )
    raw = " ".join(seg.text.strip() for seg in segments)
    return correct_words(normalize_hypothesis(raw))


def separate(mp3_path: Path) -> Path | None:
    """Run demucs htdemucs on mp3_path, return path to vocals wav."""
    name = mp3_path.stem
    vocals_path = SEP_DIR / "htdemucs" / name / "vocals.wav"
    if vocals_path.exists():
        print(f"  [demucs] {name}: already separated, skipping.")
        return vocals_path

    print(f"  [demucs] Separating {mp3_path.name} (CPU — takes a few minutes) ...")
    result = subprocess.run(
        ["python3", "-m", "demucs",
         "--two-stems", "vocals",
         "--out", str(SEP_DIR),
         str(mp3_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [demucs] ERROR:\n{result.stderr[-800:]}")
        return None
    if not vocals_path.exists():
        # demucs sometimes uses the mp3 stem differently
        for p in SEP_DIR.rglob("vocals.wav"):
            if name in str(p):
                return p
        print(f"  [demucs] Could not find vocals.wav in {SEP_DIR}")
        return None
    return vocals_path


def denoise(mp3_path: Path) -> Path | None:
    """Run noisereduce spectral gating on mp3_path, return path to denoised wav."""
    import numpy as np
    import soundfile as sf
    import noisereduce as nr

    name = mp3_path.stem
    out_path = NR_DIR / f"{name}_vocals.wav"
    if out_path.exists():
        print(f"  [noisereduce] {name}: already denoised, skipping.")
        return out_path

    print(f"  [noisereduce] Denoising {mp3_path.name} ...")
    samples, sr = sf.read(mp3_path, always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    reduced = nr.reduce_noise(y=samples, sr=sr, stationary=False)
    sf.write(out_path, reduced, sr)
    return out_path


def main():
    names = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_FILES
    pairs = []
    for name in names:
        mp3  = TESTING_DIR / f"{name}.mp3"
        docx_p = TESTING_DIR / f"{name}.docx"
        if not mp3.exists():
            print(f"Skipping {name}: {mp3} not found")
            continue
        if not docx_p.exists():
            print(f"Skipping {name}: {docx_p} not found")
            continue
        pairs.append((name, mp3, docx_p))

    if not pairs:
        print("No valid files found.")
        sys.exit(1)

    # Step 1: separate/denoise all files first (can run while model loads)
    demucs_paths = {}
    nr_paths = {}
    for name, mp3, _ in pairs:
        demucs_paths[name] = separate(mp3)
        nr_paths[name] = denoise(mp3)

    # Step 2: load model once
    print("\nLoading Whisper model ...")
    model = WhisperModel(str(MODEL_DIR), device="cpu", compute_type="int8")

    # Step 3: transcribe + evaluate
    results = []
    for name, mp3, docx_p in pairs:
        ref_norm = normalize(get_reference(docx_p))

        print(f"\n[{name}] transcribing original ...")
        t0 = time.time()
        hyp_orig = transcribe(model, mp3)
        wer_orig = jiwer.wer(ref_norm, normalize(hyp_orig))
        print(f"  original:     WER={wer_orig:.2%}  ({time.time()-t0:.0f}s)")

        wer_demucs = None
        dp = demucs_paths.get(name)
        if dp and dp.exists():
            print(f"[{name}] transcribing demucs vocals ...")
            t0 = time.time()
            hyp_demucs = transcribe(model, dp)
            wer_demucs = jiwer.wer(ref_norm, normalize(hyp_demucs))
            print(f"  demucs:        WER={wer_demucs:.2%}  ({time.time()-t0:.0f}s)")
        else:
            print(f"  [demucs track unavailable for {name}]")

        wer_nr = None
        npth = nr_paths.get(name)
        if npth and npth.exists():
            print(f"[{name}] transcribing noisereduce track ...")
            t0 = time.time()
            hyp_nr = transcribe(model, npth)
            wer_nr = jiwer.wer(ref_norm, normalize(hyp_nr))
            print(f"  noisereduce:   WER={wer_nr:.2%}  ({time.time()-t0:.0f}s)")
        else:
            print(f"  [noisereduce track unavailable for {name}]")

        results.append((name, wer_orig, wer_demucs, wer_nr))

    print("\n" + "="*70)
    print(f"{'File':<15} {'Original':>10} {'Demucs':>10} {'NoiseReduce':>13} {'Best':>10}")
    print("="*70)
    for name, wer_orig, wer_demucs, wer_nr in results:
        candidates = {"original": wer_orig}
        if wer_demucs is not None:
            candidates["demucs"] = wer_demucs
        if wer_nr is not None:
            candidates["noisereduce"] = wer_nr
        best = min(candidates, key=candidates.get)
        fmt = lambda w: f"{w:.2%}" if w is not None else "n/a"
        print(f"{name:<15} {fmt(wer_orig):>10} {fmt(wer_demucs):>10} {fmt(wer_nr):>13} {best:>10}")


if __name__ == "__main__":
    main()
