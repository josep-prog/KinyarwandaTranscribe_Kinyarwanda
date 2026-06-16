# Kinyarwanda Speech-to-Text

A fast, CPU-friendly Kinyarwanda transcriber built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(CTranslate2) using the [leophill/whisper-large-v3-turbo-sw-kinyarwanda-ct2](https://huggingface.co/leophill/whisper-large-v3-turbo-sw-kinyarwanda-ct2)
model — a Whisper large-v3-turbo model fine-tuned on Digital Umuganda's 500-hour
multi-domain Kinyarwanda dataset (health, government, finance, education, agriculture).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The model files live in `models/whisper-large-v3-turbo-kinyarwanda-ct2/` and are
downloaded separately (see below) since they're too large for git.

## Usage

```bash
source venv/bin/activate
python transcribe.py path/to/audio.wav
```

Supported audio formats: wav, mp3, m4a, ogg, flac, and anything else PyAV can decode.

### Options

- `-o, --output FILE` — write the transcript to a file instead of stdout
- `--timestamps` — include per-segment start/end timestamps
- `--no-vad` — disable voice-activity-detection filtering (useful for very quiet recordings)
- `--beam-size N` — decoding beam size (default 5; lower = faster, higher = more accurate)
- `--compute-type {int8,int8_float32,float32}` — quantization (default `int8`, fastest on CPU)
- `--cpu-threads N` — number of CPU threads (default 0 = auto)

### Examples

```bash
# Basic transcription
python transcribe.py recording.mp3

# With timestamps, saved to a file
python transcribe.py interview.wav --timestamps -o transcript.txt

# Faster but slightly less accurate
python transcribe.py speech.m4a --beam-size 1
```
