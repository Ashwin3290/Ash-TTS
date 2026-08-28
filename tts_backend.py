"""
FastAPI backend for TTS generation.
Streamlit calls this to generate audio chunks.

Usage:
    python tts_backend.py        # starts on localhost:8000
"""

import os
import sys

if sys.platform == "win32":
    os.environ.setdefault(
        "PHONEMIZER_ESPEAK_LIBRARY",
        r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    )

import torch
import json
import numpy as np
from pathlib import Path
from io import BytesIO
import base64
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import soundfile as sf

from config import audio as acfg, hifigan as hcfg, paths
from model.fastspeech2 import FastSpeech2, load_fs2_state
from vocoder.generator import Generator, config_from_hcfg
from inference import text_to_phonemes, load_phoneme_vocab

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

FS2_CKPT = Path("checkpoints/fastspeech2/best.pt")
HIFI_CKPT = Path("checkpoints/hifigan/g_best.pt")
FS2_REPO = "Ashwin-C9/tts-fastspeech2-ckpt"
HIFI_REPO = "Ashwin-C9/tts-hifigan-ckpt"

device = None
fs2 = None
hifi = None
vocab = None


class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    pitch: float = 1.0
    energy: float = 1.0


def ensure_checkpoints():
    for local, repo, fname in [(FS2_CKPT, FS2_REPO, "best.pt"),
                                (HIFI_CKPT, HIFI_REPO, "g_best.pt")]:
        if not local.exists():
            print(f"Downloading {fname} from {repo}...")
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(repo, fname, repo_type="model")
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(Path(downloaded).read_bytes())


def load_models():
    global device, fs2, hifi, vocab
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ensure_checkpoints()
    vocab = load_phoneme_vocab()

    with open(paths.processed_dir / "stats.json") as f:
        stats = json.load(f)
    fs2 = FastSpeech2().to(device)
    fs2.variance_adaptor.set_stats(**stats)
    ckpt = torch.load(FS2_CKPT, map_location=device)
    load_fs2_state(fs2, ckpt["model"])
    fs2.eval()
    print(f"FastSpeech2: step {ckpt.get('step', '?')}")

    hifi = Generator(config_from_hcfg(hcfg)).to(device)
    hifi_ckpt = torch.load(HIFI_CKPT, map_location=device)
    hifi.load_state_dict(hifi_ckpt["generator"])
    hifi.eval()
    hifi.remove_weight_norm()
    print(f"HiFi-GAN:    step {hifi_ckpt.get('step', '?')}")


def text_to_wav(text, speed=1.0, pitch=1.0, energy=1.0):
    """Convert text to audio using FastSpeech2 + HiFi-GAN."""
    logger.debug(f"text_to_wav: input text='{text}'")

    try:
        logger.debug("Converting text to phonemes...")
        phonemes = text_to_phonemes(text, vocab)
        logger.debug(f"Phonemes: {phonemes}")

        if not phonemes:
            logger.warning("Empty phonemes from text_to_phonemes")
            return None

        logger.debug(f"Creating tensor from {len(phonemes)} phonemes...")
        ph = torch.tensor(phonemes, dtype=torch.long).unsqueeze(0).to(device)
        ph_lens = torch.tensor([len(phonemes)], dtype=torch.long).to(device)
        logger.debug(f"ph shape: {ph.shape}, ph_lens: {ph_lens}")

        logger.debug("Running FastSpeech2...")
        with torch.no_grad():
            _, mel_pred, _, _, _, mel_lens = fs2(
                ph, ph_lens,
                duration_scale=1.0 / speed,
                pitch_scale=pitch,
                energy_scale=energy,
            )
            logger.debug(f"mel_pred shape: {mel_pred.shape}, mel_lens: {mel_lens}")

            mel = mel_pred[:, :mel_lens[0].item()]
            logger.debug(f"mel trimmed shape: {mel.shape}")

            mel = (mel + 1) / 2 * (acfg.mel_max - acfg.mel_min) + acfg.mel_min
            logger.debug(f"mel normalized: min={mel.min():.4f}, max={mel.max():.4f}")

            logger.debug("Running HiFi-GAN vocoder...")
            wav = hifi(mel.transpose(1, 2)).squeeze().cpu().numpy()
            logger.debug(f"WAV generated: shape={wav.shape}, dtype={wav.dtype}")

        return wav
    except Exception as e:
        logger.error(f"Error in text_to_wav: {type(e).__name__}: {str(e)}", exc_info=True)
        raise


@app.on_event("startup")
async def startup_event():
    print("Loading TTS models...")
    load_models()
    print("TTS backend ready!")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/synthesize")
async def synthesize(req: TTSRequest):
    """Synthesize audio from text and return as base64-encoded WAV."""
    logger.info(f"Synthesize request: text='{req.text}', speed={req.speed}, pitch={req.pitch}, energy={req.energy}")

    if not req.text.strip():
        logger.warning("Empty text received")
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        logger.debug("Calling text_to_wav...")
        wav = text_to_wav(req.text, speed=req.speed, pitch=req.pitch, energy=req.energy)

        if wav is None:
            logger.error("text_to_wav returned None")
            raise HTTPException(status_code=400, detail="Failed to synthesize")

        logger.debug(f"Generated WAV: shape={wav.shape}, dtype={wav.dtype}, min={wav.min():.4f}, max={wav.max():.4f}")

        logger.debug("Encoding to WAV bytes...")
        wav_bytes = BytesIO()
        sf.write(wav_bytes, wav, acfg.sample_rate, format="WAV")
        wav_bytes.seek(0)

        logger.debug(f"WAV bytes size: {len(wav_bytes.getvalue())} bytes")

        logger.debug("Encoding to base64...")
        wav_b64 = base64.b64encode(wav_bytes.getvalue()).decode()

        duration = len(wav) / acfg.sample_rate
        logger.info(f"Synthesis successful: {duration:.2f}s audio generated")

        return {
            "audio": wav_b64,
            "duration": duration
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in synthesize: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
