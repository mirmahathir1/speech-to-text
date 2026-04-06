from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import whisper

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

ALLOWED_EXTENSIONS = {
  '.aac',
  '.flac',
  '.m4a',
  '.mp3',
  '.mp4',
  '.mpeg',
  '.mpga',
  '.oga',
  '.ogg',
  '.wav',
  '.webm',
}

WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')
WHISPER_DEVICE = os.getenv('WHISPER_DEVICE', 'cpu')
WHISPER_LANGUAGE = os.getenv('WHISPER_LANGUAGE', '').strip() or None
WHISPER_DOWNLOAD_ROOT = os.getenv(
  'WHISPER_DOWNLOAD_ROOT',
  str(Path(__file__).resolve().parents[1] / '.cache' / 'whisper'),
)
CORS_ORIGINS = [
  origin.strip()
  for origin in os.getenv(
    'BACKEND_CORS_ORIGINS',
    'http://localhost:5173,http://127.0.0.1:5173',
  ).split(',')
  if origin.strip()
]
MODEL_LOCK = Lock()
Path(WHISPER_DOWNLOAD_ROOT).mkdir(parents=True, exist_ok=True)

app = FastAPI(title='Audio Transcription API')
app.add_middleware(
  CORSMiddleware,
  allow_origins=CORS_ORIGINS,
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)


@lru_cache(maxsize=1)
def get_whisper_model():
  return whisper.load_model(
    WHISPER_MODEL,
    device=WHISPER_DEVICE,
    download_root=WHISPER_DOWNLOAD_ROOT,
  )


@app.get('/api/health')
async def healthcheck() -> dict[str, str]:
  return {
    'status': 'ok',
    'mode': 'local-whisper',
    'model': WHISPER_MODEL,
    'device': WHISPER_DEVICE,
  }


@app.post('/api/transcribe')
async def transcribe_audio(file: UploadFile = File(...)) -> dict[str, str]:
  if not file.filename:
    raise HTTPException(status_code=400, detail='Missing uploaded filename.')

  extension = Path(file.filename).suffix.lower()
  content_type = file.content_type or ''

  if extension not in ALLOWED_EXTENSIONS and not content_type.startswith('audio/'):
    raise HTTPException(status_code=400, detail='Unsupported file type. Please upload an audio file.')

  try:
    payload = await file.read()
  except Exception as exc:
    raise HTTPException(status_code=400, detail='Failed to read the uploaded file.') from exc

  if not payload:
    raise HTTPException(status_code=400, detail='The uploaded file is empty.')

  temp_path: Path | None = None

  try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension or '.audio') as temp_file:
      temp_file.write(payload)
      temp_path = Path(temp_file.name)

    transcription_options = {
      'fp16': WHISPER_DEVICE.startswith('cuda'),
    }

    if WHISPER_LANGUAGE:
      transcription_options['language'] = WHISPER_LANGUAGE

    with MODEL_LOCK:
      transcription = get_whisper_model().transcribe(
        str(temp_path),
        **transcription_options,
      )

    transcript_text = str(transcription.get('text', '')).strip()

    if not transcript_text:
      raise HTTPException(status_code=502, detail='Local Whisper returned an empty transcript.')

    return {
      'filename': file.filename,
      'mode': 'local-whisper',
      'model': WHISPER_MODEL,
      'transcript': transcript_text,
    }
  except HTTPException:
    raise
  except RuntimeError as exc:
    raise HTTPException(status_code=500, detail=str(exc) or 'Local Whisper failed to load or transcribe audio.') from exc
  except Exception as exc:
    raise HTTPException(status_code=500, detail='Unexpected local transcription error.') from exc
  finally:
    if temp_path:
      temp_path.unlink(missing_ok=True)

    await file.close()
