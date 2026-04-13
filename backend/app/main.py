from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .audio_prep import (
  AudioPreparationError,
  DEFAULT_MAX_LOCAL_UPLOAD_BYTES,
  DEFAULT_SEGMENT_DURATION_SECONDS,
  DEFAULT_SEGMENT_OVERLAP_SECONDS,
  PreparedAudioUpload,
  prepare_audio_uploads,
)
from .openai_audio import (
  OpenAIDiarizedSegment,
  OpenAISettings,
  OpenAITranscriber,
  OpenAITranscriptionError,
  OpenAITranscriptionResult,
)
from .reconcile import (
  format_transcript,
  normalize_speaker_labels,
  normalize_text_parts,
  overlap_duration,
)
from .schemas import (
  HealthResponse,
  SpeakerSegment,
  TranscriptSegment,
  TranscriptTurn,
  TranscriptionResponse,
)

load_dotenv(Path(__file__).resolve().parents[1] / '.env')
logger = logging.getLogger('uvicorn.error').getChild('backend.transcribe')

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
OPENAI_MODE = 'openai'
OPENAI_DEVICE = 'remote'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '').strip()
OPENAI_API_BASE_URL = os.getenv('OPENAI_API_BASE_URL', 'https://api.openai.com/v1').strip().rstrip('/')
OPENAI_TRANSCRIPTION_MODEL = os.getenv('OPENAI_TRANSCRIPTION_MODEL', 'gpt-4o-transcribe-diarize').strip() or 'gpt-4o-transcribe-diarize'
OPENAI_TRANSCRIPTION_LANGUAGE = os.getenv('OPENAI_TRANSCRIPTION_LANGUAGE', '').strip() or None
OPENAI_TIMEOUT_SECONDS = float(os.getenv('OPENAI_TIMEOUT_SECONDS', '600'))
OPENAI_SEGMENT_SECONDS = float(os.getenv('OPENAI_SEGMENT_SECONDS', str(DEFAULT_SEGMENT_DURATION_SECONDS)))
OPENAI_SEGMENT_OVERLAP_SECONDS = float(
  os.getenv('OPENAI_SEGMENT_OVERLAP_SECONDS', str(DEFAULT_SEGMENT_OVERLAP_SECONDS))
)
MAX_UPLOAD_BYTES = int(os.getenv('MAX_UPLOAD_BYTES', str(DEFAULT_MAX_LOCAL_UPLOAD_BYTES)))
CORS_ORIGINS = [
  origin.strip()
  for origin in os.getenv(
    'BACKEND_CORS_ORIGINS',
    'http://localhost:5173,http://127.0.0.1:5173',
  ).split(',')
  if origin.strip()
]

app = FastAPI(title='Audio Transcription API')
app.add_middleware(
  CORSMiddleware,
  allow_origins=CORS_ORIGINS,
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*'],
)


@lru_cache(maxsize=1)
def get_transcriber() -> OpenAITranscriber:
  logger.info(
    'openai transcription init model=%s base_url=%s language=%s',
    OPENAI_TRANSCRIPTION_MODEL,
    OPENAI_API_BASE_URL,
    OPENAI_TRANSCRIPTION_LANGUAGE or 'auto',
  )
  return OpenAITranscriber(
    OpenAISettings(
      api_key=OPENAI_API_KEY,
      api_base_url=OPENAI_API_BASE_URL,
      model=OPENAI_TRANSCRIPTION_MODEL,
      language=OPENAI_TRANSCRIPTION_LANGUAGE,
      timeout_seconds=OPENAI_TIMEOUT_SECONDS,
    )
  )


def build_diarized_payload(
  transcription: OpenAITranscriptionResult,
) -> tuple[list[SpeakerSegment], list[TranscriptSegment], list[TranscriptTurn], str]:
  if not transcription.segments:
    transcript = transcription.transcript.strip()
    return [], [], [], transcript

  raw_speaker_segments = [
    SpeakerSegment(
      speaker=segment.speaker,
      start=segment.start,
      end=segment.end,
    )
    for segment in transcription.segments
  ]
  normalized_speaker_segments, _ = normalize_speaker_labels(raw_speaker_segments)
  transcript_segments: list[TranscriptSegment] = []
  turns: list[TranscriptTurn] = []

  for index, (raw_segment, speaker_segment) in enumerate(
    zip(transcription.segments, normalized_speaker_segments),
    start=1,
  ):
    text = raw_segment.text.strip()

    if not text:
      continue

    transcript_segments.append(
      TranscriptSegment(
        id=index,
        start=speaker_segment.start,
        end=speaker_segment.end,
        text=text,
      )
    )

    if turns and turns[-1].speaker == speaker_segment.speaker:
      turns[-1].end = speaker_segment.end
      turns[-1].text = normalize_text_parts([turns[-1].text, text])
      turns[-1].segment_ids.append(index)
      continue

    turns.append(
      TranscriptTurn(
        speaker=speaker_segment.speaker,
        start=speaker_segment.start,
        end=speaker_segment.end,
        text=text,
        segment_ids=[index],
      )
    )

  transcript = format_transcript(turns) or transcription.transcript.strip()

  return normalized_speaker_segments, transcript_segments, turns, transcript


def merge_chunk_transcriptions(
  chunk_results: list[tuple[PreparedAudioUpload, OpenAITranscriptionResult]],
) -> OpenAITranscriptionResult:
  merged_segments: list[OpenAIDiarizedSegment] = []
  language: str | None = None
  duration_seconds: float | None = None
  next_global_speaker_id = 1

  for chunk_index, (prepared_audio, result) in enumerate(chunk_results, start=1):
    if not language and result.language:
      language = result.language

    chunk_duration_seconds = result.duration_seconds or prepared_audio.duration_seconds

    if chunk_duration_seconds is not None:
      absolute_duration_seconds = prepared_audio.offset_seconds + chunk_duration_seconds
      duration_seconds = max(duration_seconds or 0.0, absolute_duration_seconds)

    absolute_segments = sorted(
      [
        OpenAIDiarizedSegment(
          speaker=segment.speaker,
          start=segment.start + prepared_audio.offset_seconds,
          end=segment.end + prepared_audio.offset_seconds,
          text=segment.text,
        )
        for segment in result.segments
        if segment.text.strip()
      ],
      key=lambda segment: (segment.start, segment.end, segment.speaker, segment.text),
    )
    speaker_map = stitch_chunk_speakers(
      existing_segments=merged_segments,
      current_segments=absolute_segments,
      overlap_start=prepared_audio.offset_seconds,
      overlap_end=prepared_audio.keep_from_seconds,
    )

    for speaker in unique_speaker_order(absolute_segments):
      if speaker in speaker_map:
        continue

      speaker_map[speaker] = f'global_speaker_{next_global_speaker_id:04d}'
      next_global_speaker_id += 1

    for segment in absolute_segments:
      text = segment.text.strip()

      if not text:
        continue

      start = segment.start
      end = segment.end

      if end <= prepared_audio.keep_from_seconds:
        continue

      speaker = speaker_map.get(segment.speaker)

      if not speaker:
        speaker = f'global_speaker_{next_global_speaker_id:04d}'
        speaker_map[segment.speaker] = speaker
        next_global_speaker_id += 1

      previous_segment = find_recent_segment_for_speaker(merged_segments, speaker)

      if start < prepared_audio.keep_from_seconds:
        start = prepared_audio.keep_from_seconds
        text = trim_repeated_prefix(previous_segment.text if previous_segment else '', text)
      elif (
        previous_segment
        and previous_segment.speaker == speaker
        and prepared_audio.keep_from_seconds > prepared_audio.offset_seconds
        and previous_segment.end >= prepared_audio.keep_from_seconds - OPENAI_SEGMENT_OVERLAP_SECONDS
        and start <= prepared_audio.keep_from_seconds + OPENAI_SEGMENT_OVERLAP_SECONDS
        and start - previous_segment.end <= 5.0
      ):
        text = trim_repeated_prefix(previous_segment.text, text)

      if not text:
        continue

      merged_segments.append(
        OpenAIDiarizedSegment(
          speaker=speaker,
          start=start,
          end=end,
          text=text,
        )
      )

    logger.info(
      'stitched chunk=%d overlap_start=%.3f keep_from=%.3f merged_segments=%d',
      chunk_index,
      prepared_audio.offset_seconds,
      prepared_audio.keep_from_seconds,
      len(merged_segments),
    )

  merged_segments = sorted(
    merged_segments,
    key=lambda segment: (segment.start, segment.end, segment.speaker, segment.text),
  )
  transcript = '\n\n'.join(segment.text.strip() for segment in merged_segments if segment.text.strip()).strip()

  return OpenAITranscriptionResult(
    transcript=transcript,
    duration_seconds=duration_seconds,
    language=language,
    segments=merged_segments,
  )


def normalize_compare_token(token: str) -> str:
  return re.sub(r"[^\w']+", '', token.lower())


def trim_repeated_prefix(previous_text: str, current_text: str) -> str:
  previous_words = previous_text.strip().split()
  current_words = current_text.strip().split()

  if not previous_words or not current_words:
    return current_text.strip()

  normalized_previous_words = [normalize_compare_token(word) for word in previous_words]
  normalized_current_words = [normalize_compare_token(word) for word in current_words]
  max_overlap_words = min(len(previous_words), len(current_words), 24)

  for overlap_words in range(max_overlap_words, 0, -1):
    previous_slice = normalized_previous_words[-overlap_words:]
    current_slice = normalized_current_words[:overlap_words]

    if previous_slice != current_slice:
      continue

    trimmed_text = ' '.join(current_words[overlap_words:]).strip()
    return trimmed_text

  return current_text.strip()


def unique_speaker_order(segments: list[OpenAIDiarizedSegment]) -> list[str]:
  speakers: list[str] = []

  for segment in segments:
    if segment.speaker not in speakers:
      speakers.append(segment.speaker)

  return speakers


def find_recent_segment_for_speaker(
  segments: list[OpenAIDiarizedSegment],
  speaker: str,
) -> OpenAIDiarizedSegment | None:
  for segment in reversed(segments):
    if segment.speaker == speaker:
      return segment

  return None


def stitch_chunk_speakers(
  *,
  existing_segments: list[OpenAIDiarizedSegment],
  current_segments: list[OpenAIDiarizedSegment],
  overlap_start: float,
  overlap_end: float,
) -> dict[str, str]:
  if overlap_end <= overlap_start or not existing_segments or not current_segments:
    return {}

  relevant_existing_segments = [
    segment
    for segment in existing_segments
    if segment.end > overlap_start and segment.start < overlap_end
  ]
  relevant_current_segments = [
    segment
    for segment in current_segments
    if segment.end > overlap_start and segment.start < overlap_end
  ]

  if not relevant_existing_segments or not relevant_current_segments:
    return {}

  pair_scores: dict[tuple[str, str], float] = {}
  local_totals: dict[str, float] = {}

  for current_segment in relevant_current_segments:
    current_window_start = max(current_segment.start, overlap_start)
    current_window_end = min(current_segment.end, overlap_end)
    current_overlap_seconds = current_window_end - current_window_start

    if current_overlap_seconds <= 0:
      continue

    local_totals[current_segment.speaker] = (
      local_totals.get(current_segment.speaker, 0.0) + current_overlap_seconds
    )

    for existing_segment in relevant_existing_segments:
      existing_window_start = max(existing_segment.start, overlap_start)
      existing_window_end = min(existing_segment.end, overlap_end)
      shared_seconds = overlap_duration(
        current_window_start,
        current_window_end,
        existing_window_start,
        existing_window_end,
      )

      if shared_seconds <= 0:
        continue

      pair_key = (current_segment.speaker, existing_segment.speaker)
      pair_scores[pair_key] = pair_scores.get(pair_key, 0.0) + shared_seconds

  speaker_map: dict[str, str] = {}
  used_global_speakers: set[str] = set()

  for (local_speaker, global_speaker), score in sorted(
    pair_scores.items(),
    key=lambda item: item[1],
    reverse=True,
  ):
    if local_speaker in speaker_map or global_speaker in used_global_speakers:
      continue

    local_total = local_totals.get(local_speaker, 0.0)

    if local_total <= 0:
      continue

    confidence = score / local_total

    if score < 0.25 or confidence < 0.5:
      continue

    speaker_map[local_speaker] = global_speaker
    used_global_speakers.add(global_speaker)

  return speaker_map


@app.get('/api/health', response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
  return HealthResponse(
    status='ok' if OPENAI_API_KEY else 'misconfigured',
    mode=OPENAI_MODE,
    model=OPENAI_TRANSCRIPTION_MODEL,
    device=OPENAI_DEVICE,
  )


@app.post('/api/transcribe', response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)) -> TranscriptionResponse:
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

  if len(payload) > MAX_UPLOAD_BYTES:
    raise HTTPException(
      status_code=400,
      detail='The uploaded file exceeds the server upload limit.',
    )

  try:
    prepared_uploads = prepare_audio_uploads(
      filename=file.filename,
      payload=payload,
      content_type=content_type,
      max_segment_seconds=OPENAI_SEGMENT_SECONDS,
      overlap_seconds=OPENAI_SEGMENT_OVERLAP_SECONDS,
    )
    chunk_results: list[tuple[PreparedAudioUpload, OpenAITranscriptionResult]] = []

    for index, prepared_audio in enumerate(prepared_uploads, start=1):
      logger.info(
        'transcribe chunk file=%s model=%s chunk=%d/%d bytes=%d prepared_bytes=%d offset=%.3f transcoded=%s',
        file.filename,
        OPENAI_TRANSCRIPTION_MODEL,
        index,
        len(prepared_uploads),
        len(payload),
        len(prepared_audio.payload),
        prepared_audio.offset_seconds,
        prepared_audio.transcoded,
      )
      try:
        chunk_transcription = await get_transcriber().transcribe_bytes(
          filename=prepared_audio.filename,
          payload=prepared_audio.payload,
          content_type=prepared_audio.content_type,
        )
      except OpenAITranscriptionError as exc:
        raise HTTPException(
          status_code=exc.status_code,
          detail=f'Chunk {index} of {len(prepared_uploads)} failed: {exc}',
        ) from exc

      chunk_results.append((prepared_audio, chunk_transcription))

    transcription = merge_chunk_transcriptions(chunk_results)

    if not transcription.transcript and not transcription.segments:
      raise HTTPException(status_code=502, detail='OpenAI returned an empty transcript.')

    speaker_segments, segments, turns, transcript = build_diarized_payload(transcription)
    response = TranscriptionResponse(
      filename=file.filename,
      mode=OPENAI_MODE,
      model=OPENAI_TRANSCRIPTION_MODEL,
      language=transcription.language,
      duration_seconds=transcription.duration_seconds,
      diarization_enabled=bool(speaker_segments),
      diarization_model=OPENAI_TRANSCRIPTION_MODEL,
      diarization_device=OPENAI_DEVICE,
      speaker_segments_source='openai-diarized_json',
      speaker_segments=speaker_segments,
      transcript=transcript,
      segments=segments,
      turns=turns,
    )
    logger.info(
      'transcribe done file=%s model=%s turns=%d speakers=%d',
      file.filename,
      OPENAI_TRANSCRIPTION_MODEL,
      len(turns),
      len({turn.speaker for turn in turns if turn.speaker}),
    )
    return response
  except HTTPException:
    raise
  except AudioPreparationError as exc:
    raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
  except RuntimeError as exc:
    raise HTTPException(status_code=500, detail=str(exc) or 'OpenAI transcription failed.') from exc
  except Exception as exc:
    logger.exception('unexpected transcription failure for %s', file.filename)
    raise HTTPException(status_code=500, detail='Unexpected transcription error.') from exc
  finally:
    await file.close()
