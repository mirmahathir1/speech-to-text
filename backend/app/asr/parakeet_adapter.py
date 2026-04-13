from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any

from ..schemas import TranscriptSegment, TranscriptWord
from .base import AsrTranscription

logger = logging.getLogger('uvicorn.error').getChild('backend.transcribe')


@dataclass(frozen=True)
class ParakeetSettings:
  model: str
  device: str


def build_transcript_text(segments: list[TranscriptSegment], fallback_text: str) -> str:
  transcript = ' '.join(segment.text for segment in segments if segment.text).strip()

  if transcript:
    return transcript

  return fallback_text.strip()


def get_time_stride(model: Any) -> float | None:
  try:
    return float(8 * model.cfg.preprocessor.window_stride)
  except (AttributeError, TypeError, ValueError):
    return None


def stamp_times(stamp: dict[str, Any], time_stride: float | None) -> tuple[float | None, float | None]:
  if stamp.get('start') is not None and stamp.get('end') is not None:
    return float(stamp['start']), float(stamp['end'])

  if (
    time_stride is not None
    and stamp.get('start_offset') is not None
    and stamp.get('end_offset') is not None
  ):
    return (
      float(stamp['start_offset']) * time_stride,
      float(stamp['end_offset']) * time_stride,
    )

  return None, None


def build_words(timestamp_data: dict[str, Any], time_stride: float | None) -> list[TranscriptWord]:
  words: list[TranscriptWord] = []

  for raw_word in timestamp_data.get('word') or []:
    text = str(
      raw_word.get('word')
      or raw_word.get('char')
      or raw_word.get('text')
      or ''
    ).strip()

    if not text:
      continue

    start, end = stamp_times(raw_word, time_stride)
    words.append(
      TranscriptWord(
        word=text,
        start=start,
        end=end,
      )
    )

  return words


def segment_words(words: list[TranscriptWord], start: float, end: float) -> list[TranscriptWord]:
  matched_words: list[TranscriptWord] = []

  for word in words:
    word_start = word.start
    word_end = word.end if word.end is not None else word.start

    if word_start is None or word_end is None:
      continue

    if max(start, word_start) < min(end, word_end):
      matched_words.append(word)

  return matched_words


def build_segments(
  hypothesis: Any,
  words: list[TranscriptWord],
  time_stride: float | None,
) -> list[TranscriptSegment]:
  timestamp_data = getattr(hypothesis, 'timestamp', {}) or {}
  raw_segments = timestamp_data.get('segment') or []
  segments: list[TranscriptSegment] = []

  for index, raw_segment in enumerate(raw_segments):
    start, end = stamp_times(raw_segment, time_stride)

    if start is None:
      continue

    if end is None or end < start:
      end = start

    matched_words = segment_words(words, start, end)
    text = str(raw_segment.get('segment') or '').strip()

    if not text and matched_words:
      text = ' '.join(word.word for word in matched_words).strip()

    if not text:
      continue

    segments.append(
      TranscriptSegment(
        id=index,
        start=start,
        end=end,
        text=text,
        words=matched_words,
      )
    )

  return segments


def build_fallback_segment(
  transcript_text: str,
  words: list[TranscriptWord],
) -> list[TranscriptSegment]:
  if not transcript_text:
    return []

  known_word_times = [
    (word.start, word.end if word.end is not None else word.start)
    for word in words
    if word.start is not None
  ]

  if known_word_times:
    start = min(start for start, _ in known_word_times if start is not None)
    end = max(end for _, end in known_word_times if end is not None)
  else:
    start = 0.0
    end = 0.0

  return [
    TranscriptSegment(
      id=0,
      start=float(start),
      end=float(end),
      text=transcript_text,
      words=words,
    )
  ]


@lru_cache(maxsize=1)
def get_parakeet_model(model: str, device: str):
  try:
    import torch
    import nemo.collections.asr as nemo_asr
  except ImportError as exc:
    raise RuntimeError('NeMo ASR dependencies are not installed.') from exc

  logger.info('load asr provider=parakeet model=%s device=%s', model, device)
  asr_model = nemo_asr.models.ASRModel.from_pretrained(model)
  asr_model = asr_model.to(torch.device(device))
  asr_model.eval()
  return asr_model


class ParakeetAdapter:
  def __init__(self, settings: ParakeetSettings):
    self.settings = settings

  def transcribe_file(self, audio_path: Path, *, language: str | None = None) -> AsrTranscription:
    del language

    model = get_parakeet_model(
      self.settings.model,
      self.settings.device,
    )
    hypotheses = model.transcribe([str(audio_path)], timestamps=True)

    if not hypotheses:
      return AsrTranscription(transcript='')

    hypothesis = hypotheses[0]
    transcript_text = str(getattr(hypothesis, 'text', '') or '').strip()
    time_stride = get_time_stride(model)
    timestamp_data = getattr(hypothesis, 'timestamp', {}) or {}
    words = build_words(timestamp_data, time_stride)
    segments = build_segments(hypothesis, words, time_stride)

    if not segments:
      segments = build_fallback_segment(transcript_text, words)

    duration_seconds = max(
      (segment.end for segment in segments),
      default=max((word.end for word in words if word.end is not None), default=None),
    )

    return AsrTranscription(
      transcript=build_transcript_text(segments, transcript_text),
      language='en',
      duration_seconds=duration_seconds,
      segments=segments,
    )
