from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from pathlib import Path

from faster_whisper import WhisperModel

from ..schemas import TranscriptSegment, TranscriptWord
from .base import AsrTranscription

logger = logging.getLogger('uvicorn.error').getChild('backend.transcribe')


@dataclass(frozen=True)
class FasterWhisperSettings:
  model: str
  device: str
  compute_type: str
  download_root: str
  beam_size: int = 5
  word_timestamps: bool = True


def build_transcript_text(segments: list[TranscriptSegment]) -> str:
  return ' '.join(segment.text for segment in segments).strip()


def build_words(raw_words: object) -> list[TranscriptWord]:
  words: list[TranscriptWord] = []

  for raw_word in raw_words or []:
    word_text = str(getattr(raw_word, 'word', '') or '').strip()

    if not word_text:
      continue

    words.append(
      TranscriptWord(
        word=word_text,
        start=float(raw_word.start) if getattr(raw_word, 'start', None) is not None else None,
        end=float(raw_word.end) if getattr(raw_word, 'end', None) is not None else None,
        probability=float(raw_word.probability) if getattr(raw_word, 'probability', None) is not None else None,
      )
    )

  return words


def resolve_segment_end(start: float, end: float | None, words: list[TranscriptWord]) -> float:
  if end is not None:
    return float(end)

  word_end = max((word.end for word in words if word.end is not None), default=None)
  return float(word_end) if word_end is not None else start


def build_segment(raw_segment: object, index: int) -> TranscriptSegment:
  words = build_words(getattr(raw_segment, 'words', None))
  start = float(getattr(raw_segment, 'start', 0.0) or 0.0)
  end = resolve_segment_end(start, getattr(raw_segment, 'end', None), words)

  return TranscriptSegment(
    id=index,
    start=start,
    end=end,
    text=str(getattr(raw_segment, 'text', '') or '').strip(),
    avg_logprob=float(raw_segment.avg_logprob) if getattr(raw_segment, 'avg_logprob', None) is not None else None,
    compression_ratio=float(raw_segment.compression_ratio) if getattr(raw_segment, 'compression_ratio', None) is not None else None,
    no_speech_prob=float(raw_segment.no_speech_prob) if getattr(raw_segment, 'no_speech_prob', None) is not None else None,
    temperature=float(raw_segment.temperature) if getattr(raw_segment, 'temperature', None) is not None else None,
    words=words,
  )


@lru_cache(maxsize=1)
def get_faster_whisper_model(
  model: str,
  device: str,
  compute_type: str,
  download_root: str,
) -> WhisperModel:
  logger.info(
    'load asr provider=faster-whisper model=%s device=%s compute=%s',
    model,
    device,
    compute_type,
  )
  return WhisperModel(
    model,
    device=device,
    compute_type=compute_type,
    download_root=download_root,
  )


class FasterWhisperAdapter:
  def __init__(self, settings: FasterWhisperSettings):
    self.settings = settings

  def transcribe_file(self, audio_path: Path, *, language: str | None = None) -> AsrTranscription:
    model = get_faster_whisper_model(
      self.settings.model,
      self.settings.device,
      self.settings.compute_type,
      self.settings.download_root,
    )
    segments_iter, info = model.transcribe(
      str(audio_path),
      beam_size=self.settings.beam_size,
      language=language,
      word_timestamps=self.settings.word_timestamps,
    )
    segments = [
      segment
      for segment in (
        build_segment(raw_segment, index)
        for index, raw_segment in enumerate(segments_iter)
      )
      if segment.text
    ]
    duration_seconds = float(info.duration) if getattr(info, 'duration', None) is not None else None

    return AsrTranscription(
      transcript=build_transcript_text(segments),
      language=str(info.language) if getattr(info, 'language', None) else language,
      duration_seconds=duration_seconds,
      segments=segments,
    )
