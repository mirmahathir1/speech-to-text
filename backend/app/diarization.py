from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import logging
from pathlib import Path
from typing import Any

from .schemas import SpeakerSegment

logger = logging.getLogger('uvicorn.error').getChild('backend.transcribe')


@dataclass(frozen=True)
class PyannoteSettings:
  model: str
  device: str
  token: str | None = None
  min_speakers: int | None = None
  max_speakers: int | None = None


@dataclass
class DiarizationResult:
  enabled: bool
  model: str | None = None
  device: str | None = None
  segments_source: str | None = None
  speaker_segments: list[SpeakerSegment] = field(default_factory=list)


def normalize_speaker_segments(annotation: Any) -> list[SpeakerSegment]:
  if annotation is None:
    return []

  segments: list[SpeakerSegment] = []

  if hasattr(annotation, 'itertracks'):
    iterator = annotation.itertracks(yield_label=True)

    for item in iterator:
      if len(item) == 3:
        turn, _, speaker = item
      elif len(item) == 2:
        turn, speaker = item
      else:
        continue

      start = float(turn.start)
      end = float(turn.end)

      if end <= start:
        continue

      segments.append(
        SpeakerSegment(
          speaker=str(speaker),
          start=start,
          end=end,
        )
      )

    return sorted(segments, key=lambda segment: (segment.start, segment.end, segment.speaker))

  for turn, speaker in annotation:
    start = float(turn.start)
    end = float(turn.end)

    if end <= start:
      continue

    segments.append(
      SpeakerSegment(
        speaker=str(speaker),
        start=start,
        end=end,
      )
    )

  return sorted(segments, key=lambda segment: (segment.start, segment.end, segment.speaker))


@lru_cache(maxsize=1)
def get_pyannote_pipeline(model: str, device: str, token: str):
  try:
    import torch
    from pyannote.audio import Pipeline
  except ImportError as exc:
    raise RuntimeError('pyannote.audio dependencies are not installed.') from exc

  logger.info('load diarization model=%s device=%s', model, device)
  pipeline = Pipeline.from_pretrained(model, token=token)
  pipeline.to(torch.device(device))
  return pipeline


class PyannoteDiarizer:
  def __init__(self, settings: PyannoteSettings):
    self.settings = settings

  @property
  def enabled(self) -> bool:
    return bool(self.settings.token)

  def diarize_file(self, audio_path: Path) -> DiarizationResult:
    if not self.enabled:
      return DiarizationResult(enabled=False)

    logger.info('diarization start model=%s file=%s', self.settings.model, audio_path.name)
    pipeline = get_pyannote_pipeline(
      self.settings.model,
      self.settings.device,
      self.settings.token or '',
    )
    diarization_options: dict[str, int] = {}

    if self.settings.min_speakers is not None and self.settings.max_speakers is not None:
      if self.settings.min_speakers == self.settings.max_speakers:
        diarization_options['num_speakers'] = self.settings.min_speakers
      else:
        diarization_options['min_speakers'] = self.settings.min_speakers
        diarization_options['max_speakers'] = self.settings.max_speakers
    else:
      if self.settings.min_speakers is not None:
        diarization_options['min_speakers'] = self.settings.min_speakers

      if self.settings.max_speakers is not None:
        diarization_options['max_speakers'] = self.settings.max_speakers

    output = pipeline(str(audio_path), **diarization_options)
    exclusive_annotation = getattr(output, 'exclusive_speaker_diarization', None)
    speaker_segments = normalize_speaker_segments(exclusive_annotation)
    segments_source = 'exclusive'

    if not speaker_segments:
      speaker_segments = normalize_speaker_segments(getattr(output, 'speaker_diarization', output))
      segments_source = 'regular'

    logger.info(
      'diarization done model=%s source=%s segments=%d',
      self.settings.model,
      segments_source,
      len(speaker_segments),
    )
    return DiarizationResult(
      enabled=True,
      model=self.settings.model,
      device=self.settings.device,
      segments_source=segments_source,
      speaker_segments=speaker_segments,
    )
