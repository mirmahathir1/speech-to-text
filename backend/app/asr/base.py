from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..schemas import TranscriptSegment


@dataclass
class AsrTranscription:
  transcript: str
  language: str | None = None
  duration_seconds: float | None = None
  segments: list[TranscriptSegment] = field(default_factory=list)


class AsrAdapter(Protocol):
  def transcribe_file(self, audio_path: Path, *, language: str | None = None) -> AsrTranscription:
    ...
