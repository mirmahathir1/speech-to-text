from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
  status: str
  mode: str
  model: str
  device: str


class TranscriptWord(BaseModel):
  word: str
  start: float | None = None
  end: float | None = None
  probability: float | None = None


class TranscriptSegment(BaseModel):
  id: int | None = None
  start: float
  end: float
  text: str
  avg_logprob: float | None = None
  compression_ratio: float | None = None
  no_speech_prob: float | None = None
  temperature: float | None = None
  words: list[TranscriptWord] = Field(default_factory=list)


class TranscriptTurn(BaseModel):
  speaker: str | None = None
  start: float
  end: float
  text: str
  segment_ids: list[int] = Field(default_factory=list)


class SpeakerSegment(BaseModel):
  speaker: str
  start: float
  end: float


class TranscriptionResponse(BaseModel):
  filename: str
  mode: str
  model: str
  language: str | None = None
  duration_seconds: float | None = None
  diarization_enabled: bool = False
  diarization_model: str | None = None
  diarization_device: str | None = None
  speaker_segments_source: str | None = None
  speaker_segments: list[SpeakerSegment] = Field(default_factory=list)
  transcript: str
  segments: list[TranscriptSegment] = Field(default_factory=list)
  turns: list[TranscriptTurn] = Field(default_factory=list)
