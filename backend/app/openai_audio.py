from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(frozen=True)
class OpenAISettings:
  api_key: str
  api_base_url: str
  model: str
  language: str | None = None
  timeout_seconds: float = 600.0


@dataclass(frozen=True)
class OpenAIDiarizedSegment:
  speaker: str
  start: float
  end: float
  text: str


@dataclass
class OpenAITranscriptionResult:
  transcript: str
  duration_seconds: float | None = None
  language: str | None = None
  segments: list[OpenAIDiarizedSegment] = field(default_factory=list)
  usage: dict[str, Any] | None = None


class OpenAITranscriptionError(Exception):
  def __init__(self, detail: str, *, status_code: int = 502):
    super().__init__(detail)
    self.status_code = status_code


def _coerce_float(value: Any) -> float | None:
  try:
    if value is None:
      return None

    return float(value)
  except (TypeError, ValueError):
    return None


def _extract_error_message(response: httpx.Response) -> str | None:
  try:
    payload = response.json()
  except ValueError:
    return response.text.strip() or None

  if not isinstance(payload, dict):
    return response.text.strip() or None

  error = payload.get('error')

  if isinstance(error, dict):
    message = error.get('message')

    if isinstance(message, str) and message.strip():
      return message.strip()

  detail = payload.get('detail')

  if isinstance(detail, str) and detail.strip():
    return detail.strip()

  return response.text.strip() or None


def _map_error(response: httpx.Response) -> OpenAITranscriptionError:
  detail = _extract_error_message(response) or 'OpenAI transcription request failed.'

  if response.status_code == 401:
    return OpenAITranscriptionError(
      'OpenAI rejected the configured API key.',
      status_code=502,
    )

  if response.status_code == 403:
    return OpenAITranscriptionError(
      'The configured OpenAI project cannot access the requested transcription model.',
      status_code=502,
    )

  if response.status_code == 413:
    return OpenAITranscriptionError(
      'The uploaded file exceeds OpenAI\'s current 25 MB transcription limit.',
      status_code=400,
    )

  if response.status_code == 429:
    return OpenAITranscriptionError(
      'OpenAI rate limits are currently being hit. Try again shortly.',
      status_code=503,
    )

  if response.status_code >= 500:
    return OpenAITranscriptionError(
      'OpenAI failed to transcribe the audio upstream. Try again shortly.',
      status_code=502,
    )

  return OpenAITranscriptionError(detail, status_code=400)


def _normalize_segments(raw_segments: Any) -> list[OpenAIDiarizedSegment]:
  if not isinstance(raw_segments, list):
    return []

  segments: list[OpenAIDiarizedSegment] = []

  for item in raw_segments:
    if not isinstance(item, dict):
      continue

    text = str(item.get('text') or '').strip()
    speaker = str(item.get('speaker') or '').strip() or 'Unknown'
    start = _coerce_float(item.get('start'))
    end = _coerce_float(item.get('end'))

    if not text or start is None or end is None or end <= start:
      continue

    segments.append(
      OpenAIDiarizedSegment(
        speaker=speaker,
        start=start,
        end=end,
        text=text,
      )
    )

  return segments


class OpenAITranscriber:
  def __init__(self, settings: OpenAISettings):
    self.settings = settings

  async def transcribe_bytes(
    self,
    *,
    filename: str,
    payload: bytes,
    content_type: str | None = None,
  ) -> OpenAITranscriptionResult:
    if not self.settings.api_key:
      raise RuntimeError('OPENAI_API_KEY is not configured.')

    data: dict[str, str] = {
      'model': self.settings.model,
      'response_format': 'diarized_json',
      'chunking_strategy': 'auto',
    }

    if self.settings.language:
      data['language'] = self.settings.language

    headers = {
      'Authorization': f'Bearer {self.settings.api_key}',
    }
    files = {
      'file': (
        filename,
        payload,
        content_type or 'application/octet-stream',
      )
    }
    endpoint = f'{self.settings.api_base_url}/audio/transcriptions'

    async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
      try:
        response = await client.post(
          endpoint,
          headers=headers,
          data=data,
          files=files,
        )
      except httpx.TimeoutException as exc:
        raise RuntimeError('OpenAI transcription timed out.') from exc
      except httpx.HTTPError as exc:
        raise RuntimeError('Failed to reach OpenAI for transcription.') from exc

    if response.is_error:
      raise _map_error(response)

    try:
      result = response.json()
    except ValueError as exc:
      raise RuntimeError('OpenAI returned a non-JSON transcription response.') from exc

    if not isinstance(result, dict):
      raise RuntimeError('OpenAI returned an invalid transcription payload.')

    segments = _normalize_segments(result.get('segments'))
    transcript = str(result.get('text') or '').strip()

    return OpenAITranscriptionResult(
      transcript=transcript,
      duration_seconds=_coerce_float(result.get('duration')),
      language=self.settings.language,
      segments=segments,
      usage=result.get('usage') if isinstance(result.get('usage'), dict) else None,
    )
