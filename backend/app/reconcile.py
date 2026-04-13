from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import SpeakerSegment, TranscriptSegment, TranscriptTurn, TranscriptWord


NO_SPACE_BEFORE_RE = re.compile(r"\s+([,.;:!?%)\]\}])")
NO_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{])\s+")
NO_SPACE_BEFORE_CONTRACTION_RE = re.compile(r"\s+(['’](?:s|t|re|ve|ll|d|m))\b", re.IGNORECASE)


@dataclass
class ReconciliationResult:
  transcript: str
  turns: list[TranscriptTurn]
  speaker_segments: list[SpeakerSegment]


@dataclass
class AttributedChunk:
  text: str
  start: float
  end: float
  speaker: str | None
  segment_ids: list[int] = field(default_factory=list)


def normalize_speaker_labels(speaker_segments: list[SpeakerSegment]) -> tuple[list[SpeakerSegment], dict[str, str]]:
  sorted_segments = sorted(
    speaker_segments,
    key=lambda segment: (segment.start, segment.end, segment.speaker),
  )
  speaker_labels: dict[str, str] = {}
  normalized_segments: list[SpeakerSegment] = []

  for segment in sorted_segments:
    label = speaker_labels.setdefault(segment.speaker, f'Speaker {len(speaker_labels) + 1}')
    normalized_segments.append(
      SpeakerSegment(
        speaker=label,
        start=segment.start,
        end=segment.end,
      )
    )

  return normalized_segments, speaker_labels


def chunk_text_supports_word_alignment(word: TranscriptWord) -> bool:
  return bool(word.word.strip()) and word.start is not None and word.end is not None


def segment_supports_word_alignment(segment: TranscriptSegment) -> bool:
  return bool(segment.words) and all(chunk_text_supports_word_alignment(word) for word in segment.words)


def overlap_duration(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
  return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def distance_to_interval(point: float, start: float, end: float) -> float:
  if point < start:
    return start - point

  if point > end:
    return point - end

  return 0.0


def choose_best_speaker(start: float, end: float, speaker_segments: list[SpeakerSegment]) -> str | None:
  if not speaker_segments:
    return None

  best_speaker: str | None = None
  best_overlap = -1.0

  for segment in speaker_segments:
    overlap = overlap_duration(start, end, segment.start, segment.end)

    if overlap > best_overlap:
      best_overlap = overlap
      best_speaker = segment.speaker

  if best_overlap > 0:
    return best_speaker

  midpoint = start if end <= start else (start + end) / 2
  nearest_segment = min(
    speaker_segments,
    key=lambda segment: (
      distance_to_interval(midpoint, segment.start, segment.end),
      segment.start,
      segment.end,
    ),
  )
  return nearest_segment.speaker


def build_chunks(
  segments: list[TranscriptSegment],
  speaker_segments: list[SpeakerSegment],
) -> list[AttributedChunk]:
  chunks: list[AttributedChunk] = []

  for segment in segments:
    segment_ids = [segment.id] if segment.id is not None else []

    if segment_supports_word_alignment(segment):
      for word in segment.words:
        word_text = word.word.strip()

        if not word_text:
          continue

        start = float(word.start or segment.start)
        end = float(word.end if word.end is not None else start)
        chunks.append(
          AttributedChunk(
            text=word_text,
            start=start,
            end=end,
            speaker=choose_best_speaker(start, end, speaker_segments),
            segment_ids=segment_ids,
          )
        )
      continue

    segment_text = segment.text.strip()

    if not segment_text:
      continue

    chunks.append(
      AttributedChunk(
        text=segment_text,
        start=segment.start,
        end=segment.end,
        speaker=choose_best_speaker(segment.start, segment.end, speaker_segments),
        segment_ids=segment_ids,
      )
    )

  return chunks


def normalize_text_parts(parts: list[str]) -> str:
  cleaned_parts = [part.strip() for part in parts if part and part.strip()]

  if not cleaned_parts:
    return ''

  text = ' '.join(cleaned_parts)
  text = NO_SPACE_BEFORE_RE.sub(r'\1', text)
  text = NO_SPACE_AFTER_OPEN_RE.sub(r'\1', text)
  text = NO_SPACE_BEFORE_CONTRACTION_RE.sub(r'\1', text)
  text = re.sub(r'\s+', ' ', text).strip()

  return text


def merge_chunks_into_turns(chunks: list[AttributedChunk]) -> list[TranscriptTurn]:
  if not chunks:
    return []

  turns: list[TranscriptTurn] = []
  current_speaker = chunks[0].speaker
  current_start = chunks[0].start
  current_end = chunks[0].end
  current_parts: list[str] = [chunks[0].text]
  current_segment_ids = list(chunks[0].segment_ids)

  for chunk in chunks[1:]:
    if chunk.speaker == current_speaker:
      current_end = max(current_end, chunk.end)
      current_parts.append(chunk.text)

      for segment_id in chunk.segment_ids:
        if segment_id not in current_segment_ids:
          current_segment_ids.append(segment_id)

      continue

    turns.append(
      TranscriptTurn(
        speaker=current_speaker,
        start=current_start,
        end=current_end,
        text=normalize_text_parts(current_parts),
        segment_ids=current_segment_ids,
      )
    )
    current_speaker = chunk.speaker
    current_start = chunk.start
    current_end = chunk.end
    current_parts = [chunk.text]
    current_segment_ids = list(chunk.segment_ids)

  turns.append(
    TranscriptTurn(
      speaker=current_speaker,
      start=current_start,
      end=current_end,
      text=normalize_text_parts(current_parts),
      segment_ids=current_segment_ids,
    )
  )

  return [turn for turn in turns if turn.text]


def build_segment_turns(segments: list[TranscriptSegment]) -> list[TranscriptTurn]:
  turns: list[TranscriptTurn] = []

  for segment in segments:
    text = segment.text.strip()

    if not text:
      continue

    turns.append(
      TranscriptTurn(
        start=segment.start,
        end=segment.end,
        text=text,
        segment_ids=[segment.id] if segment.id is not None else [],
      )
    )

  return turns


def format_transcript(turns: list[TranscriptTurn]) -> str:
  blocks: list[str] = []

  for turn in turns:
    text = turn.text.strip()

    if not text:
      continue

    if turn.speaker:
      blocks.append(f'{turn.speaker}: {text}')
    else:
      blocks.append(text)

  return '\n\n'.join(blocks).strip()


def reconcile_transcript(
  segments: list[TranscriptSegment],
  speaker_segments: list[SpeakerSegment],
  fallback_transcript: str,
) -> ReconciliationResult:
  if not speaker_segments:
    turns = build_segment_turns(segments)
    transcript = format_transcript(turns) or fallback_transcript

    return ReconciliationResult(
      transcript=transcript,
      turns=turns,
      speaker_segments=[],
    )

  normalized_speaker_segments, _ = normalize_speaker_labels(speaker_segments)
  chunks = build_chunks(segments, normalized_speaker_segments)
  turns = merge_chunks_into_turns(chunks)
  transcript = format_transcript(turns) or fallback_transcript

  return ReconciliationResult(
    transcript=transcript,
    turns=turns,
    speaker_segments=normalized_speaker_segments,
  )
