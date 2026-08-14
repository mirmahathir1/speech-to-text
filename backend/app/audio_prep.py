from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
import tempfile
from pathlib import Path


OPENAI_MAX_AUDIO_BYTES = 25 * 1024 * 1024
MULTIPART_HEADROOM_BYTES = 512 * 1024
OPENAI_TARGET_AUDIO_BYTES = OPENAI_MAX_AUDIO_BYTES - MULTIPART_HEADROOM_BYTES
OPENAI_MAX_MODEL_DURATION_SECONDS = 1400.0
DEFAULT_SEGMENT_DURATION_SECONDS = 1300.0
DEFAULT_SEGMENT_OVERLAP_SECONDS = 20.0
DEFAULT_MAX_LOCAL_UPLOAD_BYTES = 100 * 1024 * 1024
MIN_AUDIO_BITRATE_KBPS = 16
MAX_AUDIO_BITRATE_KBPS = 64
FALLBACK_BITRATES_KBPS = (64, 48, 32, 24, 16)
CONTENT_TYPES_BY_SUFFIX = {
  '.aac': 'audio/aac',
  '.flac': 'audio/flac',
  '.m4a': 'audio/mp4',
  '.mp3': 'audio/mpeg',
  '.mp4': 'audio/mp4',
  '.mpeg': 'audio/mpeg',
  '.mpga': 'audio/mpeg',
  '.oga': 'audio/ogg',
  '.ogg': 'audio/ogg',
  '.wav': 'audio/wav',
  '.webm': 'audio/webm',
}
VIDEO_SUFFIXES_REQUIRING_AUDIO_EXTRACTION = {'.mp4'}
VIDEO_CONTENT_TYPES_REQUIRING_AUDIO_EXTRACTION = {'video/mp4'}


@dataclass(frozen=True)
class PreparedAudioUpload:
  filename: str
  payload: bytes
  content_type: str | None = None
  transcoded: bool = False
  offset_seconds: float = 0.0
  duration_seconds: float | None = None
  keep_from_seconds: float = 0.0


class AudioPreparationError(Exception):
  def __init__(self, detail: str, *, status_code: int = 400):
    super().__init__(detail)
    self.status_code = status_code


def _require_binary(binary_name: str) -> str:
  resolved = shutil.which(binary_name)

  if resolved:
    return resolved

  raise AudioPreparationError(
    f'{binary_name} is required for audio preprocessing but is not installed.',
    status_code=500,
  )


def _guess_content_type(suffix: str, fallback: str | None = None) -> str | None:
  normalized_suffix = suffix.lower()
  return CONTENT_TYPES_BY_SUFFIX.get(normalized_suffix, fallback)


def _requires_audio_extraction(filename: str, content_type: str | None) -> bool:
  suffix = Path(filename).suffix.lower()
  normalized_content_type = (content_type or '').split(';', maxsplit=1)[0].strip().lower()

  return (
    suffix in VIDEO_SUFFIXES_REQUIRING_AUDIO_EXTRACTION
    or normalized_content_type in VIDEO_CONTENT_TYPES_REQUIRING_AUDIO_EXTRACTION
  )


def _probe_duration_seconds(input_path: Path) -> float | None:
  ffprobe = _require_binary('ffprobe')
  result = subprocess.run(
    [
      ffprobe,
      '-v',
      'error',
      '-show_entries',
      'format=duration',
      '-of',
      'default=nokey=1:noprint_wrappers=1',
      str(input_path),
    ],
    capture_output=True,
    text=True,
    check=False,
  )

  if result.returncode != 0:
    return None

  raw_output = result.stdout.strip()

  if not raw_output:
    return None

  try:
    duration_seconds = float(raw_output)
  except ValueError:
    return None

  return duration_seconds if duration_seconds > 0 else None


def _candidate_bitrates(duration_seconds: float | None) -> list[int]:
  if duration_seconds is None:
    return list(FALLBACK_BITRATES_KBPS)

  estimated_kbps = int((OPENAI_TARGET_AUDIO_BYTES * 8 / duration_seconds) / 1000)
  target_kbps = max(
    MIN_AUDIO_BITRATE_KBPS,
    min(MAX_AUDIO_BITRATE_KBPS, estimated_kbps - 4),
  )
  candidates = [target_kbps]

  for bitrate in FALLBACK_BITRATES_KBPS:
    if bitrate < target_kbps and bitrate not in candidates:
      candidates.append(bitrate)

  if MIN_AUDIO_BITRATE_KBPS not in candidates:
    candidates.append(MIN_AUDIO_BITRATE_KBPS)

  return candidates


def _transcode_payload_to_target_size(
  *,
  filename: str,
  payload: bytes,
  offset_seconds: float,
  duration_seconds: float | None,
  keep_from_seconds: float,
) -> PreparedAudioUpload:
  ffmpeg = _require_binary('ffmpeg')
  source_suffix = Path(filename).suffix or '.audio'

  with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir_path = Path(temp_dir)
    input_path = temp_dir_path / f'input{source_suffix}'
    input_path.write_bytes(payload)

    known_duration_seconds = duration_seconds or _probe_duration_seconds(input_path)
    smallest_output: bytes | None = None

    for bitrate in _candidate_bitrates(known_duration_seconds):
      output_path = temp_dir_path / f'compressed-{bitrate}k.m4a'
      result = subprocess.run(
        [
          ffmpeg,
          '-y',
          '-i',
          str(input_path),
          '-vn',
          '-map_metadata',
          '-1',
          '-ac',
          '1',
          '-ar',
          '16000',
          '-c:a',
          'aac',
          '-b:a',
          f'{bitrate}k',
          str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
      )

      if result.returncode != 0 or not output_path.exists():
        continue

      output_payload = output_path.read_bytes()

      if smallest_output is None or len(output_payload) < len(smallest_output):
        smallest_output = output_payload

      if len(output_payload) <= OPENAI_TARGET_AUDIO_BYTES:
        return PreparedAudioUpload(
          filename=f'{Path(filename).stem}-compressed.m4a',
          payload=output_payload,
          content_type='audio/mp4',
          transcoded=True,
          offset_seconds=offset_seconds,
          duration_seconds=known_duration_seconds,
          keep_from_seconds=keep_from_seconds,
        )

    if smallest_output and len(smallest_output) <= OPENAI_MAX_AUDIO_BYTES:
      return PreparedAudioUpload(
        filename=f'{Path(filename).stem}-compressed.m4a',
        payload=smallest_output,
        content_type='audio/mp4',
        transcoded=True,
        offset_seconds=offset_seconds,
        duration_seconds=known_duration_seconds,
        keep_from_seconds=keep_from_seconds,
      )

  raise AudioPreparationError(
    'OpenAI currently limits transcription uploads to 25 MB. '
    'The server tried to compress this segment automatically, but it is still too large. '
    'Trim the recording or split it into smaller files and try again.',
    status_code=400,
  )


def _prepare_single_upload(
  *,
  filename: str,
  payload: bytes,
  content_type: str | None,
  offset_seconds: float,
  duration_seconds: float | None,
  keep_from_seconds: float,
) -> PreparedAudioUpload:
  if len(payload) <= OPENAI_TARGET_AUDIO_BYTES:
    return PreparedAudioUpload(
      filename=filename,
      payload=payload,
      content_type=content_type,
      transcoded=False,
      offset_seconds=offset_seconds,
      duration_seconds=duration_seconds,
      keep_from_seconds=keep_from_seconds,
    )

  return _transcode_payload_to_target_size(
    filename=filename,
    payload=payload,
    offset_seconds=offset_seconds,
    duration_seconds=duration_seconds,
    keep_from_seconds=keep_from_seconds,
  )


def _extract_audio_from_video(input_path: Path, output_path: Path) -> None:
  ffmpeg = _require_binary('ffmpeg')
  copy_result = subprocess.run(
    [
      ffmpeg,
      '-y',
      '-i',
      str(input_path),
      '-vn',
      '-map',
      '0:a:0',
      '-map_metadata',
      '-1',
      '-c:a',
      'copy',
      str(output_path),
    ],
    capture_output=True,
    text=True,
    check=False,
  )

  if copy_result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
    return

  output_path.unlink(missing_ok=True)
  transcode_result = subprocess.run(
    [
      ffmpeg,
      '-y',
      '-i',
      str(input_path),
      '-vn',
      '-map',
      '0:a:0',
      '-map_metadata',
      '-1',
      '-ac',
      '1',
      '-ar',
      '16000',
      '-c:a',
      'aac',
      '-b:a',
      '64k',
      str(output_path),
    ],
    capture_output=True,
    text=True,
    check=False,
  )

  if transcode_result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
    return

  raise AudioPreparationError(
    'The uploaded MP4 does not contain an extractable audio track.',
    status_code=400,
  )


def _extract_segment(
  *,
  input_path: Path,
  output_path: Path,
  offset_seconds: float,
  segment_duration_seconds: float,
) -> tuple[bytes, str, bool]:
  ffmpeg = _require_binary('ffmpeg')
  copy_result = subprocess.run(
    [
      ffmpeg,
      '-y',
      '-ss',
      f'{offset_seconds:.3f}',
      '-i',
      str(input_path),
      '-t',
      f'{segment_duration_seconds:.3f}',
      '-vn',
      '-map',
      '0:a:0?',
      '-c',
      'copy',
      str(output_path),
    ],
    capture_output=True,
    text=True,
    check=False,
  )

  if copy_result.returncode == 0 and output_path.exists():
    return output_path.read_bytes(), output_path.suffix, False

  fallback_output_path = output_path.with_suffix('.m4a')
  fallback_result = subprocess.run(
    [
      ffmpeg,
      '-y',
      '-ss',
      f'{offset_seconds:.3f}',
      '-i',
      str(input_path),
      '-t',
      f'{segment_duration_seconds:.3f}',
      '-vn',
      '-map_metadata',
      '-1',
      '-ac',
      '1',
      '-ar',
      '16000',
      '-c:a',
      'aac',
      '-b:a',
      '64k',
      str(fallback_output_path),
    ],
    capture_output=True,
    text=True,
    check=False,
  )

  if fallback_result.returncode == 0 and fallback_output_path.exists():
    return fallback_output_path.read_bytes(), fallback_output_path.suffix, True

  raise AudioPreparationError(
    'The server could not split the uploaded audio into smaller transcription chunks.',
    status_code=500,
  )


def prepare_audio_uploads(
  *,
  filename: str,
  payload: bytes,
  content_type: str | None = None,
  max_segment_seconds: float = DEFAULT_SEGMENT_DURATION_SECONDS,
  overlap_seconds: float = DEFAULT_SEGMENT_OVERLAP_SECONDS,
) -> list[PreparedAudioUpload]:
  segment_seconds = min(max_segment_seconds, OPENAI_MAX_MODEL_DURATION_SECONDS)
  overlap_seconds = max(0.0, min(overlap_seconds, max(0.0, segment_seconds - 1.0)))

  if segment_seconds <= 0:
    raise AudioPreparationError('OPENAI_SEGMENT_SECONDS must be greater than zero.', status_code=500)

  source_suffix = Path(filename).suffix or '.audio'

  with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir_path = Path(temp_dir)
    input_path = temp_dir_path / f'input{source_suffix}'
    input_path.write_bytes(payload)

    if _requires_audio_extraction(filename, content_type):
      extracted_audio_path = temp_dir_path / 'extracted-audio.m4a'
      _extract_audio_from_video(input_path, extracted_audio_path)
      filename = f'{Path(filename).stem}-audio.m4a'
      payload = extracted_audio_path.read_bytes()
      content_type = 'audio/mp4'
      source_suffix = '.m4a'
      input_path = extracted_audio_path

    total_duration_seconds = _probe_duration_seconds(input_path)

    if total_duration_seconds is None or total_duration_seconds <= segment_seconds:
      return [
        _prepare_single_upload(
          filename=filename,
          payload=payload,
          content_type=content_type,
          offset_seconds=0.0,
          duration_seconds=total_duration_seconds,
          keep_from_seconds=0.0,
        )
      ]

    uploads: list[PreparedAudioUpload] = []
    chunk_start_seconds = 0.0
    keep_from_seconds = 0.0
    chunk_index = 1

    while chunk_start_seconds < total_duration_seconds:
      offset_seconds = chunk_start_seconds
      chunk_duration_seconds = min(segment_seconds, total_duration_seconds - offset_seconds)
      chunk_end_seconds = offset_seconds + chunk_duration_seconds
      chunk_path = temp_dir_path / f'chunk-{chunk_index:03d}{source_suffix}'
      chunk_payload, chunk_suffix, split_transcoded = _extract_segment(
        input_path=input_path,
        output_path=chunk_path,
        offset_seconds=offset_seconds,
        segment_duration_seconds=chunk_duration_seconds,
      )
      chunk_filename = f'{Path(filename).stem}-part-{chunk_index:03d}{chunk_suffix}'
      chunk_content_type = _guess_content_type(chunk_suffix, content_type)
      prepared_upload = _prepare_single_upload(
        filename=chunk_filename,
        payload=chunk_payload,
        content_type=chunk_content_type,
        offset_seconds=offset_seconds,
        duration_seconds=chunk_duration_seconds,
        keep_from_seconds=keep_from_seconds,
      )

      if split_transcoded and not prepared_upload.transcoded:
        prepared_upload = PreparedAudioUpload(
          filename=prepared_upload.filename,
          payload=prepared_upload.payload,
          content_type=prepared_upload.content_type,
          transcoded=True,
          offset_seconds=prepared_upload.offset_seconds,
          duration_seconds=prepared_upload.duration_seconds,
          keep_from_seconds=prepared_upload.keep_from_seconds,
        )

      uploads.append(prepared_upload)

      if chunk_end_seconds >= total_duration_seconds:
        break

      keep_from_seconds = chunk_end_seconds
      next_chunk_start_seconds = max(0.0, chunk_end_seconds - overlap_seconds)

      if next_chunk_start_seconds <= chunk_start_seconds:
        next_chunk_start_seconds = chunk_end_seconds

      chunk_start_seconds = next_chunk_start_seconds
      chunk_index += 1

    return uploads
