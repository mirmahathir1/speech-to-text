# Audio Transcriber

This project uses a Vue frontend and a FastAPI backend to upload audio files, send them to the OpenAI transcription API, and return a downloadable transcript with speaker-assigned turns such as `Speaker 1` and `Speaker 2`.

The backend calls `POST /v1/audio/transcriptions` with the diarized transcription model and normalizes the returned speaker labels into a stable `Speaker N` format for the UI and download output.

## Project structure

- `frontend/` contains the Vue + Vite upload interface.
- `backend/` contains the FastAPI API that forwards audio to OpenAI and reshapes the diarized response for the frontend.
- `docker-compose.yml` starts both services with bind mounts so source changes on the host are reflected inside the containers.

## Prerequisites

- Docker Desktop or Docker Engine with Compose support
- An OpenAI API key with access to audio transcription models

## Setup

1. Create `backend/.env` from `backend/.env.example`.
2. Set `OPENAI_API_KEY` in `backend/.env`.
3. Optionally adjust:
   - `OPENAI_TRANSCRIPTION_MODEL` if you want a different transcription model
   - `OPENAI_TRANSCRIPTION_LANGUAGE` to hint the input language
   - `OPENAI_SEGMENT_SECONDS` to choose how long each chunk should be before the backend loops through the file
   - `OPENAI_SEGMENT_OVERLAP_SECONDS` to control how much adjacent chunks overlap for speaker stitching
   - `MAX_UPLOAD_BYTES` if you want a stricter or looser server-side upload cap before preprocessing
4. Start the stack:

```bash
docker compose up --build
```

The frontend will be available at `http://localhost:5173` and the backend at `http://localhost:8000`.

## Live reload

- `./frontend` is bind-mounted into the frontend container.
- `./backend` is bind-mounted into the backend container.
- Vite runs with file polling enabled for reliable change detection in Docker.
- Uvicorn runs with `--reload`, so backend Python edits are picked up automatically.

## How it works

1. Drop an audio file onto the frontend.
2. The frontend posts the file to `POST /api/transcribe`.
3. The backend sends the upload to OpenAI using the configured API key.
4. If the source file is longer than the diarization model limit, the backend splits the original recording into overlapping chunks and transcribes them in sequence.
5. If an individual chunk is still larger than OpenAI's 25 MB upload limit, that chunk is compressed before upload.
6. OpenAI returns diarized transcript segments.
7. The backend uses the overlap window to stitch chunk-local speakers across boundaries, offsets timestamps back onto the original timeline, and normalizes the speaker labels to `Speaker 1`, `Speaker 2`, and so on.
8. The UI renders speaker-separated turns and exposes the formatted transcript as a `.txt` download.
