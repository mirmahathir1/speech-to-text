# Audio Transcriber

This project uses a Vue frontend and a FastAPI backend to upload audio files, run a local Whisper model for transcription, and download the finished transcript as a `.txt` file.

No OpenAI API key is required for transcription in this version.

## Project structure

- `frontend/` contains the Vue + Vite drag-and-drop interface.
- `backend/` contains the FastAPI API that runs local Whisper transcription.
- `docker-compose.yml` starts both services with bind mounts so source changes on the host are reflected inside the containers.

## Prerequisites

- Docker Desktop or Docker Engine with Compose support

## Setup

1. Review `backend/.env` if you want to change the local model.
   A starter file has already been created from `backend/.env.example`.

2. Start the stack:

```bash
docker compose up --build
```

The frontend will be available at `http://localhost:5173` and the backend at `http://localhost:8000`.
The first transcription downloads the configured Whisper model into `backend/.cache/whisper`; after that, the model stays on disk in your bind-mounted backend folder.

## Live reload

- `./frontend` is bind-mounted into the frontend container.
- `./backend` is bind-mounted into the backend container.
- Vite runs with file polling enabled for reliable change detection in Docker.
- Uvicorn runs with `--reload`, so backend Python edits are picked up automatically.

## How it works

1. Drop an audio file onto the frontend.
2. The frontend posts the file to `POST /api/transcribe`.
3. The backend runs the open-source Whisper model locally inside the backend container.
4. When the transcript returns, the UI renders it and shows a download button.
