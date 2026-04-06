<script setup>
import { computed, ref } from 'vue'

const acceptedFileTypes =
  'audio/*,.aac,.flac,.m4a,.mp3,.mp4,.mpeg,.mpga,.oga,.ogg,.wav,.webm'
const allowedExtensions = new Set([
  '.aac',
  '.flac',
  '.m4a',
  '.mp3',
  '.mp4',
  '.mpeg',
  '.mpga',
  '.oga',
  '.ogg',
  '.wav',
  '.webm',
])

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

const fileInput = ref(null)
const selectedFile = ref(null)
const dragActive = ref(false)
const isTranscribing = ref(false)
const transcript = ref('')
const errorMessage = ref('')
const statusMessage = ref('Drop an audio file to start a transcription.')
const completedFileName = ref('')

const downloadFileName = computed(() => {
  const sourceName = completedFileName.value || 'transcript'
  const stem = sourceName
    .replace(/\.[^/.]+$/, '')
    .trim()
    .replace(/\s+/g, '-')
    .replace(/[^a-zA-Z0-9-_]/g, '')
    .replace(/-+/g, '-')
    .toLowerCase()

  return `${stem || 'transcript'}-transcript.txt`
})

function openFilePicker() {
  if (!isTranscribing.value) {
    fileInput.value?.click()
  }
}

function onInputChange(event) {
  const [file] = event.target.files || []
  event.target.value = ''
  prepareFile(file)
}

function onDragEnter() {
  if (!isTranscribing.value) {
    dragActive.value = true
  }
}

function onDragLeave(event) {
  if (event.currentTarget.contains(event.relatedTarget)) {
    return
  }

  dragActive.value = false
}

function onDrop(event) {
  dragActive.value = false
  const [file] = event.dataTransfer?.files || []
  prepareFile(file)
}

function prepareFile(file) {
  errorMessage.value = ''

  if (!file) {
    return
  }

  if (isTranscribing.value) {
    errorMessage.value = 'Wait for the current transcription to finish before uploading another file.'
    return
  }

  const extension = `.${file.name.split('.').pop()?.toLowerCase() || ''}`
  const isAudioFile = file.type.startsWith('audio/') || allowedExtensions.has(extension)

  if (!isAudioFile) {
    errorMessage.value = 'Only audio uploads are supported. Try MP3, WAV, M4A, OGG, or WebM.'
    return
  }

  selectedFile.value = file
  transcript.value = ''
  completedFileName.value = ''
  statusMessage.value = `Uploading ${file.name} to the transcription service...`

  void transcribeSelectedFile(file)
}

async function transcribeSelectedFile(file) {
  isTranscribing.value = true

  try {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${apiBaseUrl}/api/transcribe`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error(await extractErrorMessage(response))
    }

    const data = await response.json()
    transcript.value = data.transcript
    completedFileName.value = data.filename || file.name
    statusMessage.value = `Transcript ready for ${completedFileName.value}.`
  } catch (error) {
    transcript.value = ''
    completedFileName.value = ''
    errorMessage.value =
      error instanceof Error ? error.message : 'The transcription request failed.'
    statusMessage.value = 'The transcription request failed.'
  } finally {
    isTranscribing.value = false
  }
}

async function extractErrorMessage(response) {
  try {
    const payload = await response.json()

    if (typeof payload.detail === 'string') {
      return payload.detail
    }

    if (typeof payload.message === 'string') {
      return payload.message
    }
  } catch {
    return 'The transcription request failed.'
  }

  return 'The transcription request failed.'
}

function downloadTranscript() {
  if (!transcript.value) {
    return
  }

  const blob = new Blob([transcript.value], { type: 'text/plain;charset=utf-8' })
  const objectUrl = URL.createObjectURL(blob)
  const anchor = document.createElement('a')

  anchor.href = objectUrl
  anchor.download = downloadFileName.value
  anchor.click()

  URL.revokeObjectURL(objectUrl)
}

function formatFileSize(bytes) {
  if (!bytes) {
    return '0 B'
  }

  const units = ['B', 'KB', 'MB', 'GB']
  const unitIndex = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / 1024 ** unitIndex

  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`
}
</script>

<template>
  <main class="app-shell">
    <section class="hero-panel">
      <div class="hero-copy">
        <p class="eyebrow">Vue + FastAPI + Whisper</p>
        <h1>Drop in audio. Get a transcript back.</h1>
        <p class="hero-text">
          The frontend streams your file to a Python backend, the backend runs a local Whisper
          model in Docker, and the finished transcript becomes downloadable as a text file.
        </p>
      </div>

      <div class="hero-status">
        <p class="status-label">Current State</p>
        <p class="status-value">
          {{ isTranscribing ? 'Transcribing audio...' : statusMessage }}
        </p>
      </div>
    </section>

    <section class="workspace">
      <div
        class="dropzone-card"
        :class="{ 'is-active': dragActive, 'is-busy': isTranscribing }"
        @dragenter.prevent="onDragEnter"
        @dragover.prevent="onDragEnter"
        @dragleave="onDragLeave"
        @drop.prevent="onDrop"
      >
        <input
          ref="fileInput"
          class="visually-hidden"
          type="file"
          :accept="acceptedFileTypes"
          @change="onInputChange"
        />

        <div class="dropzone-copy">
          <p class="eyebrow">Audio Upload</p>
          <h2>Drag and drop an interview, lecture, meeting, or voice memo.</h2>
          <p>
            Supported uploads include MP3, WAV, M4A, OGG, FLAC, WebM, and the other common audio
            formats accepted by the local Whisper pipeline.
          </p>
        </div>

        <div class="dropzone-actions">
          <button
            class="secondary-button"
            type="button"
            :disabled="isTranscribing"
            @click="openFilePicker"
          >
            Choose File
          </button>
        </div>

        <div v-if="selectedFile" class="file-card">
          <p class="file-label">Selected File</p>
          <p class="file-name">{{ selectedFile.name }}</p>
          <p class="file-meta">{{ formatFileSize(selectedFile.size) }}</p>
        </div>

        <div v-if="isTranscribing" class="loader-block" aria-live="polite">
          <span class="spinner" />
          <div>
            <p class="loader-title">Transcribing with Local Whisper</p>
            <p class="loader-copy">The first run may take longer while the model downloads.</p>
          </div>
        </div>

        <p v-if="errorMessage" class="error-banner">{{ errorMessage }}</p>
      </div>

      <aside class="transcript-card" :class="{ 'has-transcript': transcript }">
        <div class="transcript-header">
          <div>
            <p class="eyebrow">Transcript</p>
            <h2>Output</h2>
          </div>

          <button
            v-if="transcript"
            class="primary-button"
            type="button"
            @click="downloadTranscript"
          >
            Download Transcript
          </button>
        </div>

        <p v-if="!transcript && !isTranscribing" class="placeholder-copy">
          Your transcript will appear here after the upload finishes.
        </p>

        <div v-else-if="isTranscribing" class="placeholder-copy placeholder-panel">
          Working through the audio now. The download button appears as soon as the transcript is
          ready.
        </div>

        <textarea
          v-else
          class="transcript-output"
          readonly
          :value="transcript"
        ></textarea>
      </aside>
    </section>
  </main>
</template>
