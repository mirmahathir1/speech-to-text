<script setup>
import { computed, ref } from 'vue'

const acceptedFileTypes =
  'audio/*,video/mp4,.aac,.flac,.m4a,.mp3,.mp4,.mpeg,.mpga,.oga,.ogg,.wav,.webm'
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
const turns = ref([])
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
const speakerCount = computed(
  () => new Set(turns.value.map((turn) => turn.speaker).filter(Boolean)).size,
)

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
  const isSupportedFile =
    file.type.startsWith('audio/') || file.type === 'video/mp4' || allowedExtensions.has(extension)

  if (!isSupportedFile) {
    errorMessage.value = 'Upload an audio file or an MP4 video file.'
    return
  }

  selectedFile.value = file
  transcript.value = ''
  turns.value = []
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
    turns.value = Array.isArray(data.turns) ? data.turns : []
    completedFileName.value = data.filename || file.name
    statusMessage.value = turns.value.length
      ? `Transcript ready for ${completedFileName.value} with ${speakerCount.value} detected ${speakerCount.value === 1 ? 'speaker' : 'speakers'}.`
      : `Transcript ready for ${completedFileName.value}.`
  } catch (error) {
    transcript.value = ''
    turns.value = []
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

function formatTimestamp(seconds) {
  if (typeof seconds !== 'number' || Number.isNaN(seconds) || seconds < 0) {
    return '00:00'
  }

  const totalSeconds = Math.floor(seconds)
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const remainingSeconds = totalSeconds % 60

  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`
  }

  return `${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`
}
</script>

<template>
  <main class="app-shell">
    <section class="hero-panel">
      <div class="hero-copy">
        <p class="eyebrow">Vue + FastAPI + OpenAI Audio</p>
        <h1>Upload audio or MP4 video. Get speaker-labeled text back.</h1>
        <p class="hero-text">
          The frontend sends your file to a FastAPI backend, the backend calls OpenAI&apos;s
          diarized transcription API, and the finished transcript comes back with lines assigned to
          Speaker 1, Speaker 2, and the rest of the detected speakers.
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
          <h2>Drag and drop an interview, lecture, meeting, video, or voice memo.</h2>
          <p>
            Supported uploads include MP3, WAV, M4A, OGG, FLAC, WebM, and MP4 video. Video uploads
            are converted to audio before transcription.
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
            <p class="loader-title">Transcribing with OpenAI</p>
            <p class="loader-copy">Speaker labels are added before the transcript returns.</p>
          </div>
        </div>

        <p v-if="errorMessage" class="error-banner">{{ errorMessage }}</p>
      </div>

      <aside class="transcript-card" :class="{ 'has-transcript': transcript }">
        <div class="transcript-header">
          <div>
            <p class="eyebrow">Transcript</p>
            <h2>Output</h2>
            <p v-if="speakerCount" class="transcript-meta">
              {{ speakerCount }} detected {{ speakerCount === 1 ? 'speaker' : 'speakers' }}
            </p>
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
          Your speaker-attributed transcript will appear here after the upload finishes.
        </p>

        <div v-else-if="isTranscribing" class="placeholder-copy placeholder-panel">
          Working through the audio now. The download button appears as soon as the speaker-labeled
          transcript is ready.
        </div>

        <div v-else-if="turns.length" class="turn-list">
          <article
            v-for="turn in turns"
            :key="`${turn.start}-${turn.end}-${turn.speaker}`"
            class="turn-card"
          >
            <div class="turn-head">
              <p class="turn-speaker">{{ turn.speaker || 'Speaker' }}</p>
              <p class="turn-time">
                {{ formatTimestamp(turn.start) }} - {{ formatTimestamp(turn.end) }}
              </p>
            </div>
            <p class="turn-text">{{ turn.text }}</p>
          </article>
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
