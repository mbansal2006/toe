import os
import subprocess
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from TTS.api import TTS
from pynput import keyboard
import numpy as np
import threading

# Paths
WHISPER_DIR = "models/faster-whisper-small"
LLAMA_SIMPLE = "llama.cpp/build/bin/llama-simple"
LLAMA_MODEL = "models/gemma-3n-E2B-it-Q4_0.gguf"
REFERENCE_VOICE = "models/coqui_xtts_v2/reference_teacher.wav"
USER_AUDIO = "user_input.wav"

# Globals
SAMPLERATE = 16000
CHANNELS = 1
is_recording = False
audio_frames = []
stream = None
listener = None

def start_recording():
    global stream, audio_frames
    print("🔴 Recording started... Press SPACE again to stop.")
    audio_frames = []

    def callback(indata, frames, time, status):
        if is_recording:
            audio_frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, dtype='int16', callback=callback)
    stream.start()

def stop_recording():
    global stream
    stream.stop()
    stream.close()
    print("🛑 Stopping recording...")

    audio_np = np.concatenate(audio_frames, axis=0)
    sf.write(USER_AUDIO, audio_np, SAMPLERATE)
    print(f"✅ Saved to {USER_AUDIO}")

    listener.stop()

def on_press(key):
    global is_recording
    if key == keyboard.Key.space:
        if not is_recording:
            is_recording = True
            threading.Thread(target=start_recording).start()
        else:
            is_recording = False
            stop_recording()
            return False  # Stop listener

# Run recording
print("🟢 Press SPACE to start and stop recording.")
listener = keyboard.Listener(on_press=on_press)
listener.start()
listener.join()

# Transcribe
print("🧠 Transcribing...")
whisper = WhisperModel(model_size_or_path=WHISPER_DIR, compute_type="int8")
segments, _ = whisper.transcribe(USER_AUDIO)
student_prompt = " ".join([segment.text for segment in segments])
print(f"🗣 Student said: {student_prompt}")

# Run llama-simple
print("💬 Thinking...")
response = subprocess.run([
    LLAMA_SIMPLE,
    "-m", LLAMA_MODEL,
    "-p", student_prompt,
    "-n", "200"
], capture_output=True, text=True)

lines = response.stdout.splitlines()
teacher_response = next((l for l in lines if l.strip() and not l.startswith("<") and not l.startswith("main:")), "")
print(f"📘 AI Teacher: {teacher_response}")

# TTS
print("🔊 Generating voice...")
model = TTS(model_name="tts_models/multilingual/xtts_v2")
out = model.tts(
    text=teacher_response,
    speaker_wav=REFERENCE_VOICE,
    language="en"
)
sf.write("teacher_response.wav", out, 24000)
print("✅ Response audio saved to teacher_response.wav")