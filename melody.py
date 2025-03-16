import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import whisper
import google.generativeai as genai
import requests
from pydub import AudioSegment
import urllib.request

genai.configure(api_key="AIzaSyCFApgdQgxvenxQZPQVx_mBKzRmIOIPnl8")  # Replace with your actual API key

# ------------------- Web App UI -------------------

st.title("🎵 AI Melody Generator for Babies")
st.write("Sing or hum a short phrase, and AI will generate a full lullaby.")

voice_option = st.selectbox("Choose AI Singer:",  
    ["Soft Female", "Deep Male", "Childlike Voice", "Robotic Lullaby"])  

voice_mapping = {
    "Soft Female": "soft_female",
    "Deep Male": "deep_male",
    "Childlike Voice": "childlike",
    "Robotic Lullaby": "robotic"
}

selected_voice = voice_mapping[voice_option]

# ------------------- Audio Recording -------------------

fs = 44100  
duration = 5  
wavefile = "input_audio.wav"

if st.button("🎤 Record Your Voice"):
    st.write("Recording... Sing now!")  
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()

    with wave.open(wavefile, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())

    st.success("✅ Recording saved! Processing your lullaby...")  

# ------------------- Whisper AI Transcription -------------------

if st.button("📝 Transcribe & Generate Lullaby"):
    st.write("Transcribing your voice...")  
    model = whisper.load_model("base")
    result = model.transcribe(wavefile)
    input_text = result["text"]
    st.write("🎶 Transcribed Lyrics:", input_text)

    # ------------------- Gemini AI Expansion -------------------

    st.write("Generating full lullaby...")  
    prompt = f"Expand this into a soft, rhyming lullaby:\n'{input_text}'"
    
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    generated_lyrics = response.text
    
    st.write("🎼 Generated Lullaby Lyrics:", generated_lyrics)

    # ------------------- AI Melody Generation (MusicGen or Riffusion) -------------------

    st.write("Composing soft lullaby melody...")  
    music_payload = {
        "prompt": generated_lyrics,  
        "style": "soft lullaby, slow tempo, soothing tone"
    }
    
    response = requests.post("https://api.musicgen.com/generate", json=music_payload)
    melody_audio = response.json()["audio_url"]
    st.audio(melody_audio, format="audio/mp3")

    # ------------------- AI Singing with Zonos/LLaMA Speech -------------------

    st.write("Generating AI-sung lullaby...")  
    tts_payload = {
        "text": generated_lyrics,
        "voice": selected_voice,  # User-selected voice
        "style": "singing"
    }
    
    response = requests.post("https://api.zonos.ai/tts", json=tts_payload)
    singing_audio = response.json()["audio_url"]
    st.audio(singing_audio, format="audio/mp3")

    # ------------------- Merge Vocals & Melody -------------------

    st.write("Mixing vocals, melody, and background sounds...")  
    singing = AudioSegment.from_file(singing_audio, format="mp3")
    melody = AudioSegment.from_file(melody_audio, format="mp3")
    background = AudioSegment.from_file("rain_sound.mp3", format="mp3")  # Optional

    # Adjust volumes
    singing = singing - 3
    melody = melody - 2
    background = background - 15  # Softer background noise

    # Mix all elements
    final_lullaby = singing.overlay(melody).overlay(background, position=0)
    final_lullaby.export("final_lullaby.mp3", format="mp3")

    st.success("✅ Your AI lullaby is ready!")  
    st.audio("final_lullaby.mp3", format="audio/mp3")

    # ------------------- Download Button -------------------

    lullaby_filename = "final_lullaby.mp3"
    with open(lullaby_filename, "rb") as file:
        st.download_button(label="⬇ Download Lullaby",  
                           data=file,  
                           file_name=lullaby_filename,  
                           mime="audio/mp3")

