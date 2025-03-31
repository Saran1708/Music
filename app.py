import os
import tempfile
import ffmpeg
import librosa
import torch
import torchcrepe
import numpy as np
import torchaudio
from flask import Flask, request, send_file, render_template
from pydub import AudioSegment
from audiocraft.models import MusicGen

app = Flask(__name__)

# Set up device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure necessary directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('processed', exist_ok=True)

# Route for home page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Handle file upload
@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files['audio-file']
    prompt = request.form['prompt']
    
    if file:
        print("File received:", file.filename)
        print("Prompt received:", prompt)

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_path = temp_file.name + ".mp3"
            file.save(file_path)
        
        # Convert MP3 to WAV (store in processed folder)
        wav_path = os.path.join('processed', file.filename.replace(".mp3", ".wav"))
        convert_mp3_to_wav(file_path, wav_path)
        
        # Extract pitch data from the audio file
        pitch_data = extract_pitch_torchcrepe(wav_path)
        midi_notes = pitch_to_midi(pitch_data)
        
        # Generate background music based on prompt
        bgm_wav = os.path.join('processed', file.filename.replace(".mp3", "_bgm.wav"))
        generate_bgm_from_prompt(midi_notes, prompt, wav_path, bgm_wav)
        
        # Mix the original vocals with background music
        final_output = os.path.join('processed', file.filename.replace(".mp3", "_final_output.mp3"))
        mix_audio(wav_path, bgm_wav, final_output)
        
        # Return the final mixed file
        return send_file(final_output, as_attachment=True)
    else:
        return "No file uploaded", 400

# Function to convert MP3 to WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    ffmpeg.input(mp3_path).output(wav_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000').run(overwrite_output=True)

# Function to extract pitch using TorchCrepe
def extract_pitch_torchcrepe(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pitch = torchcrepe.predict(audio_tensor, sr, fmin=50, fmax=800, model='full', batch_size=512, device=device)
    
    return pitch.squeeze().cpu().numpy()

# Convert pitch data to MIDI notes
def pitch_to_midi(pitch_data):
    midi_notes = [round(69 + 12 * np.log2(pitch / 440)) for pitch in pitch_data if pitch > 0]
    return midi_notes

# Function to generate background music from prompt and midi notes
def generate_bgm_from_prompt(midi_notes, prompt, audio_file, output_wav="bgm.wav"):
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    chord_sequence = " ".join([f"Chord{note % 12}" for note in midi_notes[:10]]) + f" with tempo {tempo} BPM."
    chord_sequence += f" Create a dynamic, evolving background music in this style: {prompt}"
    
    model = MusicGen.get_pretrained('facebook/musicgen-small', device=device)
    audio_array = model.generate([chord_sequence], progress=True)
    
    audio_array_np = audio_array[0].cpu().numpy()
    if audio_array_np.ndim == 1:
        audio_array_np = np.expand_dims(audio_array_np, axis=0)
    
    audio_tensor = torch.tensor(audio_array_np, dtype=torch.float32)
    torchaudio.save(output_wav, audio_tensor, 16000)

# Function to mix the original vocals with background music
def mix_audio(original_wav, bgm_wav, output_mp3):
    vocals = AudioSegment.from_wav(original_wav)
    bgm = AudioSegment.from_wav(bgm_wav)
    mixed = vocals.overlay(bgm, position=0)
    mixed.export(output_mp3, format="mp3")

if __name__ == "__main__":
    app.run(debug=True)
