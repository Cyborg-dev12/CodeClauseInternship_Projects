from flask import Flask, render_template, request, redirect, url_for
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torchaudio
import torch
import numpy as np
import sounddevice as sd
import os

app = Flask(__name__)

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")


SAMPLE_RATE = 16000
DURATION = 10 


def transcribe_audio(audio_data):

    inputs = tokenizer(audio_data, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(inputs.input_values).logits

 
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    
    return transcription

def record_audio(duration):
    print("Recording audio...")
 
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()  
    audio_data = audio_data.flatten() 
    return audio_data

def load_audio(file_path):
    speech_array, sampling_rate = torchaudio.load(file_path)
    return speech_array, sampling_rate

def transcribe_file(file_path):
    speech_array, sampling_rate = load_audio(file_path)
    resampler = torchaudio.transforms.Resample(sampling_rate, SAMPLE_RATE)
    speech_array = resampler(speech_array).squeeze().numpy()
    
    return transcribe_audio(speech_array)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record():

    audio_data = record_audio(DURATION)
    audio_data = audio_data / np.max(np.abs(audio_data)) 
    transcription = transcribe_audio(audio_data)
    return render_template('result.html', transcription=transcription)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    transcription = transcribe_file(file_path)
    os.remove(file_path)
    
    return render_template('result.html', transcription=transcription)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, use_reloader=False)
