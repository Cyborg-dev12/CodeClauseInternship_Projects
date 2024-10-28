# from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
# import torchaudio
# import torch
# tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# def load_audio(file_path):
#     speech_array, sampling_rate = torchaudio.load(file_path)
#     return speech_array, sampling_rate

# def transcribe_audio(file_path):
#     speech_array, sampling_rate = load_audio(file_path)
#     # Resample if necessary (if the sampling rate is not 16kHz)
#     resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
#     speech_array = resampler(speech_array).squeeze().numpy()
#     inputs = tokenizer(speech_array, return_tensors="pt", padding="longest")
    
#     with torch.no_grad():
#         logits = model(inputs.input_values).logits
    
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = tokenizer.decode(predicted_ids[0])
    
#     print(f"Transcription: {transcription}")

# if __name__ == "__main__":
#     transcribe_audio("C:\\Users\\herow\\Downloads\\audio1.mp3")
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torchaudio
import torch
import numpy as np
import sounddevice as sd

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

    print(f"Transcription: {transcription}")

def record_audio(duration):
    print("Recording audio...")

    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait() 
    audio_data = audio_data.flatten()  
    return audio_data

if __name__ == "__main__":
    audio_data = record_audio(DURATION)
    audio_data = audio_data / np.max(np.abs(audio_data))
    transcribe_audio(audio_data)
