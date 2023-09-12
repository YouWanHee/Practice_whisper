import io
import whisper
import torch
import math
import time
from faster_whisper import WhisperModel

audio_model = WhisperModel("large-v2", device="cuda", compute_type="float16")
wav_data = "record2.wav"

def perform_speech_recognition(audio_data):
    
    global audio_model
    segments, info = audio_model.transcribe(audio_data, language="ko", initial_prompt="", vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
    return segments

def main():
    start = time.time()
    result = perform_speech_recognition(wav_data)
    for segment in result:
     print(segment.text.strip())
    end = time.time()
    print(f"{end - start:.5f} sec")
    
if __name__ == "__main__":
    main()
