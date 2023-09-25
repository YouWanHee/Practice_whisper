import io
import whisper
import torch
import math
import time
from faster_whisper import WhisperModel

audio_model = WhisperModel("large-v2", device="cpu", compute_type="int8")
wav_data = "./wav/5e362cdc8661d6073410fb12.wav"

def perform_speech_recognition(audio_data):
    
    global audio_model
    segments, info = audio_model.transcribe(audio_data, language="ko", initial_prompt="")
    return segments

def main():
    start = time.time()
    result = perform_speech_recognition(wav_data)
    end = time.time()
    print(f"{end - start:.5f} sec")
#   start2 = time.time()
    for segment in result:
     print(segment.text.strip())
#    print(f"{end2 - start:.5f} sec")

if __name__ == "__main__":
    main()


#thread 개수 지정하는 명령어 포함
#OMP_NUM_THREADS=8 python3 my_script.py
