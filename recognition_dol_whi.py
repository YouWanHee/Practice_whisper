import io
import whisper
import torch
import math
import time

start = time.time()
audio_model = whisper.load_model("./large-v2.pt")
wav_data = "./record3_dol.wav"

def perform_speech_recognition(audio_data):
    
    global audio_model, temp_file

    result = audio_model.transcribe(audio_data, language="ko", fp16=True)
    
    return result

def main():
    result = perform_speech_recognition(wav_data)
    text = result['text'].strip()
    print(text)

if __name__ == "__main__":
    main()
    end = time.time()
    print(f"{end - start:.5f} sec")


#파라미터로 줄이는 건 힘들다
#물리적 서버 거리로 더 줄이기는 어렵다.
