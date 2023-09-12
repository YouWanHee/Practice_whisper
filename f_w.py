from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import time

# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-large", dtype=jnp.float16)

# JIT compile the forward call - slow, but we only do once
start_time = time.time()
text = pipeline("record_small.wav", language="ko", task="transcribe")
end_time = time.time()
execution_time = end_time - start_time

print("Transcribed text:", text)
print("Execution time:", execution_time, "seconds")

# used cached function thereafter - super fast!!
start_time2 = time.time()
text2 = pipeline("record3_dol.wav", language="ko", task="transcribe")
end_time2 = time.time()
execution_time2 = end_time2 - start_time2

print("Transcribed text:", text2)
print("Execution time:", execution_time2, "seconds")

# used cached function thereafter - super fast!!
start_time3 = time.time()
text3 = pipeline("record2.wav", language="ko", task="transcribe")
end_time3 = time.time()
execution_time3 = end_time3 - start_time3

print("Transcribed text:", text3)
print("Execution time:", execution_time3, "seconds")

