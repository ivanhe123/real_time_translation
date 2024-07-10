import os
import time
import io
import torch
import torchaudio
from transformers import pipeline
from transformers import AutoProcessor, SeamlessM4TModel, AutoModelForSpeechSeq2Seq
import wave
import sounddevice
import speech_recognition as sr
import soundfile as sf
import numpy as np
r = sr.Recognizer()
device = torch.device("cuda:0")
processor1 = AutoProcessor.from_pretrained("./whisper-small")
model = AutoModelForSpeechSeq2Seq.from_pretrained("./whisper-small").to("cuda:0")
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor1.tokenizer, feature_extractor=processor1.feature_extractor, chunk_length_s=30, batch_size=24, torch_dtype=torch.float16, device="cuda:0")
model = model.half()
processor = AutoProcessor.from_pretrained("seamless-m4t-medium")
model_seam = SeamlessM4TModel.from_pretrained("seamless-m4t-medium").to(device)

while True:
    with sr.Microphone() as source:
        print("Listening...")

        audio_to_translate = r.listen(source)
    print("processing")

    wav_bytes = audio_to_translate.get_wav_data(convert_rate=16000)
    wav_stream = io.BytesIO(wav_bytes)
    audio_array, sampling_rate = sf.read(wav_stream)
    audio_array = audio_array.astype(np.float32)
    a = time.time()
    out_pip = pipe(audio_array, return_timestamps=True,generate_kwargs={"language":"mandarin"})
    chunks = out_pip["chunks"]
    alls = []
    b=time.time()
    tot = b-a
    print(chunks)
    for k, w in enumerate(chunks):

        words = audio_array[int(w["timestamp"][0] * 16000) : int(w["timestamp"][1] * 16000)]
        audio_inputs = processor(audios=words, return_tensors="pt").to(device)
        audio_array_from_audio = model_seam.generate(**audio_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()
        c = time.time()
        tot+=c-b
        print("infer: ", c - b)
        sounddevice.play(audio_array_from_audio, samplerate=16000)
        sounddevice.wait()
    print("total: ", tot)
    print("split: ", b-a)