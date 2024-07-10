import os
import time
from melo.text.english_utils import number_norm
import torch
from transformers import pipeline
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import wave
import sounddevice
import translate
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import speech_recognition as sr
import soundfile as sf
import io
from scipy.io.wavfile import read
from melo.api import TTS
import numpy as np
# Speed is adjustable
speed = 1.0


ckpt_converter = 'checkpoints_v2/converter'
device = "cuda:0"
output_dir = 'outputs_v2'
print("loading whisper")
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3").to("cuda:0")
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, torch_dtype=torch.float16, chunk_length_s=30, batch_size=24, device="cuda:0")
model = model.half()

print("loading voice cloner")
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
print("loading finished")
os.makedirs(output_dir, exist_ok=True)
r = sr.Recognizer()
times=0
while True:
    with sr.Microphone() as source:
        print("Listening...")

        audio_to_translate = r.listen(source)
    print("processing")
    clone_sample = f"input.wav"
    with wave.open(clone_sample, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(audio_to_translate.sample_width)
        f.setframerate(48000)  # You may need to adjust this depending on your microphone
        f.writeframes(audio_to_translate.get_raw_data())

    wav_bytes = audio_to_translate.get_wav_data(convert_rate=16000)
    wav_stream = io.BytesIO(wav_bytes)
    audio_array, sampling_rate = sf.read(wav_stream)
    audio_array = audio_array.astype(np.float32)
    a=time.time()

    translator=translate.Translator(from_lang="zh",to_lang="en")
    a_recogn=time.time()
    text_english = pipe(audio_array, generate_kwargs={"language":"mandarin"})['text']
    text_english=translator.translate(text_english)
    text_english = number_norm.normalize_numbers(text_english)
    b_recogn=time.time()
    print(text_english)
    reference_speaker = clone_sample # This is the voice you want to clone

    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=False)
    c_extract=time.time()
    src_path = f'tmp.wav'

    # Speed is adjustable

    model_tts_2 = TTS(language='EN', device=device)
    speaker_ids = model_tts_2.hps.data.spk2id

    output_path = 'zh.wav'
    model_tts_2.tts_to_file(text_english, speaker_ids['EN-US'], src_path, speed=speed)
    d_tts = time.time()

    source_se = torch.load(f'checkpoints_v2/base_speakers/ses/zh.pth', map_location=device)
    save_path = f'translated.wav'

    # Run the tone color converter
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message="")
    e_clone = time.time()
    b=time.time()
    print("time spend:",b-a)
    print("Recognition:",b_recogn-a_recogn)
    print("Voice Extraction:",c_extract-b_recogn)
    print("TTS:",d_tts-c_extract)
    print("Cloning:",e_clone-d_tts)
    fs, data = read(save_path)
    sounddevice.play(data, fs)
    sounddevice.wait()
