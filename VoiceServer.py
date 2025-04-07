from fastapi import FastAPI, Query
from pydantic import BaseModel
import base64
import uuid
from pathlib import Path
from pydub import AudioSegment
import os
from TTSConverter import generate_tts_audio

app = FastAPI()
class TTSRequest(BaseModel):
    text: str
    is_male: bool

@app.post("/tts")
def generate_tts(request: TTSRequest):
    # Create a unique filename
    output_dir = Path("output")
    output_filename = f"{uuid.uuid4().hex}"

    raw_wav_path = output_dir / f"{output_filename}_raw.wav"  # 初始语音输出
    final_wav_path = output_dir / f"{output_filename}.wav"  # 转换后的音频输出

    generate_tts_audio(request.is_male, request.text, str(raw_wav_path))

    # 强制转换为 16kHz 单声道 WAV（确保兼容性）
    sound = AudioSegment.from_file(raw_wav_path)
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound.export(final_wav_path, format="wav")

    # Read and return base64 audio
    with open(final_wav_path, "rb") as audio_file:
        audio_value = audio_file.read()

    os.remove(raw_wav_path)
    os.remove(final_wav_path)

    return base64.b64encode(audio_value).decode("utf-8")