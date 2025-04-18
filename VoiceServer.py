from fastapi import FastAPI, Query
from pydantic import BaseModel
import base64
import uuid
from pathlib import Path
from pydub import AudioSegment
import os

import TTSConverter
from fastapi.responses import JSONResponse
import logging

from TTSPool import TTSPool

logger = logging.getLogger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

tts_pool = TTSPool(num_workers=3)

app = FastAPI()
class TTSRequest(BaseModel):
    text: str
    voice_id: str

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    logger.info("Received TTS request: %s", request)
    # Create a unique filename
    output_dir = Path("output")
    output_filename = f"{uuid.uuid4().hex}"

    raw_wav_path = output_dir / f"{output_filename}_raw.wav"  # 初始语音输出
    final_wav_path = output_dir / f"{output_filename}.wav"  # 转换后的音频输出

    # await TTSWorker.get_audio(request.voice_id, request.text, str(raw_wav_path))
    await tts_pool.generate(request.voice_id, request.text, str(raw_wav_path))

    # 强制转换为 16kHz 单声道 WAV（确保兼容性）
    sound = AudioSegment.from_file(raw_wav_path)
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound.export(final_wav_path, format="wav")

    # Read and return base64 audio
    with open(final_wav_path, "rb") as audio_file:
        audio_value = audio_file.read()

    os.remove(raw_wav_path)
    # os.remove(final_wav_path)

    logger.info("Audio file generated and removed: %s", raw_wav_path)

    base64_audio =  base64.b64encode(audio_value).decode("utf-8")
    return JSONResponse(content={"audio": base64_audio})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "VoiceServer:app",
        host="0.0.0.0",
        port=8118,
        reload=False,
        log_config="log_config.yml"
    )