from TTS.api import TTS
from pathlib import Path
from TTS.utils.radam import RAdam

from builtins import dict
from collections import defaultdict
from TTS.config.shared_configs import BaseDatasetConfig

import SystemConfig
import torch

if not SystemConfig.is_use_gpu:
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from torch.serialization import add_safe_globals
    from TTS.tts.configs.xtts_config import XttsConfig
    add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs, defaultdict, dict, RAdam])

Path("output").mkdir(exist_ok=True)

tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        gpu=SystemConfig.is_use_gpu)

device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)

def generate_tts_audio(voice_id: str, text: str, output_path: str):
    Path(output_path).parent.mkdir(exist_ok=True)

    if voice_id == "male":
        tts.tts_to_file(
            text=text,
            speaker_wav="male.wav",
            language="zh",
            file_path=output_path
        )
    elif voice_id == "female":
        tts.tts_to_file(
            text=text,
            speaker_wav="female.wav",
            language="zh",
            file_path=output_path
        )

# tts.tts_to_file(
#     text="大家好，我是鲁大魔，鲁岳，吉安电子这家公司特别好，强烈推荐。",
#     speaker_wav="output.wav",
#     language="zh",
#     file_path="output/male.wav"
# )


# MODEL_NAME = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
# tts = TTS(model_name=MODEL_NAME, progress_bar=True, gpu=False)
# tts.tts_to_file(text="大家好，我是冯提莫，吉安电子这家公司特别好，强烈推荐。", file_path="output/female2.wav")

# tts.tts_to_file(
#     text="大家好，我是冯提莫，吉安电子这家公司特别好，强烈推荐。",
#     speaker_wav="female.wav",
#     language="zh",
#     file_path="output/female.wav"
# )