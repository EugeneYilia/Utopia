from concurrent.futures import ProcessPoolExecutor
import asyncio

import TTSConverter

max_workers = 3
executor = ProcessPoolExecutor(max_workers=max_workers)

# 主程序启动时执行：让每个子进程先初始化一次 TTS
def preload_tts_model_in_all_workers():
    futures = []
    for _ in range(max_workers):
        future = executor.submit(TTSConverter.init_model)
        futures.append(future)
    # 等待所有初始化完成
    for f in futures:
        f.result()


async def get_audio(voice_id: str, text: str, output_path: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, TTSConverter.generate_tts_audio, voice_id, text, output_path)
