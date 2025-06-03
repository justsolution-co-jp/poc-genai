import re
import os
from TTS.api import TTS

# 初始化一次模型（支持中文）
tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST")

def is_code(text: str) -> bool:
    """判断是否为代码块"""
    return bool(re.search(r"```[\s\S]+?```", text)) or bool(re.search(r"\b(def|class|import|print|for|if)\b", text))

def is_too_short(text: str, min_chars: int = 5) -> bool:
    """判断内容是否太短或缺乏语义"""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True
    if not any(p in stripped for p in "。？！?,.!，"):
        return True
    return False

def safe_speak(text: str, output_path: str = "output.wav") -> bool:
    """
    安全发音函数：
    - 过滤空内容、代码、太短句子
    - 合成有效语音并保存为 WAV 文件
    - 返回 True 表示成功发音；False 表示跳过
    """
    text = text.strip()
    if not text:
        print("⚠️ 空文本，跳过发音。")
        return False

    if is_code(text):
        print("⚠️ 检测到代码内容，跳过发音。")
        return False

    if is_too_short(text):
        print(f"⚠️ 文本太短或缺乏语义，跳过发音：'{text}'")
        return False

    print(f"🗣️ 发音中：{text}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tts.tts_to_file(text=text, file_path=output_path)
    print(f"✅ 音频已保存：{output_path}")
    return True
