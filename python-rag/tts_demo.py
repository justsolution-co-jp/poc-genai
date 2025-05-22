import torch
from coqui_tts.api import TTS

# 选择设备：GPU 优先，否则使用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化 TTS 模型（多语言、多说话人支持）
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 要合成的文本
text = "お疲れ様でした"

# 输出音频文件路径
output_path = "/mnt/c/Users/shicy/Downloads/tsukale.wav"

# 执行文本转语音，并保存为音频文件
tts.tts_to_file(text=text, file_path=output_path)

print(f"✅ 语音合成完成，文件已保存至：{output_path}")
