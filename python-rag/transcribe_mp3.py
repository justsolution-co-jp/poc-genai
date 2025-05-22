import whisper

# mp4 to mp3
# ffmpeg -i ./1747795599.799878_ytshorts.savetube.me.mp4 -vn -acodec libmp3lame -q:a 0 output.mp3

# 加载模型（可以改为 tiny, base, small, medium, large） 准确率
model = whisper.load_model("medium")

# 输入你的音频文件路径（mp3/wav/m4a 等）
audio_file = "/mnt/c/Users/shicy/Downloads/output.mp3"

# 进行转写
# result = model.transcribe("myfile.mp3", language="ja")
result = model.transcribe(audio_file, language="ja")
print(result)
print(result["language"])
# 输出转写结果
# print("🎧 转写内容如下：\n")
# print(result["text"])
# print("🎧 转写内容（已分段）：\n")
for seg in result["segments"]:
    # start = seg["start"]
    # end = seg["end"]
    text = seg["text"].strip()

    # 输出段落时间（可选）
    # print(f"[{start:.2f} ~ {end:.2f} sec]")
    print(text)
    print()  # 空行分段
