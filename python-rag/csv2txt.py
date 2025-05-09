import pandas as pd

# 读取 CSV（文件名请根据实际替换）
df = pd.read_csv("faqs.csv")  # 如果在 Jupyter 可使用 upload 确认路径

# 写入 document.txt
with open("document.txt", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(f"[EN] {row['en']}\n[ZH] {row['zh']}\n[JA] {row['ja']}\n\n")

print("✅ 已成功生成 document.txt 文件！")
