import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------
# 配置参数
# ---------------------------
# 基座模型（要和微调时用的一致）
base_model_name = "uer/gpt2-chinese-cluecorpussmall"

# LoRA 适配器路径（训练时保存的目录）
lora_path = "../phi-lora-checkpoint/lora_taskA"

# 合并后保存的目录
merged_model_path = "../phi-lora-checkpoint/lora_taskA_merged"

# ---------------------------
# 1. 加载分词器和基座模型
# ---------------------------
print(f"加载基座模型: {base_model_name}")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,
    device_map=None  # 强制CPU合并
)

# ---------------------------
# 2. 加载 LoRA 并合并
# ---------------------------
print(f"加载 LoRA 权重: {lora_path}")
model = PeftModel.from_pretrained(base_model, lora_path)

print("正在合并 LoRA 权重到基座模型...")
model = model.merge_and_unload()

# ---------------------------
# 3. 保存完整合并后的模型
# ---------------------------
print(f"保存合并后的模型到: {merged_model_path}")
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print("✅ 合并完成！")
print("现在可以用 AutoModelForCausalLM.from_pretrained(merged_model_path) 直接加载。")
