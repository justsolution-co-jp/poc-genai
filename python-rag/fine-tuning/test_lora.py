from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# ---------------------------
# 1. 配置模型名称
# ---------------------------
model_name = "uer/gpt2-chinese-cluecorpussmall"  # 训练时用的基座模型
lora_path = "../phi-lora-checkpoint/lora_default"      # 保存的 ΔW 目录

# ---------------------------
# 2. 加载分词器
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Q: 公司是否有团建活动？\nA:"

# ---------------------------
# 3. Base 模型推理
# ---------------------------
base_model = AutoModelForCausalLM.from_pretrained(model_name)
pipe_base = pipeline("text-generation", model=base_model, tokenizer=tokenizer)

print("=== Base 模型输出 ===")
print(pipe_base(prompt, max_new_tokens=50)[0]["generated_text"])

# ---------------------------
# 4. LoRA 模型推理
# ---------------------------
model_lora = AutoModelForCausalLM.from_pretrained(model_name)
model_lora = PeftModel.from_pretrained(model_lora, lora_path)

pipe_lora = pipeline("text-generation", model=model_lora, tokenizer=tokenizer)

print("\n=== LoRA 微调模型输出 ===")
print(pipe_lora(prompt, max_new_tokens=50)[0]["generated_text"])
