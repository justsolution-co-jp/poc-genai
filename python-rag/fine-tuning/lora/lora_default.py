import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from utils_lora import get_tokenizer, get_dataset, get_lora_model

# 官方 Hugging Face 模型名
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = get_tokenizer(model_name)
dataset = get_dataset(tokenizer)

# 下载完整的基座模型参数
# phi-3-mini 本身有 38 亿参数 (3.8B)
# 假设每个参数用 FP16（16 bit = 2 字节）存储：3.8B × 2 Bytes ≈ 7.6 GB（需下载）
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # load_in_8bit=True,       # 压缩模型权重，减少显存需求（必须有GPU）  / 默认：FP32/FP16
    device_map=None, # 自动选择合适的硬件设备 / 不要自动 强制CPU
    torch_dtype="float32" # 强制加载到 CPU
)
# 最好有这一步，确认模型结构信息
print(base_model)

model = get_lora_model(base_model)

# 打印“可训练参数/总参数/占比”，用来确认 只有 LoRA 参数在训练（应≈0.1%～1%）
print("=== 可训练参数/总参数/占比” ===")
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="../phi-lora-checkpoint/lora_default", # Trainer 的“工作目录”（日志、checkpoint 等会写在这里）
    per_device_train_batch_size=2, # 每张设备（GPU）每步喂多少条样本
    gradient_accumulation_steps=4, # 做 梯度累积。等价于把 4 小步合成 1 大步再 optimizer.step()。 有效 batch size = 2（每设备） × 4（累积步） × 设备数。单卡时就是 2×4=8。
    num_train_epochs=5, # 训练 1 轮
    learning_rate=5e-4, # LoRA 的常见学习率量级（可在 1e-4 ~ 5e-4 间微调）
    fp16=True, # 混合精度（需要支持 FP16 的 GPU）
    logging_steps=5, # 每隔 5 个 step 打一次日志（loss 等）
    save_strategy="epoch" # 每个 epoch 结束自动保存一次 checkpoint 到 output_dir
)

trainer = Trainer(
    model=model, # “基座 + LoRA”的模型
    args=training_args, # 需要至少包含 input_ids、attention_mask、labels
    train_dataset=dataset # 需要至少包含 input_ids、attention_mask、labels
)

# 进入训练循环
trainer.train()
trainer.save_model("../phi-lora-checkpoint/lora_default")
