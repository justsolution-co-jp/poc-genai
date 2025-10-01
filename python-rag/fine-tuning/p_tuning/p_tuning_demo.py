import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lora"))  # 复用 utils_lora

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import PromptEncoderConfig, get_peft_model
from utils_lora import get_tokenizer, get_dataset

# ---------------------------
# 1. 配置模型 & 数据
# ---------------------------
model_name = "uer/gpt2-chinese-cluecorpussmall"  # 中文 GPT2
tokenizer = get_tokenizer(model_name)
dataset = get_dataset(tokenizer)

# ---------------------------
# 2. 加载基座模型
# ---------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,       # 强制CPU
    torch_dtype=torch.float32
)

# ---------------------------
# 3. 配置 P-Tuning
# ---------------------------
p_tuning_config = PromptEncoderConfig(
    task_type="CAUSAL_LM",   # 因果语言建模
    num_virtual_tokens=20,   # prompt 长度
    encoder_hidden_size=128  # MLP/LSTM 的隐藏层维度
)

model = get_peft_model(base_model, p_tuning_config)

# 打印可训练参数比例
model.print_trainable_parameters()

# ---------------------------
# 4. 训练参数
# ---------------------------
training_args = TrainingArguments(
    output_dir="../../phi-lora-checkpoint/p_tuning_demo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-4,
    fp16=False,  # CPU 上关掉混合精度
    logging_steps=5,
    save_strategy="epoch"
)

# ---------------------------
# 5. Trainer
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# ---------------------------
# 6. 训练 & 保存
# ---------------------------
trainer.train()
trainer.save_model("../../phi-lora-checkpoint/p_tuning_demo")
