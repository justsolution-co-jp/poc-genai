import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lora"))

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import AdaLoraConfig, get_peft_model
from utils_lora import get_tokenizer, get_dataset

# 1. 加载模型 & 数据
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = get_tokenizer(model_name)
dataset = get_dataset(tokenizer)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,
    torch_dtype=torch.float32
)

# 2. 计算 total_step
epochs = 3
batch_size = 2
grad_accum = 4
total_step = len(dataset) // (batch_size * grad_accum) * epochs # 训练总步数
print("Total training steps:", total_step)

# 3. 配置 AdaLoRA
adapter_config = AdaLoraConfig(
    task_type="CAUSAL_LM",
    init_r=8,              # 初始 rank
    target_modules=["c_attn", "c_proj"],  
    lora_alpha=32,
    lora_dropout=0.1,
    total_step=total_step
)

model = get_peft_model(base_model, adapter_config)
model.print_trainable_parameters()

# 4. 训练参数
training_args = TrainingArguments(
    output_dir="../../phi-lora-checkpoint/adalora_demo",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=epochs,
    learning_rate=5e-4,
    fp16=False,
    logging_steps=5,
    save_strategy="epoch"
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
trainer.save_model("../../phi-lora-checkpoint/adalora_demo")
