import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lora"))

import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import PromptTuningConfig, get_peft_model
from utils_lora import get_tokenizer, get_dataset

# ---------------------------
# 1. 配置模型 & 数据
# ---------------------------
model_name = "uer/gpt2-chinese-cluecorpussmall"  # 中文 GPT2 小模型
tokenizer = get_tokenizer(model_name)
dataset = get_dataset(tokenizer)  # 默认加载 company_faq.txt

# ---------------------------
# 2. 加载基座模型（冻结参数）
# ---------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,       # CPU / 单GPU
    torch_dtype=torch.float32
)

# ---------------------------
# 3. 配置 Prompt-Tuning
# ---------------------------
prompt_config = PromptTuningConfig(
    task_type="CAUSAL_LM",    # 因果语言建模
    num_virtual_tokens=20     # prompt 的虚拟 token 数量
)

model = get_peft_model(base_model, prompt_config)

# 打印可训练参数比例
model.print_trainable_parameters()

# ---------------------------
# 4. 训练参数
# ---------------------------
training_args = TrainingArguments(
    output_dir="../../phi-lora-checkpoint/prompt_demo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-4,
    fp16=False,   # CPU 下关闭混合精度
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
trainer.save_model("../../phi-lora-checkpoint/prompt_demo")
