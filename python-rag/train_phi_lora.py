from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# 加载 Wikipedia 数据集（英文版简化）
dataset = load_dataset("wikipedia", "20220301.simple", split="train[:1%]")  # 只取1%用于demo

# 加载 tokenizer 和模型
model_id = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# 应用 LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Tokenize 数据集
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 训练参数
training_args = TrainingArguments(
    output_dir="./phi-lora-checkpoint",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    fp16=False,  # 若无 GPU 支持 FP16，可改为 False
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 开始训练
trainer.train()
