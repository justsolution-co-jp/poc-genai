import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel

# ---------------------------
# 1. 加载基座模型 phi3:mini
# ---------------------------
model_name = "microsoft/phi-3-mini-4k-instruct" # 官方 Hugging Face 模型名

# 去 Hugging Face Hub 或本地缓存 下载这个模型对应的 分词器配置
# 作用：保证「输入文本」和「模型权重」是一致的，不然模型根本看不懂输入
tokenizer = AutoTokenizer.from_pretrained(model_name) 
print("pad_token:", tokenizer.pad_token)
print("pad_token_id:", tokenizer.pad_token_id)
print("all special tokens:", tokenizer.special_tokens_map)

# eos_token 来临时充当 pad
# pad_token 把不同长度的句子 补齐到相同长度，常见于批量训练或推理
# 例如：
#     "我爱AI"      → [12, 45, 67]  
#     "我爱大语言模型" → [12, 45, 78, 99, 34] 
# 转换：
#     [12, 45, 67, <PAD>, <PAD>]  
#     [12, 45, 78, 99, 34]

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,       # 压缩模型权重，减少显存需求
    device_map="auto" # 自动选择合适的硬件设备
)

# ---------------------------
# 2. 准备数据集
# ---------------------------
# "wikitext" 表示要加载的数据集名称。
# "wikitext-2-raw-v1" 表示数据集的 子版本（WikiText-2 原始文本版本）
# split="train"：表示只取训练集。WikiText 通常有 "train", "validation", "test" 三个部分
# [:1%]：这是一个 切片语法，表示只取训练集的 前 1% 数据。 "train[90%:]" 就是最后 10%
# eg: dataset = load_dataset("text", data_files="company_corpus.txt", split="train")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

# truncation=True  如果文本太长，就裁剪到 max_length。避免超过模型的输入上限
# padding="max_length"  把所有文本填充到 固定长度
# max_length=128 设置输入的最大 token 数
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

# 批量地把整个数据集转换成模型可以直接训练的格式
dataset = dataset.map(tokenize, batched=True)
# 把数据转换成 PyTorch Tensor 格式，只保留模型需要的输入列，让后续训练更高效
dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 输入 (input_ids)：[我, 爱, 北, 京, 天, 安]
# 预测目标 (labels)：[爱, 北, 京, 天, 安, 门]
dataset = dataset.map(lambda x: {"labels": x["input_ids"]})

# ---------------------------
# 3. LoRA 配置函数（可设置 ΔW）
# ---------------------------
def get_lora_model(base, r=8, alpha=32, dropout=0.1, target=["q_proj","v_proj"]):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return get_peft_model(base, config)

# ---------------------------
# 4. 默认 ΔW：新建并训练
# ---------------------------
model = get_lora_model(base_model)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./outputs_phi3_lora_default",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
trainer.save_model("lora_default")  # 保存默认 ΔW

# ---------------------------
# 5. 可设置 ΔW：新任务
# ---------------------------
# 重新加载基座模型
base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
model_A = get_lora_model(base_model, r=16, alpha=64, target=["q_proj","k_proj"])  # 配置不同

training_args.output_dir = "./outputs_phi3_lora_taskA"
trainer = Trainer(model=model_A, args=training_args, train_dataset=dataset)
trainer.train()
trainer.save_model("lora_taskA")  # 保存 ΔW-A

# ---------------------------
# 6. 重复 ΔW：继续训练同一套
# ---------------------------
base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
model_repeat = PeftModel.from_pretrained(base_model, "lora_taskA")  # 加载 ΔW-A 继续训

training_args.output_dir = "./outputs_phi3_lora_taskA_repeat"
trainer = Trainer(model=model_repeat, args=training_args, train_dataset=dataset)
trainer.train()
trainer.save_model("lora_taskA_repeat")  # 覆盖或生成新版本 ΔW

# ---------------------------
# 7. 推理：切换 ΔW
# ---------------------------
from transformers import pipeline

# 基座模型
base_model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

# 加载 ΔW-default
model_default = PeftModel.from_pretrained(base_model, "lora_default")
pipe_default = pipeline("text-generation", model=model_default, tokenizer=tokenizer)
print("=== ΔW-default ===")
print(pipe_default("The capital of France is", max_new_tokens=20)[0]["generated_text"])

# 加载 ΔW-taskA
model_A = PeftModel.from_pretrained(base_model, "lora_taskA")
pipe_A = pipeline("text-generation", model=model_A, tokenizer=tokenizer)
print("=== ΔW-taskA ===")
print(pipe_A("The capital of France is", max_new_tokens=20)[0]["generated_text"])