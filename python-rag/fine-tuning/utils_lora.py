from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

IGNORE_INDEX = -100

def get_tokenizer(model_name):
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
    return tokenizer

# "wikitext" 表示要加载的数据集名称。
# "wikitext-2-raw-v1" 表示数据集的 子版本（WikiText-2 原始文本版本）
# split="train"：表示只取训练集。WikiText 通常有 "train", "validation", "test" 三个部分
# [:1%]：这是一个 切片语法，表示只取训练集的 前 1% 数据。 "train[90%:]" 就是最后 10%
# eg: dataset = load_dataset("text", data_files="company_corpus.txt", split="train")
def get_dataset(tokenizer, split="train[:1%]"):
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    dataset = load_dataset("text", data_files={"train": "company_faq.txt"})["train"]
    
    # truncation=True  如果文本太长，就裁剪到 max_length。避免超过模型的输入上限
    # padding="max_length"  把所有文本填充到 固定长度
    # max_length=128 设置输入的最大 token 数
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    dataset = dataset.map(tokenize, batched=True)

    # 忽略 padding 的 label，避免把 <pad> 位置当作学习目标。
    def mask_pad(batch):
        labels = []
        for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
            labels.append([tok if m==1 else IGNORE_INDEX for tok, m in zip(ids, mask)])
        batch["labels"] = labels
        return batch

    dataset = dataset.map(mask_pad, batched=True)
    # 把数据转换成 PyTorch Tensor 格式，只保留模型需要的输入列，让后续训练更高效
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

# 给基座模型挂上 LoRA 适配器（低秩矩阵 ΔW），能用很少的参数和算力对模型进行微调。
# qkv_proj / o_proj 要根据模型结构实际 内容来填写
def get_lora_model(base, r=8, alpha=32, dropout=0.1, target=["c_attn","c_proj"]):
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target, # 在这些层加 LoRA
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # 遍历 base_model 所有参数，把它们的 requires_grad = False（即冻结）
    # 只允许 LoRA 新增的参数参与训练
    # 会在 q_proj、v_proj 层旁边挂上低秩矩阵 A、B，形成 ΔW
    return get_peft_model(base, config)
