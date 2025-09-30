# phi-lora-checkpoint 目录结构说明

本目录保存了 LoRA 微调过程中的 **训练快照** 和 **最终导出的权重**。

---

## 顶层目录结构

- **checkpoint-2053/**
  - **adapter_config.json**  
    LoRA 配置文件，定义 LoRA 参数（如 r=8, alpha=32, target_modules 等）。
  - **adapter_model.safetensors**  
    LoRA 训练得到的权重（ΔW）。新版 transformers/peft 默认保存为 safetensors 格式。
  - **added_tokens.json**  
    如果训练过程中给 tokenizer 新增了 token，这里会保存。
  - **merges.txt**  
    GPT2/BPE tokenizer 的合并规则文件。
  - **optimizer.pt**  
    优化器状态，用于继续训练时恢复。
  - **README.md**  
    Hugging Face `Trainer` 自动生成的 checkpoint 说明文件。
  - **rng_state.pth**  
    随机数种子状态，用于保证恢复训练时结果一致。
  - **scheduler.pt**  
    学习率调度器状态。
  - **special_tokens_map.json**  
    特殊 token（如 [PAD], [CLS], [SEP] 等）的映射。
  - **tokenizer_config.json**  
    分词器配置文件。
  - **tokenizer.json**  
    分词器核心文件，包含词汇表和分词规则。
  - **trainer_state.json**  
    训练过程中的状态（已完成的 step、loss 等）。
  - **training_args.bin**  
    保存 `TrainingArguments` 的序列化文件。
  - **vocab.json**  
    tokenizer 的词表文件。

---

- **lora_default/**
  - **checkpoint-3/**  
  - **checkpoint-6/**  
  - **checkpoint-9/**  
  - **checkpoint-12/**  
  - **checkpoint-15/**  
  - **checkpoint-46/**  
    👉 这些是训练过程中的中间检查点，可以用来恢复训练或比较效果。
  - **adapter_config.json**  
    LoRA 的配置文件（推理部署时需要）。
  - **adapter_model.safetensors**  
    LoRA 的权重文件（推理部署时需要）。
  - **training_args.bin**  
    训练参数快照。

---

## 使用说明

1. **继续训练**  
   - 使用 `checkpoint-*` 目录（包含 optimizer/scheduler/rng_state），能恢复训练进度。

2. **推理 / 部署**  
   - 只需要 `adapter_config.json` + `adapter_model.safetensors`。  
   - 用法示例：
     ```python
     from transformers import AutoModelForCausalLM, AutoTokenizer
     from peft import PeftModel

     base_model = AutoModelForCausalLM.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
     tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

     lora_model = PeftModel.from_pretrained(base_model, "phi-lora-checkpoint/lora_default")
     ```

3. **Tokenizer 文件**  
   - `vocab.json`, `merges.txt`, `tokenizer.json`, `tokenizer_config.json`，用于文本编码和解码。

---

✅ 总结：  
- **推理时最小需要**：`adapter_config.json` + `adapter_model.safetensors`  
- **继续训练时需要**：对应 `checkpoint-*` 目录
