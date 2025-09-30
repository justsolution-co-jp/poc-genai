# 📘 Fine-Tuning with LoRA (phi3-mini)

本目录包含 **LoRA 微调脚本**，基于 `microsoft/phi-3-mini-4k-instruct` 模型，演示如何通过 **低秩适配器 (ΔW)** 进行轻量化微调。  

每个脚本对应一种场景，结果会保存到 `../phi-lora-checkpoint/` 目录中。  

---

## 📁 目录结构

```
fine-tuning/
├── utils_lora.py           # 公共函数（Tokenizer、数据集、LoRA 配置）
├── lora_default.py         # 训练默认 ΔW
├── lora_taskA.py           # 训练任务A ΔW
├── lora_taskA_repeat.py    # 在 ΔW-A 基础上继续训练
├── lora_infer_default.py   # 推理：加载 ΔW-default
├── lora_infer_taskA.py     # 推理：加载 ΔW-taskA
└── README.md               # 本文件
```

---

## 🚀 环境准备

1. **进入虚拟环境**
```bash
source ../venv/bin/activate   # Linux / macOS
# 或
..\venv\Scripts\activate      # Windows
```

2. **安装依赖**
```bash
pip install -r ../requirements.txt
```

3. **检查 GPU / 8-bit 支持**
```bash
pip install bitsandbytes accelerate
```

---

## 🛠 脚本说明

### 1️⃣ 训练默认 ΔW
```bash
cd fine-tuning
python lora_default.py
```
- 输出目录：`../phi-lora-checkpoint/lora_default`
- 使用 `q_proj` + `v_proj` 作为 LoRA 插入点  

---

### 2️⃣ 新任务 ΔW-A
```bash
cd fine-tuning
python lora_taskA.py
```
- 输出目录：`../phi-lora-checkpoint/lora_taskA`
- 使用 `q_proj` + `k_proj` 作为 LoRA 插入点  

---

### 3️⃣ 在 ΔW-A 基础上继续训练
```bash
cd fine-tuning
python lora_taskA_repeat.py
```
- 输出目录：`../phi-lora-checkpoint/lora_taskA_repeat`
- 会加载已有 `lora_taskA` 并进一步训练  

---


#### 任务A ΔW
```bash
cd fine-tuning
python lora_infer_taskA.py
```

---

## 📊 训练效果检查

- 查看可训练参数占比：
```python
model.print_trainable_parameters()
```
期望输出：
```
trainable params: ~0.1% - 1%
```

- 日志与 checkpoint 会自动保存到 `output_dir`。  

---

## 📌 注意事项

- `ΔW`（LoRA 权重）是独立文件，不会修改原始模型。  
- 不同任务可以有不同的 ΔW，推理时可随时切换。  
- `../phi-lora-checkpoint/` 下的文件夹就是对应的 ΔW 保存位置。  
