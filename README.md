# 🧬 DPO-Based Text Simplification for Biological Passages

![模型框架图](./model_framework.png)


本项目旨在构建一个适用于生物医学文本的 **DPO（Direct Preference Optimization）文本简化系统**，目标是将学术或科研风格的英文段落简化为高中生可理解的中文文本。主要能力包括：

* 将英文段落翻译为中文（使用 DeepSeek API）。
* 提取文本中的生物学名词（命名实体 / 关键词抽取，使用 DeepSeek API）。
* 评估高中生可理解度（基于 text2vec + 高中词汇库）。
* 使用 **DPO + LoRA 微调** 对话模型，实现面向高中生的文本简化。
* 评估生成文本的语义相似性与语义完整度（SentenceTransformer 等）。
* 提供交互式命令行输入，实时处理文本。

## 🧾 目录

* [特性](#特性)
* [模型与依赖](#模型与依赖)
* [安装（推荐 Conda 环境）](#安装推荐-conda-环境)
* [资源准备](#资源准备)
* [快速开始（推理）](#快速开始推理)
* [配置说明](#配置说明)

---

## 特性

* 将科研/学术英文段落翻译并简化为面向高中生的中文文本。
* 基于高中词汇表与向量相似度判断句子可理解性。
* 使用 DPO 损失与 LoRA 权重对模型进行高效微调（低算力友好）。
* 可评估生成文本的语义相似性与语义完整度以保证信息不丢失。
* 提供易用的推理脚本与交互式终端。

---

## 模型与依赖

* **基础对话模型**：DeepSeek-V2-Lite-Chat
* **微调方式**：LoRA 权重（PEFT） + DPO 训练流程
* **语义评估**：SentenceTransformer（英文/中文语义模型）
* **中文句向量**：text2vec

---

## 安装（推荐 Conda 环境）

建议使用 Conda 创建隔离环境以避免依赖冲突。

### 1) 创建并激活环境

```bash
conda create -n dpo-env python=3.9 -y
conda activate dpo-env
```

### 2) 安装 PyTorch（示例：CUDA 12.1）

如果你有支持的 GPU，建议安装对应 CUDA 版本的 PyTorch, ：

```bash
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3) 安装核心依赖

```bash
pip install \
    transformers==4.41.2 \
    accelerate==0.33.0 \
    peft==0.11.1 \
    trl==0.9.6 \
    sentencepiece \
    datasets \
    sentence-transformers \
    tqdm \
    openai \
    requests \
    numpy
```

---

## 资源准备

项目需要以下资源文件（示例路径请根据你本地调整）：

```
text_resource/
│── texts_words.json   # 高中词库（词 -> 频次或词表形式）
│── hs_emb.npy         # 高中词库对应 embedding（numpy 格式）
```

示例代码中默认路径：

```python
hs_emb_path = 'datasets/textbook_resource/hs_emb.npy'
hs_words_dict_path = 'datasets/textbook_resource/texts_words.json'
```

请将以上路径替换为你本地实际路径，或在配置文件中修改。

---

## 快速开始（推理）

运行推理脚本：

```bash
python run.py
```

运行后将进入交互式模式：

```
=== 文档处理系统 ===
输入 'quit' 或 '退出' 结束程序
```

在交互式终端中，你可以粘贴英文段落（或中文），脚本将依次完成：翻译 → 生物名词提取 → 可理解度评估 → 简化生成 → 语义评估。

---

## 配置说明

* `inference.py`：推理主程序，负责加载模型、LoRA 权重、句向量模型与高中词库。
* 模型权重路径：在脚本中指定基础模型（例如 DeepSeek-V2-Lite-Chat）的路径与 LoRA 权重路径。
* 高中词库与 embedding：确保 `hs_emb_path` 与 `hs_words_dict_path` 指向正确的文件。
* 若使用 OpenAI 或第三方 API，请在环境变量或配置文件中配置 API Key（不要硬编码到代码中）。

示例环境变量加载（bash）：

```bash
export OPENAI_API_KEY="your_api_key_here"
```

---


