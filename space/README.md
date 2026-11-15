---
title: Tujia ASR
emoji: 🗣️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
license: mit
short_description: Tujia speech to toned syllables, with confidence highlighting
---

# 土家语语音转写台

音节级 CTC 语音识别，输出带调音节序列（如 `la⁵⁵`），并对低置信度音节标黄提示人工核对。

- 模型：`TujiaNLP/tujia-ctc-v0.2`（CTC 头）+ `TujiaNLP/tujia-xlsr53-ft`（XLSR-53 微调编码器）
- 代码：https://github.com/ibaibing/tujia-asr

## 许可边界

- 本创空间的代码为 **MIT**。
- 模型权重与微调编码器仅供**研究用途**，不授予训练录音及其转写的任何权利。
- 上游编码器 `facebook/wav2vec2-xlsr-53-espeak-cv-ft` 为 **Apache-2.0**，须保留署名。
- 详见主仓库 `NOTICE`。

模型环境变量（如果仓库改了名）：

| 变量 | 默认值 |
|---|---|
| `TUJIA_CTC_REPO` | `TujiaNLP/tujia-ctc-v0.2` |
| `TUJIA_ENC_REPO` | `TujiaNLP/tujia-xlsr53-ft` |
