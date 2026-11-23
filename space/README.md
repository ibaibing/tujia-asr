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

- 模型：`TujiaNLP/tujia-ctc-v0.4`（932 类去调音节头 + 四调头）+ `TujiaNLP/tujia-xlsr53-v03f`（编码器）
- 代码：https://github.com/ibaibing/tujia-asr

> v0.2 与 v0.3 的音节表不同（709 vs 1199 类），v0.4 改为 932 个去调音节 + 独立调头；同一段音频的输出不能直接逐字对比。

## 界面参数

| 参数 | 默认 | 说明 |
|---|---|---|
| 输出密度（`blank_bias`） | 0.0 | 调高减少漏字、可能多出噪声音节；v0.4 删除率已很低，默认无需调整 |
| 声调校正（辅助头） | 开 | 关闭后只用主 CTC 头自带的调号；与本地 `infer.py` 对比时请关闭 |
| 标黄阈值 | 0.55 | 音节置信度低于此值标黄，只影响显示 |

长音频按 10 秒分窗、相邻窗重叠 2 秒解码，单次上限 5 分钟。

## MCP / HTTP API（手机客户端）

公开端点只有一个：`transcribe`。它返回 JSON，不接受 multipart 上传。

```text
POST /gradio_api/call/transcribe
{"data": ["data:audio/wav;base64,....", 1.0, true, 300]}
```

MCP 端点在 `https://<owner>-<name>.ms.show/gradio_api/mcp/sse`（另有 Streamable
HTTP `/gradio_api/mcp/`）。客户端需要带浏览器 `User-Agent`，否则平台网关返回 403。

**为什么要传 base64**：容器没有公网出口，且 `ms.show` 的文件域名对非浏览器 UA
返回 403。把 `gr.Audio` 的 URL 交给服务端下载会失败，所以音频必须以
`data:audio/...;base64,...` 的形式随请求传入。

返回结构（`schema` 版本 1）：

```json
{
  "schema": 1,
  "ok": true,
  "meta": {"model": "tujia-ctc-v0.4.pt", "encoder": "tujia-xlsr53-v03f",
           "duration": 6.6, "n_windows": 1, "elapsed_s": 1.8,
           "blank_bias": 0.0, "tone_correct": true, "low_conf": 0.55,
           "n_syllables": 22, "n_low": 14},
  "text": "kai35 ta53 tɕiau35 ...",
  "text_sup": "kai³⁵ ta⁵³ tɕiau³⁵ ...",
  "syllables": [{"syl": "kai", "tone": "35", "score": 0.757,
                 "tone_conf": 0.566, "start": 0.0, "end": 0.22}],
  "segments": [{"start": 0.0, "end": 6.46, "text": "kai35 ...", "text_sup": "kai³⁵ ..."}]
}
```

失败时返回 `{"schema": 1, "ok": false, "error": "...", "syllables": []}`，HTTP 仍是 200。

## 许可边界

- 本创空间的代码为 **MIT**。
- 模型权重与微调编码器仅供**研究用途**，不授予训练录音及其转写的任何权利。
- 上游编码器 `facebook/wav2vec2-xlsr-53-espeak-cv-ft` 为 **Apache-2.0**，须保留署名。
- 详见主仓库 `NOTICE`。

模型环境变量（如果仓库改了名）：

| 变量 | 默认值 |
|---|---|
| `TUJIA_CTC_REPO` | `TujiaNLP/tujia-ctc-v0.4` |
| `TUJIA_ENC_REPO` | `TujiaNLP/tujia-xlsr53-v03f` |
| `TUJIA_CTC_DIR` | 空（本地调试时指向含 `.pt` 的目录） |
| `TUJIA_ENC_DIR` | 空（本地调试时指向编码器目录） |
