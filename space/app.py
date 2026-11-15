#!/usr/bin/env python3
"""土家语语音转写台 —— Gradio 应用（Hugging Face Spaces / ModelScope 创空间通用）。

设计要点
--------
低资源语言不可能做到全自动正确，所以界面把「不确定」显式化：
每个音节都带置信度，低置信度标黄，用户可以直接看到模型在哪里没把握。

模型不在浏览器里跑：启动时把编码器与 CTC 头拉到容器本地，CPU 常驻推理。
"""

from __future__ import annotations

import html
import os
import re
import sys
import tempfile
from pathlib import Path

import gradio as gr

# Gradio 6 / gradio_client 1.x 的 schema 解析 bug：Audio / File 组件的
# additionalProperties 是 bool，get_api_info() 会抛 TypeError，导致 /info 返回 500、
# 前端组件初始化失败。这里在解析失败时返回空文档——界面与事件交互不依赖它。
import gradio.blocks as _gr_blocks  # noqa: E402

_orig_api_info = _gr_blocks.Blocks.get_api_info


def _safe_api_info(self, *args, **kwargs):
    try:
        return _orig_api_info(self, *args, **kwargs)
    except Exception:  # noqa: BLE001
        return {"named_endpoints": {}, "unnamed_endpoints": {}}


_gr_blocks.Blocks.get_api_info = _safe_api_info

HERE = Path(__file__).resolve().parent

CKPT_REPO = os.environ.get("TUJIA_CTC_REPO", "TujiaNLP/tujia-ctc-v0.3")
ENC_REPO = os.environ.get("TUJIA_ENC_REPO", "TujiaNLP/tujia-xlsr53-v03f")
# v0.3 在弱音与连读处过量输出 blank（held-out 删除率 41%、长度只有参考的 62%）。
# 校对场景里「看得见的错字」比「整段消失」有用，压 1.0 后删除率降到 31%，
# SER 只差 0.6 个点。
os.environ.setdefault("TUJIA_BLANK_BIAS", "1.0")
MAX_SEC = 15.0          # 免费 CPU 上再长会等太久
LOW_CONF = 0.55         # 低于此分数的音节标黄

SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
TONE_RE = re.compile(r"(\d+)$")
_front = None


def _snapshot(repo: str) -> str:
    """优先 ModelScope（国内快），失败回退 Hugging Face。"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        return snapshot_download(repo)
    except Exception:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo_id=repo)


def _resolve(env_dir: str, repo: str) -> str:
    """本地目录优先（调试用），否则从模型仓库拉取。"""
    d = os.environ.get(env_dir, "")
    if d and Path(d).is_dir():
        return d
    return _snapshot(repo)


def get_front():
    global _front
    if _front is None:
        ckpt_dir = _resolve("TUJIA_CTC_DIR", CKPT_REPO)
        enc_dir = _resolve("TUJIA_ENC_DIR", ENC_REPO)
        ckpt = next(Path(ckpt_dir).rglob("*.pt"), None)
        if ckpt is None:
            raise FileNotFoundError(f"{CKPT_REPO} 里没有 .pt 权重")
        from tujia_asr.ctc import CtcFront
        _front = CtcFront.get(ckpt=str(ckpt), encoder=enc_dir, device="cpu")
    return _front


def _load_16k(path: str) -> str:
    """统一成 16 kHz 单声道，并截到 MAX_SEC，避免前端格式差异影响特征。"""
    import librosa
    import soundfile as sf

    y, _ = librosa.load(path, sr=16000, mono=True, duration=MAX_SEC)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, 16000)
    return tmp.name


def _split_tone(label: str):
    """v0.2 主头输出的标签自带调号（如 ``la35``）；没有的（v0.1）再借辅助头。"""
    m = TONE_RE.search(label)
    if m:
        return label[:m.start()], m.group(1)
    return label, ""


def _decode(path: str):
    front = get_front()
    toks = front.decode(front.encode(path))
    items = []
    for vid, f0, f1, score, tone_id, tone_conf in toks:
        label = front.vocab[vid]
        if label == "<sep>" or not label.strip():
            continue
        base, tone = _split_tone(label)
        if not tone:
            tone = front._tone_of(tone_id) or ""
        items.append({
            "syl": base,
            "tone": tone,
            "score": round(float(score), 3),
            "tone_conf": round(float(tone_conf), 3),
            "start": round(f0 * front.frame_sec, 2),
            "end": round((f1 + 1) * front.frame_sec, 2),
        })
    return items


def _pick(audio_path, file_path):
    """录音与文件上传二选一；Gradio 各版本对 gr.File 的返回类型不一致，这里统一。"""
    if file_path:
        if isinstance(file_path, list):
            file_path = file_path[0] if file_path else None
        if isinstance(file_path, dict):
            file_path = file_path.get("path") or file_path.get("name")
        if file_path:
            return file_path
    return audio_path


def transcribe(audio_path: str, file_path: str = None):
    path_in = _pick(audio_path, file_path)
    if not path_in:
        return "", [], ""

    path = _load_16k(path_in)
    try:
        items = _decode(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    if not items:
        return (
            '<div class="empty">没有识别到音节，请检查音量或换一段更清楚的录音。</div>',
            [],
            "",
        )

    chips = []
    for it in items:
        disp = html.escape(it["syl"] + it["tone"].translate(SUP))
        low = it["score"] < LOW_CONF or it["tone_conf"] < 0.5
        chips.append(
            f'<span class="chip{" low" if low else ""}" '
            f'title="音节置信度 {it["score"]:.2f} / 声调置信度 {it["tone_conf"]:.2f} / '
            f'{it["start"]}–{it["end"]}s">{disp}</span>'
        )
    n_low = sum(1 for it in items if it["score"] < LOW_CONF)
    banner = (
        f'<div class="meta">共 {len(items)} 个音节'
        + (f'，其中 <b>{n_low}</b> 个置信度偏低（标黄，建议人工核对）' if n_low else "，全部为高置信度")
        + "</div>"
    )
    view = f'{banner}<div class="wrap">{"".join(chips)}</div>'

    rows = [[it["syl"], it["tone"], it["score"], it["tone_conf"], it["start"], it["end"]]
            for it in items]
    plain = " ".join(it["syl"] + it["tone"] for it in items)
    return view, rows, plain


CSS = """
.wrap { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; line-height:1.6; }
.chip { font-size:clamp(17px, 4.2vw, 26px); padding:5px 9px; border-radius:8px;
        background:#eef3ff; color:#1b3a6b; font-family: "Noto Sans", system-ui, sans-serif; }
.chip.low { background:#fff3d6; color:#8a5a00; border:1px solid #f0c36d; }
.meta { font-size:14px; color:#555; }
.empty { color:#8a5a00; }

/* ---------- 窄屏适配 ----------
   注意：不要用 @media。Gradio 注入自定义 CSS 时会改写选择器，媒体查询块
   会被破坏而完全不生效（实测 4.44：同一个样式写在 @media 里不生效、写在外面生效）。
   所以这里全部用不依赖断点的写法：可换行的 flex + clamp() + min()。 */

/* 任何组件都不允许把外层容器顶得比屏幕还宽；toned chip 与表格列是常见元凶 */
.gradio-container, .gradio-container > * { width:100% !important; max-width:100% !important; }
.gradio-container * { min-width: 0 !important; }
.gradio-container .prose, .gradio-container .md, .gradio-container p,
.gradio-container label, .gradio-container .label-wrap,
.gradio-container .label-wrap span, .gradio-container button {
  white-space: normal !important;
  overflow-wrap: anywhere !important;
}

/* 两栏在窄屏自动堆叠：每列最小 320px，放不下就换行 */
#main-row { flex-wrap: wrap !important; gap: 12px !important; }
#main-row > div { flex: 1 1 320px !important; min-width: 0 !important; max-width: 100% !important; }

/* Gradio 的 .wrap 是 align-items:flex-start 的纵向容器，子块按 max-content 撑开；
   在窄屏上就是整页比屏幕宽、右侧被切。强制按父宽拉伸。 */
.gradio-container .contain,
.gradio-container .gap,
.gradio-container .block { width: 100% !important; max-width: 100% !important; }
.gradio-container .contain,
.gradio-container .gap { align-items: stretch !important; }

/* 6 列明细表在窄屏横向滚动，而不是把列挤成一条线 */
.table-wrap, .table-container { overflow-x: auto !important; }

/* 上传区在手机上占掉整屏，按视口高度收一收 */
.upload-container, .file-preview { min-height: min(140px, 24vh) !important; }
"""

INTRO = """
### 土家语语音转写台

输入土家语语音，输出**带调音节序列**（如 `la⁵⁵`，声调承载语义）。每个音节都带置信度，**低置信度标黄**，方便人工核对。
"""

INTRO_MORE = """
土家语使用人口不足十万，长期没有任何可用的语音识别技术：没有语料库、没有发音词典、没有声学模型。
这是一个音节级 CTC 模型，**把「人机协同校对」做成了产品的一部分** —— 低资源语言做不了全自动正确，所以让用户一眼看到模型在哪里没把握。

- 建议录 **3–10 秒**的土家语语音，安静环境，麦克风靠近一些
- 免费 CPU 推理，一段 10 秒音频约 5–10 秒出结果
- 模型开源：`github.com/ibaibing/tujia-asr`
"""

# Gradio 6 moved `css` from the Blocks constructor to launch(); pass it to
# whichever one this version accepts so the layout rules are not silently lost.
_GR_MAJOR = int(gr.__version__.split(".")[0])
_BLOCKS_KW = {} if _GR_MAJOR >= 6 else {"css": CSS}
_LAUNCH_KW = {"css": CSS} if _GR_MAJOR >= 6 else {}

with gr.Blocks(title="土家语语音转写台", **_BLOCKS_KW) as demo:
    gr.Markdown(INTRO)
    with gr.Accordion("使用说明与模型信息", open=False):
        gr.Markdown(INTRO_MORE)
    with gr.Row(elem_id="main-row"):
        with gr.Column(scale=1):
            audio = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="录音（不超过 15 秒）",
            )
            upload = gr.File(
                file_types=["audio"],
                type="filepath",
                label="或上传音频文件",
            )
            gr.Markdown(
                "<sub>支持 wav / mp3 / m4a / flac，超过 15 秒只转写前 15 秒。</sub>"
            )
            btn = gr.Button("开始转写", variant="primary")
            gr.Markdown(
                "> 录音与上传填一个即可。第一次点击会下载模型（约 1.2 GB），需要等 1–2 分钟。"
            )
            gr.Markdown("**没有音频？直接试预置示例：**")
            ex_field = gr.Button("示例 · 真实访谈录音")
            gr.Markdown(
                "<sub>示例音频来自公开短视频，已获博主授权，仅用于本项目技术演示。</sub>"
            )
        with gr.Column(scale=1):
            view = gr.HTML(label="转写结果")
            table = gr.Dataframe(
                headers=["音节", "声调", "音节置信度", "声调置信度", "开始(s)", "结束(s)"],
                datatype=["str", "str", "number", "number", "number", "number"],
                label="逐音节明细",
            )
            text = gr.Textbox(label="纯文本（可复制）", lines=3)

    btn.click(transcribe, inputs=[audio, upload], outputs=[view, table, text],
              api_name="transcribe")

    EX = HERE / "examples"
    ex_field.click(
        lambda: transcribe(None, str(EX / "sample_field.wav")),
        outputs=[view, table, text],
        api_name=False,
    )


if __name__ == "__main__":
    demo.launch(**_LAUNCH_KW)
