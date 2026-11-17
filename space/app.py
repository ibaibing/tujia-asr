#!/usr/bin/env python3
"""土家语语音转写台 —— Gradio 应用（Hugging Face Spaces / ModelScope 创空间通用）。

设计要点
--------
低资源语言不可能做到全自动正确，所以界面把「不确定」显式化：
每个音节都带置信度，低置信度标黄，用户可以直接看到模型在哪里没把握。

模型不在浏览器里跑：启动时把编码器与 CTC 头拉到容器本地，CPU 常驻推理。

两种入口，同一套推理
--------------------
1. 网页 UI：上传/录音，返回渲染好的音节、明细表与纯文本。
2. MCP / HTTP API：``transcribe`` 只收字符串，返回 JSON。手机客户端把音频编码成
   ``data:audio/...;base64,...`` 传进来，不依赖容器去下载 URL——创空间的
   ``ms.show`` 文件域名对非浏览器 UA 返回 403，容器自己拉不到。
"""
from __future__ import annotations

import base64
import html
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

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
DEFAULT_BLANK_BIAS = 1.0
os.environ.setdefault("TUJIA_BLANK_BIAS", str(DEFAULT_BLANK_BIAS))

SCHEMA_VERSION = 1
WINDOW_SEG = 10.0       # 分窗解码的窗长（秒）
WINDOW_SEARCH = 2.0     # 相邻窗重叠（秒）
MAX_SEC = 300.0         # 单次请求上限：超过则只处理前 5 分钟
MAX_B64_BYTES = 8 << 20  # data URL 解码后的上限：平台网关对请求体有限制
MAX_URL_BYTES = 24 << 20  # 服务端自己下载 URL 时的上限
LOW_CONF = 0.55         # 低于此分数的音节标黄

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
MIME_EXT = {
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a", "audio/aac": ".aac", "audio/flac": ".flac",
    "audio/ogg": ".ogg", "audio/webm": ".webm", "audio/amr": ".amr",
}

SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
TONE_RE = re.compile(r"(\d+)$")
_front = None
_lock = threading.Lock()


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


# ── 音频入口 ────────────────────────────────────────────────────────────────

def _temp_with(suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def _mime_ext(header: str) -> str:
    """``data:audio/wav;base64`` → ``.wav``；未知类型按 wav 处理。"""
    mime = header.split(";")[0].split(":")[-1].strip().lower()
    return MIME_EXT.get(mime, ".wav")


def _fetch_audio(src: str) -> tuple[str, bool]:
    """data URL / http(s) URL / 本地路径 → (本地文件, 是否临时文件)。

    MCP 客户端应当传 data URL：容器没有公网出口，``ms.show`` 的文件域名还会
    对非浏览器 UA 返回 403，让服务端去下载 URL 十有八九失败。
    """
    src = (src or "").strip()
    if not src:
        raise ValueError("没有收到音频")

    if src.startswith("data:"):
        header, _, payload = src.partition(",")
        try:
            data = base64.b64decode(payload, validate=False)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"base64 解码失败：{exc}") from exc
        if not data:
            raise ValueError("base64 内容为空")
        if len(data) > MAX_B64_BYTES:
            raise ValueError(
                f"data URL 超过 {MAX_B64_BYTES >> 20} MB（约 3 分钟 16 kHz wav）；"
                "请切片，或压成 m4a/opus 再传")
        path = _temp_with(_mime_ext(header))
        Path(path).write_bytes(data)
        return path, True

    if src.startswith(("http://", "https://")):
        import requests

        r = requests.get(src, timeout=60, headers={"User-Agent": BROWSER_UA})
        r.raise_for_status()
        if len(r.content) > MAX_URL_BYTES:
            raise ValueError(f"音频超过 {MAX_URL_BYTES >> 20} MB，请先压缩或切片")
        suffix = Path(src.split("?")[0]).suffix or ".wav"
        path = _temp_with(suffix)
        Path(path).write_bytes(r.content)
        return path, True

    p = Path(src).expanduser()
    if not p.is_file():
        raise ValueError(f"找不到音频文件：{src}")
    return str(p), False


def _to_wav16k(path: str, max_sec: float) -> str:
    """统一成 16 kHz 单声道；超过上限只取前 max_sec 秒。"""
    import librosa
    import soundfile as sf

    dur = None if not max_sec or max_sec <= 0 else float(max_sec)
    try:
        y, _ = librosa.load(path, sr=16000, mono=True, duration=dur)
    except Exception as exc:  # noqa: BLE001
        detail = str(exc) or type(exc).__name__
        raise ValueError(
            f"无法解码音频（{Path(path).suffix or '未知格式'}）：{detail}。"
            "请确认是 wav / mp3 / m4a / flac。"
        ) from exc
    out = _temp_with(".wav")
    sf.write(out, y, 16000)
    return out


# ── 推理 ────────────────────────────────────────────────────────────────────

def _syllables(spikes: list[dict]) -> list[dict]:
    """把 phone 级条目合并回音节（分窗解码返回的是逐音素）。"""
    items: list[dict] = []
    for s in spikes:
        if s.get("is_boundary"):
            if items and not items[-1].get("boundary"):
                items.append({"boundary": True, "start": s["start"], "end": s["end"]})
            continue
        syl = s.get("syl") or ""
        if not syl:
            continue
        tone = s.get("tone") or ""
        if (items and not items[-1].get("boundary")
                and items[-1]["syl"] == syl
                and abs(items[-1]["end"] - s["start"]) < 0.02):
            items[-1]["end"] = s["end"]
            items[-1]["score"] = min(items[-1]["score"], s.get("score", 0.0))
            continue
        items.append({
            "syl": syl,
            "tone": tone,
            "score": float(s.get("score") or 0.0),
            "tone_conf": float(s.get("tone_conf") or 0.0),
            "start": s["start"],
            "end": s["end"],
            "boundary": False,
        })
    return items


def _run(path: str, blank_bias: float = DEFAULT_BLANK_BIAS,
         tone_correct: bool = True, low_conf: float = LOW_CONF) -> dict:
    from tujia_asr.ctc import lattice

    front = get_front()
    with _lock:  # CtcFront 是单例，参数改在实例上，串行化避免并发用户互相覆盖
        front.blank_bias = float(blank_bias)
        front.tone_force = bool(tone_correct)
        # lattice() 内部会再调一次 CtcFront.get()，必须把路径传进去，
        # 否则它会认为 ckpt 未指定而抛 FileNotFoundError。
        res = lattice(path, seg=WINDOW_SEG, search=WINDOW_SEARCH,
                      ckpt=front.ckpt_path, encoder=front.encoder_arg,
                      device=str(front.device),
                      settings=SimpleNamespace(blank_bias=float(blank_bias)))

    items = _syllables(res["spikes"])
    syls = [it for it in items if not it.get("boundary")]
    enc = str(getattr(front, "xlsr_path", "") or ENC_REPO)
    meta = {
        "model": res.get("model") or Path(front.ckpt_path).name,
        "encoder": Path(enc).name if os.path.sep in enc else enc,
        "duration": round(float(res.get("dur") or 0.0), 2),
        "n_windows": int(res.get("n_windows") or 1),
        "elapsed_s": float(res.get("elapsed_s") or 0.0),
        "blank_bias": round(float(blank_bias), 2),
        "tone_correct": bool(tone_correct),
        "low_conf": round(float(low_conf), 2),
        "n_syllables": len(syls),
        "n_low": sum(1 for it in syls
                     if it["score"] < low_conf or it["tone_conf"] < 0.5),
    }
    return {"items": items, "syllables": syls, "meta": meta}


def _tones(it: dict) -> str:
    return it["syl"] + (it["tone"] or "")


def _toned(it: dict) -> str:
    return it["syl"] + (it["tone"] or "").translate(SUP)


# ── UI 入口 ─────────────────────────────────────────────────────────────────

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


def _ui_transcribe(audio_path, file_path, blank_bias, tone_correct, low_conf):
    path_in = _pick(audio_path, file_path)
    if not path_in:
        return "", [], "", ""

    try:
        path = _to_wav16k(path_in, MAX_SEC)
    except Exception as exc:  # noqa: BLE001
        return (f'<div class="empty">{html.escape(str(exc))}</div>', [], "", "")
    try:
        r = _run(path, blank_bias, tone_correct, low_conf)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    m, syls = r["meta"], r["syllables"]
    params = (f"模型 {m['model']} · 编码器 {m['encoder'].split('/')[-1]} · "
              f"blank_bias {m['blank_bias']:.1f} · 声调校正 {'开' if m['tone_correct'] else '关'} · "
              f"{m['n_windows']} 个窗口 · {m['elapsed_s']:.1f}s")
    if not syls:
        return (
            '<div class="empty">没有识别到音节，请检查音量或换一段更清楚的录音。</div>',
            [], "", params,
        )

    chips: list[str] = []
    for it in r["items"]:
        if it.get("boundary"):
            chips.append('<span class="sep"></span>')
            continue
        low = it["score"] < low_conf or it["tone_conf"] < 0.5
        chips.append(
            f'<span class="chip{" low" if low else ""}" '
            f'title="音节置信度 {it["score"]:.2f} / 声调置信度 {it["tone_conf"]:.2f} / '
            f'{it["start"]}–{it["end"]}s">{html.escape(_toned(it))}</span>'
        )
    n_low = sum(1 for it in syls if it["score"] < low_conf)
    banner = (
        f'<div class="meta">共 {len(syls)} 个音节，分 {m["n_windows"]} 窗解码，'
        f'音频 {m["duration"]:.1f}s'
        + (f'；其中 <b>{n_low}</b> 个置信度偏低（标黄，建议人工核对）' if n_low else "；全部为高置信度")
        + "</div>"
    )
    view = f'{banner}<div class="wrap">{"".join(chips)}</div>'

    rows = [[it["syl"], it["tone"], it["score"], it["tone_conf"], it["start"], it["end"]]
            for it in syls]
    plain = " ".join(_tones(it) for it in syls)
    return view, rows, plain, params


# ── MCP / HTTP API ──────────────────────────────────────────────────────────

def transcribe(audio: str, blank_bias: float = DEFAULT_BLANK_BIAS,
               tone_correct: bool = True, max_seconds: float = MAX_SEC) -> str:
    """转写一段音频，返回 JSON。

    ``audio`` 接受三种形式：``data:audio/...;base64,...``（手机客户端推荐）、
    ``http(s)`` URL、容器内本地路径。``blank_bias`` 调高会减少漏字、增加错字。
    返回字段见 ``SCHEMA_VERSION``。
    """
    keep: list[str] = []
    try:
        src, owned = _fetch_audio(audio)
        if owned:
            keep.append(src)
        path = _to_wav16k(src, max_seconds)
        keep.append(path)
        r = _run(path, blank_bias, tone_correct)
        payload = {
            "schema": SCHEMA_VERSION,
            "ok": True,
            "meta": r["meta"],
            "text": " ".join(_tones(it) for it in r["syllables"]),
            "text_sup": " ".join(_toned(it) for it in r["syllables"]),
            "syllables": [
                {k: it[k] for k in ("syl", "tone", "score", "tone_conf", "start", "end")}
                for it in r["syllables"]
            ],
            "segments": _segments(r["items"]),
        }
    except Exception as exc:  # noqa: BLE001
        payload = {"schema": SCHEMA_VERSION, "ok": False,
                   "error": f"{type(exc).__name__}: {exc}", "syllables": []}
    finally:
        for p in keep:
            try:
                os.unlink(p)
            except OSError:
                pass
    return json.dumps(payload, ensure_ascii=False)


def _segments(items: list[dict]) -> list[dict]:
    """按 <sep> 把音节切成语段，客户端可以直接逐句展示。"""
    out, cur = [], []
    for it in items:
        if it.get("boundary"):
            if cur:
                out.append(cur)
                cur = []
            continue
        cur.append(it)
    if cur:
        out.append(cur)
    return [
        {
            "start": g[0]["start"],
            "end": g[-1]["end"],
            "text": " ".join(_tones(it) for it in g),
            "text_sup": " ".join(_toned(it) for it in g),
        }
        for g in out
    ]


# ── 界面 ────────────────────────────────────────────────────────────────────

CSS = """
.wrap { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; line-height:1.6; }
.chip { font-size:clamp(17px, 4.2vw, 26px); padding:5px 9px; border-radius:8px;
        background:#eef3ff; color:#1b3a6b; font-family: "Noto Sans", system-ui, sans-serif; }
.chip.low { background:#fff3d6; color:#8a5a00; border:1px solid #f0c36d; }
.sep { flex-basis:100%; height:0; }
.meta { font-size:14px; color:#555; }
.run { font-size:12px; color:#6b6b6b; font-family: ui-monospace, monospace;
       overflow-wrap:anywhere; }
.empty { color:#8a5a00; }

/* ---------- 窄屏适配 ----------
   注意：不要用 @media。Gradio 注入自定义 CSS 时会改写选择器，媒体查询块
   会被破坏而完全不生效（实测 4.44：同一个样式写在 @media 里不生效、写在外面生效）。
   所以这里全部用不依赖断点的写法：可换行的 flex + clamp() + min()。 */

.gradio-container, .gradio-container > * { width:100% !important; max-width:100% !important; }
.gradio-container * { min-width: 0 !important; }
.gradio-container .prose, .gradio-container .md, .gradio-container p,
.gradio-container label, .gradio-container .label-wrap,
.gradio-container .label-wrap span, .gradio-container button {
  white-space: normal !important;
  overflow-wrap: anywhere !important;
}

#main-row { flex-wrap: wrap !important; gap: 12px !important; }
#main-row > div { flex: 1 1 320px !important; min-width: 0 !important; max-width: 100% !important; }

.gradio-container .contain,
.gradio-container .gap,
.gradio-container .block { width: 100% !important; max-width: 100% !important; }
.gradio-container .contain,
.gradio-container .gap { align-items: stretch !important; }

.table-wrap, .table-container { overflow-x: auto !important; }
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
- 长录音会按 10 秒分窗解码（相邻窗重叠 2 秒），单次上限 5 分钟
- 免费 CPU 推理，一段 10 秒音频约 5–10 秒出结果
- 模型开源：`github.com/ibaibing/tujia-asr`
"""

_GR_MAJOR = int(gr.__version__.split(".")[0])
_BLOCKS_KW = {} if _GR_MAJOR >= 6 else {"css": CSS}
_LAUNCH_KW = {"css": CSS} if _GR_MAJOR >= 6 else {}
if _GR_MAJOR >= 5:
    # 平台通常会用 GRADIO_MCP_SERVER 打开，但本地 `python app.py` 不会；
    # 显式打开后本地与线上行为一致，手机客户端连本地也能调试。
    _LAUNCH_KW["mcp_server"] = True

with gr.Blocks(title="土家语语音转写台", **_BLOCKS_KW) as demo:
    gr.Markdown(INTRO)
    with gr.Accordion("使用说明与模型信息", open=False):
        gr.Markdown(INTRO_MORE)
    with gr.Row(elem_id="main-row"):
        with gr.Column(scale=1):
            audio = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="录音",
            )
            upload = gr.File(
                file_types=["audio"],
                type="filepath",
                label="或上传音频文件",
            )
            gr.Markdown(
                "<sub>支持 wav / mp3 / m4a / flac；长音频自动分窗，单次上限 5 分钟。</sub>"
            )
            with gr.Accordion("高级参数（校对时再动）", open=False):
                adv_blank = gr.Slider(
                    0.0, 2.0, value=DEFAULT_BLANK_BIAS, step=0.1,
                    label="输出密度",
                    info="调高：更少漏字、可能多出噪声音节；调低：更少错字、可能整段消失。"
                         "非专业校对建议保持 1.0。",
                )
                adv_tone = gr.Checkbox(
                    value=True, label="声调校正（辅助头）",
                    info="关闭后只用主 CTC 头自带的调号；与本地 infer.py 对比时请关闭。",
                )
                adv_conf = gr.Slider(
                    0.3, 0.9, value=LOW_CONF, step=0.05,
                    label="标黄阈值",
                    info="音节置信度低于此值时标黄，只影响显示。",
                )
            btn = gr.Button("开始转写", variant="primary")
            gr.Markdown(
                "> 录音与上传填一个即可。第一次点击会下载模型（约 1.3 GB），需要等 1–2 分钟。"
            )
            gr.Markdown("**没有音频？直接试预置示例：**")
            ex_field = gr.Button("示例 · 真实访谈录音")
            gr.Markdown(
                "<sub>示例音频来自公开短视频，已获博主授权，仅用于本项目技术演示。</sub>"
            )
        with gr.Column(scale=1):
            view = gr.HTML(label="转写结果")
            run_meta = gr.Markdown("", elem_classes="run")
            table = gr.Dataframe(
                headers=["音节", "声调", "音节置信度", "声调置信度", "开始(s)", "结束(s)"],
                datatype=["str", "str", "number", "number", "number", "number"],
                label="逐音节明细",
            )
            text = gr.Textbox(label="纯文本（可复制）", lines=3)

    btn.click(_ui_transcribe,
              inputs=[audio, upload, adv_blank, adv_tone, adv_conf],
              outputs=[view, table, text, run_meta],
              api_name=False)

    EX = HERE / "examples"
    ex_field.click(
        lambda: _ui_transcribe(None, str(EX / "sample_field.wav"),
                               DEFAULT_BLANK_BIAS, True, LOW_CONF),
        outputs=[view, table, text, run_meta],
        api_name=False,
    )

    # MCP / HTTP API 由平台（Gradio + ModelScope）提供，界面上不展示。
    # 但公开端点必须挂在某个事件上，所以保留一组隐藏组件承载 transcribe。
    api_in = gr.Textbox(visible=False, value="")
    api_max = gr.Number(value=MAX_SEC, visible=False)
    api_out = gr.Code(visible=False)
    api_btn = gr.Button(visible=False)
    api_btn.click(transcribe,
                  inputs=[api_in, adv_blank, adv_tone, api_max],
                  outputs=api_out,
                  api_name="transcribe")

if __name__ == "__main__":
    # 0.0.0.0 让同一 WiFi 的手机能连本地 MCP；平台会用 GRADIO_SERVER_NAME 覆盖。
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
                **_LAUNCH_KW)
