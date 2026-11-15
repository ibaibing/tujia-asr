"""Optional second channel: a weak external transcript used as evidence.

moss    MOSS-Audio-Instruct syllable-by-syllable transcription (weight 0.5)
whisper Whisper near-homophone characters (0.35)
qwen3   Qwen3-ASR Chinese sentence (0.2, fallback only)
none    disabled (0)

Disabled by default: it needs an OpenAI-compatible endpoint.
"""

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
import uuid

WEIGHTS = {"moss": 0.5, "whisper": 0.35, "qwen3": 0.2, "none": 0.0}

MOSS_INSTRUCTION = (
    "逐音节记音：把听到的每个音节用拼音或汉字近音写出来，音节之间用空格分隔，"
    "不要翻译成汉语意思，不要补全句子。"
)


def degenerate(text):
    """Repetition collapse: any 2-6 char fragment >12 times, or length >4000."""
    if not text:
        return False
    for n in (2, 3, 4, 6):
        for i in range(0, max(0, len(text) - n * 13), max(1, n)):
            frag = text[i:i + n]
            if frag and text.count(frag) > 12:
                return True
    return len(text) > 4000


def _post(url, body, headers, timeout=300):
    req = urllib.request.Request(url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _multipart(fields, file_path):
    boundary = "----" + uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        )
    mime = mimetypes.guess_type(file_path)[0] or "audio/wav"
    with open(file_path, "rb") as f:
        data = f.read()
    parts.append(
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{os.path.basename(file_path)}\"\r\nContent-Type: {mime}\r\n\r\n"
        ).encode()
        + data
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def _transcribe(path, settings, dur):
    url = settings.asr_endpoint + "/audio/transcriptions"
    body, headers = _multipart(
        {"model": settings.asr_model, "language": "zh", "response_format": "json"},
        path,
    )
    headers.update(settings.auth_headers())
    data = _post(url, body, headers)
    text = (data.get("text") or "").strip()
    segs = [
        {"text": s.get("text", "").strip(), "t0": s.get("start", 0.0), "t1": s.get("end", dur)}
        for s in (data.get("segments") or [])
    ]
    return text, segs


def _moss(path, settings, dur):
    url = settings.asr_endpoint + "/chat/completions"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": settings.asr_chat_model or settings.asr_model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": MOSS_INSTRUCTION},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": b64, "format": "wav"},
                    },
                ],
            }
        ],
    }
    headers = {"Content-Type": "application/json"}
    headers.update(settings.auth_headers())
    data = _post(url, json.dumps(payload).encode(), headers)
    msg = (data.get("choices") or [{}])[0].get("message", {})
    text = (msg.get("content") or "").strip()
    return text, [{"text": text, "t0": 0.0, "t1": dur}] if text else []


def run(path, dur, mode="none", settings=None):
    """Always returns {status, mode, model, text, segments, weight, note}."""
    from .config import Settings

    settings = settings or Settings.from_env()
    mode = (mode or "none").lower()
    if mode not in WEIGHTS:
        mode = "none"
    out = {
        "status": "ok",
        "mode": mode,
        "model": "",
        "text": "",
        "segments": [],
        "weight": WEIGHTS[mode],
        "note": "no token-level confidence",
    }
    if mode == "none":
        out["status"] = "unavailable"
        return out
    if not settings.asr_endpoint:
        out["status"] = "not-configured"
        out["note"] = "未配置 ASR 端点（TUJIA_ASR_CONFIG: asr.endpoint 或 ASR_ENDPOINT）"
        return out

    try:
        if mode == "moss":
            out["model"] = settings.asr_chat_model or settings.asr_model
            text, segs = _moss(path, settings, dur)
        else:
            out["model"] = settings.asr_model
            text, segs = _transcribe(path, settings, dur)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, KeyError) as e:
        out["status"] = "error"
        out["note"] = f"{type(e).__name__}: {e}"
        return out

    if degenerate(text):
        out["status"] = "error"
        out["note"] = "degenerate output"
        return out
    out["text"] = text
    out["segments"] = segs or ([{"text": text, "t0": 0.0, "t1": dur}] if text else [])
    return out
