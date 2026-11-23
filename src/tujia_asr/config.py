"""Runtime settings for the recogniser.

Precedence: defaults < optional JSON file < environment variables.

The JSON file is read from ``TUJIA_ASR_CONFIG`` (or the ``config_path``
argument) and may contain::

    {
      "ckpt": "~/models/tujia-ctc-v0.2.pt",
      "encoder": "~/models/tujia-xlsr53-ft",
      "device": "auto",
      "frontend": "ctc",
      "dicts": ["~/dicts/lexicon.jsonl"],
      "tone_bank": "",
      "zipa": {"root": "", "python": ""},
      "asr": {"mode": "none", "endpoint": "", "model": "", "chat_model": "", "api_key": ""}
    }

A lexicon is optional: without one the recogniser falls back to syllable-level
output.  The open core vocabulary shipped with ``tujia-nlp`` is loaded by
default; ``--dict`` / ``TUJIA_DICTS`` rows are appended after it, and
``TUJIA_NO_DEFAULT_DICT=1`` turns the built-in one off.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FRONTEND = "ctc"
SUPPORTED_MODEL = "tujia-ctc-v0.4"


@dataclass
class Settings:
    ckpt: str = ""
    encoder: str = ""
    device: str = "auto"
    dict_paths: tuple = ()
    default_dict: bool = True
    frontend: str = DEFAULT_FRONTEND
    tone_bank: str = ""
    audio_root: str = ""
    audio_py: str = ""
    blank_bias: float = 0.0
    asr_mode: str = "none"
    asr_endpoint: str = ""
    asr_model: str = ""
    asr_chat_model: str = ""
    asr_api_key: str = ""

    @classmethod
    def from_env(cls, env=None, config_path=None):
        env = os.environ if env is None else env
        cfg = load_config(config_path or env.get("TUJIA_ASR_CONFIG"))
        zipa = cfg.get("zipa") or {}
        asr = cfg.get("asr") or {}

        def pick(env_key, cfg_value, default=""):
            value = env.get(env_key)
            if value is None:
                value = cfg_value
            return default if value is None else value

        listed = [p for p in env.get("TUJIA_DICTS", "").split(os.pathsep) if p.strip()]
        if not listed:
            listed = [str(p) for p in (cfg.get("dicts") or []) if str(p).strip()]

        no_default = str(env.get("TUJIA_NO_DEFAULT_DICT", "")).strip().lower()
        if not no_default:
            no_default = bool(cfg.get("no_default_dict"))

        try:
            blank_bias = float(pick("TUJIA_BLANK_BIAS", cfg.get("blank_bias"), 0) or 0)
        except (TypeError, ValueError):
            blank_bias = 0.0

        return cls(
            ckpt=str(pick("TUJIA_CTC_CKPT", cfg.get("ckpt"))),
            encoder=str(pick("TUJIA_XLSR", cfg.get("encoder"))),
            device=str(pick("TUJIA_CTC_DEVICE", cfg.get("device"), "auto")),
            dict_paths=tuple(str(Path(p).expanduser()) for p in listed),
            default_dict=not (no_default in (True, "1", "true", "yes", "on")),
            frontend=str(pick("TUJIA_FRONTEND", cfg.get("frontend"), DEFAULT_FRONTEND)).strip().lower(),
            tone_bank=str(pick("TUJIA_TONE_BANK", cfg.get("tone_bank"))),
            audio_root=str(pick("TUJIA_AUDIO_ROOT", zipa.get("root"))),
            audio_py=str(pick("TUJIA_AUDIO_PY", zipa.get("python"))),
            blank_bias=blank_bias,
            asr_mode=str(pick("ASR_MODE", asr.get("mode"), "none")).strip().lower(),
            asr_endpoint=str(pick("ASR_ENDPOINT", asr.get("endpoint"))).rstrip("/"),
            asr_model=str(pick("ASR_MODEL", asr.get("model"))),
            asr_chat_model=str(pick("ASR_CHAT_MODEL", asr.get("chat_model"))),
            asr_api_key=str(pick("ASR_API_KEY", asr.get("api_key"))),
        )

    def auth_headers(self):
        return {"Authorization": f"Bearer {self.asr_api_key}"} if self.asr_api_key else {}

    def to_dict(self):
        return {
            "ckpt": self.ckpt,
            "encoder": self.encoder,
            "device": self.device,
            "frontend": self.frontend,
            "dicts": [str(p) for p in self.dict_paths],
            "default_dict": self.default_dict,
            "tone_bank": self.tone_bank,
            "blank_bias": self.blank_bias,
            "zipa": {"root": self.audio_root, "python": self.audio_py},
            "asr": {
                "mode": self.asr_mode,
                "endpoint": self.asr_endpoint,
                "model": self.asr_model,
                "chat_model": self.asr_chat_model,
                "api_key": self.asr_api_key,
            },
        }


def load_config(path=None):
    """Read the optional JSON config; a missing explicit path is an error."""
    if not path:
        return {}
    target = Path(str(path)).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"配置文件不存在：{target}")
    with open(target, encoding="utf-8") as f:
        return json.load(f)


def duration_of(path):
    """Duration in seconds: wave header first, ffprobe as a fallback."""
    import wave

    try:
        with wave.open(str(path), "rb") as f:
            return f.getnframes() / float(f.getframerate())
    except Exception:
        pass
    import subprocess

    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return float(out) if out else 0.0
