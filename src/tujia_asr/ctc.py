"""CTC acoustic front-end: released syllable model -> phone spikes.

The spikes are the same shape the optional ZIPA backend emits, so everything
downstream is frontend-agnostic: a list of ``{phone, start, end, score, tone,
tone_conf, is_boundary}``.
"""

import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio

SR = 16000
HOP = 160          # 10 ms
DOWNSAMPLE = 4     # two stride-2 convolutions
FRAME_SEC = HOP * DOWNSAMPLE / SR
XLSR_FRAME_SEC = 0.02   # wav2vec2 downsamples by 320x -> 50 Hz
MIN_FRAMES = 8
PEAK_PAD_SEC = 0.30     # a spike is one syllable: keep it within ±0.30 s of its peak
DEFAULT_XLSR = os.environ.get("TUJIA_XLSR", "")


class XlsrHead(nn.Module):
    """XLSR-53 frame features (50 Hz / 1024 dim) -> toneless syllable CTC head."""

    def __init__(self, n_in=1024, hidden=256, n_classes=264, layers=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(n_in)
        self.rnn = nn.LSTM(
            n_in, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, feats, lengths):
        h = self.norm(feats)
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        return self.head(h), lengths


class HeadTone(nn.Module):
    """XLSR frame features -> toned-syllable CTC head (v0.2: main + tone head)."""

    def __init__(self, n_in=1024, hidden=256, n_classes=709, n_tone=5,
                 layers=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(n_in)
        self.rnn = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True,
                           bidirectional=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden * 2, n_classes)
        self.tone_head = nn.Linear(hidden * 2, n_tone)

    def forward(self, feats, lengths):
        h = self.norm(feats)
        packed = nn.utils.rnn.pack_padded_sequence(
            h, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        return self.head(h), self.tone_head(h), lengths


class CTCNet(nn.Module):
    def __init__(self, n_mels=80, hidden=256, n_classes=710, layers=2, dropout=0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.rnn = nn.LSTM(
            hidden, hidden, num_layers=layers, batch_first=True,
            bidirectional=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x, lengths):
        h = self.conv(x.transpose(1, 2))
        out_len = (lengths + 1) // 2
        out_len = (out_len + 1) // 2
        h = h.transpose(1, 2)
        packed = nn.utils.rnn.pack_padded_sequence(
            h, out_len.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.rnn(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        return self.head(h), out_len


class CtcFront:
    """Lazy singleton: weights are reused across calls in one process."""

    _inst = None

    def __init__(self, ckpt, device="auto", encoder=None):
        self.ckpt_path = str(Path(ckpt).expanduser())
        self.encoder_arg = str(encoder or "")
        if device == "auto" and os.environ.get("TUJIA_CTC_DEVICE"):
            device = os.environ["TUJIA_CTC_DEVICE"]
        self.device = torch.device(
            device if device != "auto"
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.vocab = ck["vocab"]
        self.tones = ck.get("tones") or []
        self.arch = ck.get("arch", "conv_bilstm")
        if self.arch == "conv_bilstm" and "state_dict" in ck and "tones" in ck:
            self.arch = "xlsr_head_tone"   # v0.2 checkpoints omit ``arch``
        self.blank = 0
        self.sep = self.vocab.index("<sep>") if "<sep>" in self.vocab else -1
        # v0.3 在弱音与连读处过量输出 blank：held-out 上删除率 41%、输出长度只有
        # 参考的 62%。压一压 blank 的 logit 可以把「丢字」换成可校对的可疑音节，
        # 代价是替换率上升。1.0 是 SER 基本无损的拐点，1.5 更偏「宁错不删」。
        try:
            self.blank_bias = float(os.environ.get("TUJIA_BLANK_BIAS") or 0.0)
        except ValueError:
            self.blank_bias = 0.0
        if self.arch in ("xlsr_head", "xlsr_head_tone"):
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

            path = (encoder or os.environ.get("TUJIA_XLSR_ENC")
                    or DEFAULT_XLSR or ck.get("encoder_ft") or ck.get("encoder"))
            if not str(path or "").strip():
                raise FileNotFoundError(
                    "未找到 XLSR 编码器：请用 --encoder 或 TUJIA_XLSR 指向 "
                    "tujia-xlsr53-v03f 目录（含 config.json）"
                )
            self.xlsr_path = str(Path(path).expanduser())
            if not Path(self.xlsr_path).is_dir():
                raise FileNotFoundError(
                    f"编码器目录不存在：{self.xlsr_path}\n"
                    "下载 TujiaNLP/tujia-xlsr53-v03f（Hugging Face / ModelScope），"
                    "再指向该目录。v0.4 不能用原始 XLSR-53 权重替代。"
                )
            self.feat = Wav2Vec2FeatureExtractor.from_pretrained(self.xlsr_path)
            self.encoder = Wav2Vec2Model.from_pretrained(self.xlsr_path).to(self.device).eval()
            if self.arch == "xlsr_head_tone":
                self.model = HeadTone(**ck["head_args"]).to(self.device)
                self.model.load_state_dict(ck["state_dict"])
            else:
                self.model = XlsrHead(**ck["head_args"]).to(self.device)
                self.model.load_state_dict(ck["head"])
            self.model.eval()
            self.frame_sec = XLSR_FRAME_SEC
            return
        a = ck["args"]
        self.model = CTCNet(
            hidden=a.get("hidden", 256), n_classes=len(self.vocab),
            layers=a.get("layers", 2), dropout=a.get("dropout", 0.15),
        ).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.frame_sec = FRAME_SEC
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=400, hop_length=HOP, n_mels=80, f_min=40, f_max=7600
        ).to(self.device)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    @classmethod
    def get(cls, ckpt=None, device="auto", encoder=None):
        ckpt = ckpt or os.environ.get("TUJIA_CTC_CKPT", "")
        if not str(ckpt).strip():
            raise FileNotFoundError(
                "未找到 CTC 权重：请用 --ckpt 或 TUJIA_CTC_CKPT 指向 "
                "tujia-ctc-v0.4.pt（见 https://github.com/ibaibing/tujia-asr/releases）"
            )
        ckpt = str(Path(ckpt).expanduser())
        if not Path(ckpt).is_file():
            raise FileNotFoundError(f"CTC 权重不存在：{ckpt}")
        if (cls._inst is None or cls._inst.ckpt_path != ckpt
                or cls._inst.encoder_arg != str(encoder or "")):
            cls._inst = cls(ckpt, device, encoder)
        return cls._inst

    def read_wav(self, path, offset=None, duration=None):
        info = sf.info(str(path))
        start = int((offset or 0) * info.samplerate)
        stop = int(((offset or 0) + duration) * info.samplerate) if duration else None
        y, sr = sf.read(str(path), dtype="float32", always_2d=False, start=start, stop=stop)
        if y.ndim > 1:
            y = y.mean(axis=1)
        t = torch.from_numpy(np.ascontiguousarray(y)).to(self.device)
        if sr != SR:
            t = torchaudio.functional.resample(t, sr, SR)
        return t

    def log_mel(self, path, offset=None, duration=None):
        t = self.read_wav(path, offset, duration)
        m = self.db(self.mel(t)).squeeze(0).transpose(0, 1)  # (T, 80)
        m = (m - m.mean(0, keepdim=True)) / (m.std(0, keepdim=True) + 1e-5)
        return m.contiguous()

    def features(self, path, offset=None, duration=None):
        """XLSR front-end: waveform -> (T, 1024) frame features."""
        t = self.read_wav(path, offset, duration)
        inp = self.feat(t.cpu().numpy(), sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            h = self.encoder(inp.input_values.to(self.device)).last_hidden_state
        return h[0].float()

    def encode(self, path, offset=None, duration=None):
        return self.features(path, offset, duration) \
            if self.arch in ("xlsr_head", "xlsr_head_tone") \
            else self.log_mel(path, offset, duration)

    def decode(self, x, blank_bias=None):
        n = torch.tensor([x.shape[0]])
        with torch.no_grad():
            out = self.model(x.unsqueeze(0), n)
        if self.arch == "xlsr_head_tone":
            logits, tone_logits, out_len = out
        else:
            logits, out_len, tone_logits = out[0], out[1], None
        m = int(out_len[0])
        logits = logits[0, :m].float().cpu()
        bias = self.blank_bias if blank_bias is None else float(blank_bias)
        if bias:
            logits[:, self.blank] -= bias
        prob = logits.softmax(-1)
        ids = prob.argmax(-1).tolist()
        best = prob.max(-1).values.tolist()
        if tone_logits is not None:
            tp = tone_logits[0, :m].float().cpu().softmax(-1)
            if getattr(self, "tone_force", True):
                # Skip the "none" class and pick among the four tones.
                sub = tp[:, 1:]
                tone_ids = (sub.argmax(-1) + 1).tolist()
                tone_conf = sub.max(-1).values.tolist()
            else:
                tone_ids = tp.argmax(-1).tolist()
                tone_conf = tp.max(-1).values.tolist()
        else:
            tone_ids, tone_conf = [0] * m, [0.0] * m
        toks, prev = [], -1
        for k, v in enumerate(ids):
            if v != prev and v != self.blank:
                toks.append([v, k, k, best[k], tone_ids[k], tone_conf[k]])
            elif v != self.blank and toks and toks[-1][0] == v:
                toks[-1][2] = k
                toks[-1][3] = max(toks[-1][3], best[k])
                if tone_conf[k] > toks[-1][5]:
                    toks[-1][4], toks[-1][5] = tone_ids[k], tone_conf[k]
            prev = v
        return toks

    def _tone_of(self, tone_id):
        if tone_id and 0 < tone_id <= len(self.tones):
            return self.tones[tone_id - 1]
        return None

    def spikes(self, path, offset=0.0, duration=None, relative=False):
        x = self.encode(path, offset, duration)
        if x.shape[0] < MIN_FRAMES:
            return []
        toks = self.decode(x)
        if not toks:
            return []
        fs = self.frame_sec
        peaks = [t[1] for t in toks]
        out = []
        for i, (tid, f0, f1, score, tone_id, tone_conf) in enumerate(toks):
            label = self.vocab[tid]
            tone = self._tone_of(tone_id)
            tconf = round(float(tone_conf), 3)
            lo = 0.0 if i == 0 else (peaks[i - 1] + peaks[i]) / 2.0 * fs
            hi = (peaks[i] + peaks[i + 1]) / 2.0 * fs if i + 1 < len(toks) else (f1 + 1) * fs
            lo, hi = offset + lo, offset + min(hi, x.shape[0] * fs)
            if hi <= lo:
                hi = lo + fs
            # The midpoint rule above grows without bound when the next spike is
            # seconds away: a lone syllable in a silent stretch used to claim the
            # whole silence (measured 3.9 s on a 0.3 s syllable), which then
            # blocked syllable grouping (MAX_DUR) and inflated every coverage
            # statistic. A syllable cannot be wider than PEAK_PAD_SEC each side
            # of its own peak.
            peak_t = offset + peaks[i] * fs
            lo = max(lo, peak_t - PEAK_PAD_SEC)
            hi = min(hi, peak_t + PEAK_PAD_SEC)
            if relative:  # window-relative; _scale_and_merge shifts it later
                lo, hi = lo - offset, hi - offset
            if label == "<sep>":
                out.append({"phone": "", "start": round(lo, 3), "end": round(hi, 3),
                            "is_boundary": True, "score": round(score, 3), "syl": label,
                            "tone": None, "tone_conf": 0.0})
                continue
            phones = [c for c in label if not c.isdigit()]
            if not phones:
                continue
            lp = round(math.log(max(score, 1e-6)), 3)
            step = (hi - lo) / len(phones)
            for j, ph in enumerate(phones):
                out.append({"phone": ph, "start": round(lo + j * step, 3),
                            "end": round(lo + (j + 1) * step, 3),
                            "is_boundary": False, "score": round(score, 3),
                            "top": [[ph, lp]], "syl": "".join(phones),
                            "tone": tone, "tone_conf": tconf})
        return out


def lattice(wav, seg=10.0, search=2.0, settings=None, ckpt=None, device="auto",
            encoder=None):
    """wav -> {spikes, n_frames, dur, n_windows, elapsed_s, model}."""
    import time

    from .config import duration_of
    from .zipa import _scale_and_merge, plan_windows

    t0 = time.time()
    wav = str(Path(wav).expanduser().resolve())
    dur = duration_of(wav)
    front = CtcFront.get(ckpt, device, encoder)
    front.blank_bias = float(getattr(settings, "blank_bias", 0) or 0)
    plan = [(0.0, None)] if dur <= seg + search else plan_windows(wav, target=seg, search=search)
    windows = []
    for offset, wdur in plan:
        win = front.spikes(wav, offset, wdur, relative=True)
        windows.append(({"dur": wdur if wdur else dur, "spikes": win},
                        offset, wdur if wdur is not None else dur))
    spikes = _scale_and_merge(windows, dur)
    return {
        "spikes": spikes,
        "n_frames": int(round(dur / 0.01)),
        "dur": round(dur, 3),
        "n_windows": len(plan),
        "elapsed_s": round(time.time() - t0, 2),
        "model": Path(front.ckpt_path).name,
    }
