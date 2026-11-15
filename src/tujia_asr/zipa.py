"""Optional ZIPA-NS backend: external CTC lattice through its own CLI.

Only used when ``frontend="zipa"``.  It runs the user's local ZIPA checkout in a
subprocess and never imports its model code.  Users without ZIPA should stay on
the default ``ctc`` backend.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from .config import duration_of


class ZipaError(RuntimeError):
    pass


def _require(settings):
    if not settings.audio_root or not settings.audio_py:
        raise ZipaError(
            "未配置 ZIPA：请设置 --zipa-root / TUJIA_AUDIO_ROOT 与 --zipa-python / "
            "TUJIA_AUDIO_PY，或改用默认 --frontend ctc"
        )


def _run(cmd, cwd, timeout=900):
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        raise ZipaError((proc.stderr or proc.stdout or "").strip()[-1200:])
    return proc


def _scale_and_merge(windows, dur, dur_tol=0.15):
    """Scale each window's time axis to its real length, then concatenate."""
    merged = []
    for lat, offset, win_dur in windows:
        scale = win_dur / lat["dur"] if lat.get("dur") else 1.0
        for sp in lat["spikes"]:
            sp = dict(sp)
            sp["start"] = round(offset + sp["start"] * scale, 3)
            sp["end"] = round(offset + sp["end"] * scale, 3)
            merged.append(sp)
    merged.sort(key=lambda s: (s["start"], s["end"]))
    kept = []
    pos_by_phone = {}
    for sp in merged:
        k = pos_by_phone.get(sp["phone"])
        if k is not None and sp["start"] - kept[k]["start"] <= dur_tol:
            if sp["end"] - sp["start"] > kept[k]["end"] - kept[k]["start"]:
                kept[k] = sp
            continue
        pos_by_phone[sp["phone"]] = len(kept)
        kept.append(sp)
    for i, sp in enumerate(kept):
        sp["i"] = i
    for sp in kept:
        sp["end"] = min(sp["end"], dur)
        sp["start"] = min(sp["start"], dur)
    return kept


def _silence_runs(wav, sr=16000, frame=0.03, hop=0.01, drop_db=45.0):
    """Consecutive frames below "peak - drop_db" -> [(t0, t1), ...] in seconds."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(wav), sr=sr, mono=True)
    rms = librosa.feature.rms(
        y=y, frame_length=int(frame * sr), hop_length=int(hop * sr)
    )[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    if db.size == 0:
        return []
    thr = float(np.percentile(db, 95) - drop_db)
    quiet = db < thr
    runs, i, n = [], 0, len(quiet)
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            runs.append((i * hop, j * hop))
            i = j
        else:
            i += 1
    return runs


def plan_windows(wav, target=10.0, search=2.0, min_sil=0.12, max_dur=14.0):
    """Split at silence near each target position so syllables are not cut."""
    dur = duration_of(str(wav))
    if dur <= target + search:
        return [(0.0, None)]
    runs = _silence_runs(wav)
    cuts, k = [], 1
    while target * k < dur - 1.0:
        t = target * k
        lo, hi = t - search, t + search
        cand = [
            (max(r0, lo), min(r1, hi))
            for r0, r1 in runs
            if r1 > lo and r0 < hi and min(r1, hi) - max(r0, lo) >= min_sil
        ]
        c = round((sum(max(cand, key=lambda r: r[1] - r[0])) / 2.0), 3) if cand else round(t, 3)
        cuts.append((t, c))
        k += 1
    plan, start = [], 0.0
    for t, c in cuts:
        if c - start < 1.0 or dur - c < 1.0:
            continue
        if c - start > max_dur:
            c = t if t - start <= max_dur else start + max_dur
        plan.append((start, round(c - start, 3)))
        start = c
    plan.append((start, round(dur - start, 3)))
    return plan


def denoise(wav, dst):
    """Light spectral-gate denoising (soft Wiener mask), timing unchanged."""
    import librosa
    import numpy as np
    import soundfile as sf

    y, sr = librosa.load(str(wav), sr=16000, mono=True)
    n_fft, hop = 2048, 512
    spec = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(spec), np.angle(spec)
    noise = np.percentile(mag, 10, axis=1, keepdims=True)
    mask = mag / (mag + 3.0 * noise + 1e-9)
    clean = librosa.istft(mag * mask * np.exp(1j * phase), hop_length=hop, length=len(y))
    sf.write(str(dst), clean, sr, subtype="PCM_16")
    return str(dst)


def lattice(wav, seg=10.0, search=2.0, topk=6, settings=None, denoise_audio=False):
    """wav -> {spikes, n_frames, dur, n_windows, elapsed_s, model}."""
    from .config import Settings

    settings = settings or Settings.from_env()
    _require(settings)
    wav = str(Path(wav).expanduser().resolve())
    dur = duration_of(wav)
    if dur <= 0:
        raise ZipaError(f"无法读取音频时长: {wav}")

    plan = [(0.0, None)] if dur <= seg + search else plan_windows(wav, target=seg, search=search)

    t0 = time.time()
    windows = []
    with tempfile.TemporaryDirectory() as td:
        src = denoise(wav, os.path.join(td, "clean.wav")) if denoise_audio else wav
        for k, (offset, wdur) in enumerate(plan):
            out = os.path.join(td, f"lat_{k}.json")
            cmd = [
                str(settings.audio_py), "zipa_lattice.py", src,
                "--topk", str(topk), "--offset", f"{offset:.3f}", "--json", out,
            ]
            if wdur is not None:
                cmd += ["--duration", f"{wdur:.3f}"]
            _run(cmd, settings.audio_root)
            with open(out, encoding="utf-8") as f:
                lat = json.load(f)
            windows.append((lat, offset, wdur if wdur is not None else dur))

    spikes = _scale_and_merge(windows, dur)
    return {
        "spikes": spikes,
        "n_frames": int(round(dur / 0.01)),
        "dur": round(dur, 3),
        "n_windows": len(plan),
        "elapsed_s": round(time.time() - t0, 2),
        "model": os.environ.get("TUJIA_AUDIO_MODEL", "zipa-small-crctc-ns-700k"),
    }


def _dict_arg(settings, td):
    """The CLI tools accept one dictionary; merge several into a temp file."""
    paths = [Path(p).expanduser() for p in (settings.dict_paths or ())]
    if not paths:
        raise ZipaError("ZIPA 词解码需要词典：--dict / TUJIA_DICTS")
    if len(paths) == 1:
        return str(paths[0])
    merged = Path(td) / "dict_merged.jsonl"
    with open(merged, "w", encoding="utf-8") as out:
        for p in paths:
            with open(p, encoding="utf-8") as f:
                out.writelines(f)
    return str(merged)


def lexdec(lat, top=200, min_avg=1.5, settings=None):
    """lattice -> word hits + best chain (zipa_lexdec LLR)."""
    from .config import Settings

    settings = settings or Settings.from_env()
    _require(settings)
    with tempfile.TemporaryDirectory() as td:
        lat_path = os.path.join(td, "lat.json")
        out_path = os.path.join(td, "lex.json")
        with open(lat_path, "w", encoding="utf-8") as f:
            json.dump({"spikes": lat["spikes"], "n_frames": lat["n_frames"], "dur": lat["dur"]}, f)
        _run(
            [
                str(settings.audio_py), "zipa_lexdec.py", "--lattice", lat_path,
                "--dict", _dict_arg(settings, td),
                "--top", str(top), "--min-avg", str(min_avg), "--json", out_path,
            ],
            settings.audio_root,
        )
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)


def viterbi_fallback(spikes, top=3, settings=None):
    """1-best phone string through global word splitting (fallback only)."""
    from .config import Settings

    settings = settings or Settings.from_env()
    _require(settings)
    text = " ".join(s["phone"] for s in spikes if not s.get("is_boundary"))
    if not text:
        return {"segments": [], "total_cost": 0.0}
    with tempfile.TemporaryDirectory() as td:
        proc = _run(
            [
                str(settings.audio_py), "zipa_viterbi.py", text,
                "--dict", _dict_arg(settings, td), "--top", str(top), "--json",
            ],
            settings.audio_root,
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"segments": [], "total_cost": 0.0}
