"""E2b/E2c: F0 tracking, syllable features, four-tone prototypes (+ optional TTS DTW)."""

import numpy as np

FMIN, FMAX = 60.0, 500.0
FRAME_LENGTH = 1024
HOP = 64
N_CONTOUR = 10

# Semitone contours relative to the speaker's median F0.
PROTOTYPES = {
    55: np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
    35: np.array([-0.5, 0.0, 0.5, 1.5, 3.0]),
    21: np.array([-2.5, -3.0, -3.5, -4.0, -4.5]),
    53: np.array([4.5, 3.5, 1.5, -0.5, -2.5]),
}

# Southwestern Mandarin four tones (55/21/42/213), kept for the second channel.
PROTOTYPES_ZH = {
    55: np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
    21: np.array([-2.0, -2.5, -3.0, -3.5, -4.0]),
    42: np.array([4.5, 3.0, 1.0, -0.5, -2.0]),
    213: np.array([-0.5, -3.0, -4.0, -2.0, 1.0]),
}


def _hz_to_st(f0, ref):
    return 12.0 * np.log2(np.maximum(f0, 1e-6) / max(ref, 1e-6))


def _median_filter(x, k=5):
    if len(x) < k:
        return x
    pad = k // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.array([np.median(padded[i:i + k]) for i in range(len(x))])


def _repair_octaves(f0):
    """Pull octave / half-octave frames back to the local median."""
    out = f0.copy()
    voiced = np.isfinite(out)
    if voiced.sum() < 5:
        return out
    ref = np.nanmedian(out)
    if not np.isfinite(ref) or ref <= 0:
        return out
    for _ in range(2):
        v = np.isfinite(out)
        if v.sum() < 5:
            break
        med = np.nanmedian(out[v])
        high = v & (out > med * 1.75) & (out / 2.0 >= FMIN)
        low = v & (out < med * 0.55) & (out * 2.0 <= FMAX)
        out[high] = out[high] / 2.0
        out[low] = out[low] * 2.0
    return out


def f0_track(y, sr=16000):
    """Return {f0_hz, voiced, hop_s, median_hz, voiced_ratio, times}; None without librosa."""
    try:
        import librosa
    except ImportError:
        return None
    f0, voiced_flag, _ = librosa.pyin(
        np.asarray(y, dtype=np.float32),
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP,
    )
    f0 = np.asarray(f0, dtype=float)
    voiced = np.isfinite(f0) & np.asarray(voiced_flag, dtype=bool)
    f0[~voiced] = np.nan
    f0 = _repair_octaves(f0)
    f0 = _median_filter(f0, 5)
    f0[~voiced] = np.nan
    hop_s = HOP / float(sr)
    times = np.arange(len(f0)) * hop_s
    med = float(np.nanmedian(f0)) if voiced.any() else 0.0
    return {
        "f0_hz": f0,
        "voiced": voiced,
        "hop_s": hop_s,
        "times": times,
        "median_hz": med,
        "voiced_ratio": float(voiced.mean()) if len(voiced) else 0.0,
    }


def _interp_segment(f0, times, t0, t1):
    mask = (times >= t0) & (times <= t1)
    seg = f0[mask]
    if seg.size == 0:
        return np.array([])
    idx = np.arange(len(seg))
    good = np.isfinite(seg)
    if good.sum() < 2:
        return np.array([])
    seg = np.interp(idx, idx[good], seg[good])
    return seg


def contour(f0, times, ref_hz, t0, t1, n=N_CONTOUR):
    seg = _interp_segment(f0, times, t0, t1)
    if seg.size < 2:
        return None
    st = _hz_to_st(seg, ref_hz)
    xp = np.linspace(0, 1, len(st))
    x = np.linspace(0, 1, n)
    return np.interp(x, xp, st)


def local_ref(track, t0, t1, span=3.0, min_frames=25):
    """Speaker median F0 near the syllable (excluding the syllable itself)."""
    times, f0 = track["times"], track["f0_hz"]
    m = (times >= t0 - span) & (times <= t1 + span) & ~((times > t0) & (times < t1))
    seg = f0[m]
    seg = seg[np.isfinite(seg)]
    if seg.size >= min_frames:
        return float(np.median(seg))
    return track["median_hz"]


def syllable_features(track, t0, t1, ref_hz=None):
    """Syllable span -> acoustic features; None when no usable F0."""
    if not track or not track.get("median_hz"):
        return None
    raw = _interp_segment(track["f0_hz"], track["times"], t0, t1)
    if raw.size < 2:
        return None
    ref = ref_hz or local_ref(track, t0, t1)
    st = _hz_to_st(raw, ref)
    c10 = contour(track["f0_hz"], track["times"], ref, t0, t1)
    dur = max(t1 - t0, 1e-3)
    slope = float((st[-1] - st[0]) / dur)  # semitones per second
    voiced_ratio = float(np.isfinite(
        track["f0_hz"][(track["times"] >= t0) & (track["times"] <= t1)]
    ).mean()) if np.any((track["times"] >= t0) & (track["times"] <= t1)) else 0.0
    return {
        "f0_median_hz": round(float(np.median(raw)), 1),
        "level_st": round(float(np.mean(st)), 2),
        "slope_st_100ms": round(slope * 0.1, 2),
        "range_st": round(float(np.max(st) - np.min(st)), 2),
        "end_minus_start_st": round(float(st[-1] - st[0]), 2),
        "dur": round(dur, 3),
        "voiced_ratio": round(voiced_ratio, 3),
        "creak": round(1.0 - voiced_ratio, 3),
        "contour": [round(float(x), 2) for x in c10] if c10 is not None else None,
    }


def _proto_array(feats):
    c = feats.get("contour")
    if not c:
        return None
    xp = np.linspace(0, 1, len(c))
    x = np.linspace(0, 1, 5)
    return np.interp(x, xp, np.asarray(c, dtype=float))


def classify(feats, prototypes=None):
    """Return (tone|None, conf) against the given table (Tujia four tones by default)."""
    if not feats:
        return None, 0.0
    arr = _proto_array(feats)
    if arr is None:
        return None, 0.0
    table = prototypes or PROTOTYPES
    dists = sorted(
        ((float(np.mean(np.abs(arr - p))), tone) for tone, p in table.items())
    )
    d1, tone = dists[0]
    d2 = dists[1][0] if len(dists) > 1 else d1 + 1e-6
    gap = (d2 - d1) / d2 if d2 > 0 else 0.0
    conf = max(0.0, min(1.0, gap * (0.5 + 0.5 * feats.get("voiced_ratio", 0.0))))
    if feats.get("dur", 0) < 0.12 or feats.get("voiced_ratio", 0) < 0.25:
        return None, round(conf, 3)
    return (tone if conf >= 0.5 else None), round(conf, 3)


def dtw_distance(a, b):
    """DTW distance between two contours (works after time normalization)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("inf")
    d = np.abs(a[:, None] - b[None, :])
    acc = np.full_like(d, np.inf)
    acc[0, 0] = d[0, 0]
    for i in range(1, a.size):
        acc[i, 0] = d[i, 0] + acc[i - 1, 0]
    for j in range(1, b.size):
        acc[0, j] = d[0, j] + acc[0, j - 1]
    for i in range(1, a.size):
        for j in range(1, b.size):
            acc[i, j] = d[i, j] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[-1, -1] / (a.size + b.size))


def template_match(feats, templates):
    """Optional TTS four-template match (tone -> contour) -> (tone|None, conf)."""
    if not feats or not feats.get("contour") or not templates:
        return None, 0.0
    c = np.asarray(feats["contour"], float)
    ds = sorted((dtw_distance(c, np.asarray(t, float)), tone) for tone, t in templates.items())
    d1, tone = ds[0]
    d2 = ds[1][0] if len(ds) > 1 else d1 + 1e-6
    conf = (d2 - d1) / d2 if d2 > 0 else 0.0
    conf = max(0.0, min(1.0, conf)) * (0.5 + 0.5 * feats.get("voiced_ratio", 0.0))
    return (tone if conf >= 0.5 else None), round(conf, 3)
