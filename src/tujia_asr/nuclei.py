"""Syllable-nucleus detection from acoustic cues alone.

1. short-time energy -> dB, smoothed;
2. voiced region = energy above "peak - drop_db", split by pauses >= min_gap;
3. within a region, split at energy-valley prominence to approximate syllables;
4. over-long pieces are cut further at the lowest energy points.

The output is plain time spans ``[(t0, t1), ...]`` in seconds.  It assumes only
that speech has syllable structure and energy valleys between syllables, so it
works for Tujia, Chinese, and mixed speech.
"""

import librosa
import numpy as np
from scipy.signal import find_peaks

SR = 16000
HOP = 0.01          # 10 ms
WIN = 0.025         # RMS window (s)
SMOOTH = 5          # smoothing frames
DROP_DB = 30.0      # voiced threshold: below the 95th percentile by this much
MIN_GAP = 0.06      # pauses shorter than this are not split points
MIN_NUC = 0.07      # shortest nucleus
MAX_CORE = 0.40     # longer pieces are split again
SPLIT_PROM = 3.0    # valley prominence (dB)
SPLIT_DIST = 0.12   # minimum split distance (s)


def _envelope(y, sr):
    rms = librosa.feature.rms(y=y, frame_length=int(WIN * sr), hop_length=int(HOP * sr))[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    return np.convolve(db, np.ones(SMOOTH) / SMOOTH, mode="same")


def _regions(env, drop_db=DROP_DB, min_gap=MIN_GAP):
    thr = float(np.percentile(env, 95) - drop_db)
    act = env > thr
    regs, i, n = [], 0, len(act)
    while i < n:
        if act[i]:
            j = i
            while j < n and act[j]:
                j += 1
            if regs and (i - regs[-1][1]) * HOP < min_gap:
                regs[-1][1] = j
            else:
                regs.append([i, j])
            i = j
        else:
            i += 1
    return regs


def detect(y, sr=SR, drop_db=DROP_DB, min_gap=MIN_GAP, min_dur=MIN_NUC,
           split_prom=SPLIT_PROM, split_dist=SPLIT_DIST, max_core=MAX_CORE):
    """Waveform -> syllable nucleus spans [(t0, t1), ...]."""
    env = _envelope(y, sr)
    out = []
    for a, b in _regions(env, drop_db, min_gap):
        seg = env[a:b]
        idx, _ = find_peaks(-seg, distance=max(1, int(split_dist / HOP)),
                            prominence=split_prom)
        pts = [a] + [a + int(k) for k in idx] + [b]
        pieces = [(pts[k], pts[k + 1]) for k in range(len(pts) - 1)]
        final = []
        for p0, p1 in pieces:
            n_split = int(np.ceil((p1 - p0) * HOP / max_core)) - 1
            if n_split > 0:
                m = p1 - p0
                cand = [(env[p0 + k], p0 + k) for k in range(2, m - 2)]
                cand.sort()
                picks = []
                for _, pos in cand:
                    if len(picks) == n_split:
                        break
                    if all(abs(pos - q) * HOP >= min_dur * 1.5 for q in picks):
                        picks.append(pos)
                cuts = sorted(picks)
                prev = p0
                for c in cuts:
                    final.append((prev, c))
                    prev = c
                final.append((prev, p1))
            else:
                final.append((p0, p1))
        for p0, p1 in final:
            t0, t1 = p0 * HOP, p1 * HOP
            if t1 - t0 >= min_dur:
                out.append((round(t0, 3), round(t1, 3)))
    return out


def detect_file(path, **kw):
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    return detect(y, sr, **kw)
