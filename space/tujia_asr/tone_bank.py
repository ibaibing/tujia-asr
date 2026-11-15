"""E2a: build and match templates from a bank of toned syllables."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import tone_acoustic as ta


def load_manifest(path):
    """Read manifest.csv; skip malformed rows."""
    path = Path(path).expanduser()
    if not path.is_file():
        return []
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                row["tone_value"] = int(row.get("tone_value") or 0)
            except ValueError:
                continue
            if not row.get("shape") or row["tone_value"] not in (55, 35, 21, 53):
                continue
            for key in ("start", "end", "dur", "voiced_ratio", "f0_median"):
                try:
                    row[key] = float(row.get(key) or 0)
                except ValueError:
                    row[key] = 0.0
            rows.append(row)
    return rows


def _load_audio(path):
    try:
        import librosa
    except ImportError:
        return None, 16000
    y, sr = librosa.load(str(path), sr=16000, mono=True)
    return y, sr


def build_templates(rows, root):
    """(speaker, shape, tone) -> median 10-point contour + IQR + duration/creak."""
    root = Path(root).expanduser()
    cache = {}
    groups = defaultdict(list)
    for r in rows:
        if not r.get("file") or r.get("rep") in ("", None):
            continue
        path = root / r["file"]
        if not path.is_file():
            continue
        key = (r.get("speaker", ""), r["shape"])
        if key not in cache:
            cache[key] = ta.f0_track(*_load_audio(path))
        track = cache[key]
        if not track:
            continue
        t0 = r.get("start") or 0.0
        t1 = r.get("end") or (t0 + r.get("dur") or 0.0)
        if t1 <= t0:
            t1 = t0 + (r.get("dur") or 0.5)
        feats = ta.syllable_features(track, t0, t1)
        if not feats:
            continue
        groups[(r.get("speaker", ""), r["shape"], r["tone_value"])].append((feats, r))

    templates = []
    for (speaker, shape, tone), items in groups.items():
        cs = np.array([f["contour"] for f, _ in items if f.get("contour")])
        if len(cs) == 0:
            continue
        med = np.median(cs, axis=0)
        iqr = np.percentile(cs, 75, axis=0) - np.percentile(cs, 25, axis=0)
        durs = np.array([f["dur"] for f, _ in items], dtype=float)
        creaks = np.array([f["creak"] for f, _ in items], dtype=float)
        templates.append(
            {
                "speaker": speaker,
                "shape": shape,
                "tone": tone,
                "contour": med,
                "iqr": iqr,
                "dur_med": float(np.median(durs)),
                "dur_std": float(np.std(durs)) or 0.05,
                "creak": float(np.median(creaks)),
                "n": len(items),
            }
        )
    return templates


def match(shape, feats, templates, speaker=None):
    """Same-shape templates -> {shape,tone,conf,dist,speaker,n_templates}; None if undecidable."""
    if not feats or not feats.get("contour") or not templates:
        return None
    same = [t for t in templates if t["shape"] == shape]
    if speaker:
        preferred = [t for t in same if t["speaker"] == speaker]
        same = preferred or same
    if not same:
        return None
    c = np.asarray(feats["contour"], float)
    scored = []
    for t in same:
        mae = float(np.mean(np.abs(c - t["contour"])))
        dz = abs(feats["dur"] - t["dur_med"]) / max(t["dur_std"], 1e-3)
        dc = abs(feats["creak"] - t["creak"])
        dist = 0.7 * mae + 0.2 * dz + 0.1 * dc
        scored.append((dist, t))
    scored.sort(key=lambda x: x[0])
    d1, best = scored[0]
    others = [d for d, t in scored if t["tone"] != best["tone"]]
    if not others:
        return {
            "shape": shape, "tone": best["tone"], "conf": 0.0, "dist": round(d1, 3),
            "speaker": best["speaker"], "n_templates": sum(t["n"] for t in same),
        }
    d2 = others[0]
    conf = (d2 - d1) / d2 if d2 > 0 else 0.0
    return {
        "shape": shape,
        "tone": best["tone"],
        "conf": round(max(0.0, min(1.0, conf)), 3),
        "dist": round(d1, 3),
        "speaker": best["speaker"],
        "n_templates": sum(t["n"] for t in same),
    }


def bank_root(settings):
    """Bank root, if it exists."""
    root = Path(settings.tone_bank).expanduser()
    return root if root.is_dir() else None


def load_bank(settings):
    """Return (templates, note); a missing bank yields ([], 'tone bank absent')."""
    root = bank_root(settings)
    if not root:
        return [], "tone bank absent"
    manifest = root / "manifest.csv"
    rows = load_manifest(manifest)
    if not rows:
        return [], f"manifest empty: {manifest}"
    return build_templates(rows, root), f"tone bank: {len(rows)} rows"
