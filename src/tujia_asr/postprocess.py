"""Inventory constraint for recognised syllables.

The acoustic model emits syllables from a 708-item training vocabulary. That
vocabulary was derived from the training transcriptions, so it inherits their
spelling: pre-fold rhymes such as ``iøŋ``, digraph onsets such as ``ʦʰ``, and
occasional OCR shapes. The frozen inventory in :mod:`tujia_nlp.inventory`
knows which syllables are attested and which spellings are the same phone.

Two modes, both side-effect free:

``check(tokens)``
    Count what is off the inventory. Never changes the output. Use this to see
    whether a model upgrade helped.

``fix(tokens)``
    Return a new token list with unknown shapes corrected. Two tiers, and they
    are kept apart on purpose:

    *spelling*  the inventory's own equivalence table (`ian` → `iã`, `ye` →
    `ɣɤ`, `khe` → `kʰɤ`). Deterministic: the same phones written another way.
    *fuzzy*     the nearest attested syllable by edit distance, requested with
    ``fuzzy=True``. This can be wrong, so it is never on by default.

    The original spelling is kept in ``shape_raw``; the tone is never touched.

A token is a dict with at least ``shape``. Extra keys (timing, scores, tone
evidence) are preserved as-is, so the output of :func:`tujia_asr.recognize`
can be constrained in place without touching the audio pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tujia_nlp import inventory as inventory_mod


@dataclass
class Report:
    """What the inventory made of a token sequence."""

    n_tokens: int = 0
    n_known: int = 0
    n_unknown: int = 0
    n_fixed_spelling: int = 0
    n_fixed_fuzzy: int = 0
    n_unfixed: int = 0
    version: str = ""
    unknown: dict = field(default_factory=dict)      # shape -> count
    fixes: dict = field(default_factory=dict)        # "raw -> fixed" -> count
    unfixed: dict = field(default_factory=dict)      # shape -> count

    @property
    def n_fixed(self) -> int:
        return self.n_fixed_spelling + self.n_fixed_fuzzy

    @property
    def unknown_rate(self) -> float:
        return self.n_unknown / self.n_tokens if self.n_tokens else 0.0

    @property
    def fix_rate(self) -> float:
        return self.n_fixed / self.n_unknown if self.n_unknown else 0.0

    def to_dict(self) -> dict:
        """Compact form for logging or a metrics file."""
        return {
            "version": self.version,
            "n_tokens": self.n_tokens,
            "n_known": self.n_known,
            "n_unknown": self.n_unknown,
            "n_fixed_spelling": self.n_fixed_spelling,
            "n_fixed_fuzzy": self.n_fixed_fuzzy,
            "n_unfixed": self.n_unfixed,
            "unknown_rate": round(self.unknown_rate, 4),
            "fix_rate": round(self.fix_rate, 4),
            "unknown": dict(sorted(self.unknown.items(), key=lambda kv: -kv[1])[:50]),
            "fixes": dict(sorted(self.fixes.items(), key=lambda kv: -kv[1])[:50]),
            "unfixed": dict(sorted(self.unfixed.items(), key=lambda kv: -kv[1])[:50]),
        }


def _inv(inv):
    return inv if inv is not None else inventory_mod.default()


def _row(inv, shape):
    """The inventory row for a shape *as written*, without canonicalising it.

    `inventory.status()` folds spellings first, so it would call `pian` known
    even though the table only has `piã`. Here "known" means "already the key
    we would emit"; a spelling variant is a fix, and is reported as one.
    """
    return inv.syllables.get(shape)


def check(tokens, inv=None, legal_only: bool = False) -> Report:
    """Count tokens the inventory does not know. Output is not modified."""
    inv = _inv(inv)
    report = Report(version=inv.version)
    for tok in tokens:
        shape = tok.get("shape") or ""
        report.n_tokens += 1
        row = _row(inv, shape)
        if row is not None and (not legal_only or row.status == inventory_mod.LEGAL):
            report.n_known += 1
        else:
            report.n_unknown += 1
            report.unknown[shape] = report.unknown.get(shape, 0) + 1
    return report


def fix(tokens, inv=None, legal_only: bool = False, fuzzy: bool = False, max_cost: int = 2):
    """Correct unknown spellings. Returns ``(new_tokens, report)``.

    Spelling corrections come from the inventory's equivalence table and are
    safe. ``fuzzy=True`` additionally snaps anything still unknown to the
    nearest attested syllable; those are counted separately in
    ``report.n_fixed_fuzzy`` because edit distance can land on the wrong word.
    Shapes with no candidate within ``max_cost`` are left alone and counted in
    ``report.unfixed``.
    """
    inv = _inv(inv)
    report = Report(version=inv.version)
    out = []
    for tok in tokens:
        shape = tok.get("shape") or ""
        report.n_tokens += 1
        row = _row(inv, shape)
        known = row is not None and (not legal_only or row.status == inventory_mod.LEGAL)
        if known:
            report.n_known += 1
            out.append(dict(tok))
            continue

        report.n_unknown += 1
        report.unknown[shape] = report.unknown.get(shape, 0) + 1

        target = inv.canonical(shape)
        target_row = _row(inv, target)
        accepted = target_row is not None and (
            not legal_only or target_row.status == inventory_mod.LEGAL)
        tier = "spelling"
        if not (target != shape and accepted):
            if not fuzzy:
                out.append(dict(tok))
                report.n_unfixed += 1
                report.unfixed[shape] = report.unfixed.get(shape, 0) + 1
                continue
            target = inv.nearest(shape, max_cost=max_cost, legal_only=legal_only)
            tier = "fuzzy"
            if not target:
                out.append(dict(tok))
                report.n_unfixed += 1
                report.unfixed[shape] = report.unfixed.get(shape, 0) + 1
                continue

        fixed = dict(tok)
        fixed["shape_raw"] = shape
        fixed["shape"] = target
        if fixed.get("ipa") == shape:
            fixed["ipa"] = target
        out.append(fixed)
        if tier == "spelling":
            report.n_fixed_spelling += 1
        else:
            report.n_fixed_fuzzy += 1
        key = f"{shape} -> {target}"
        report.fixes[key] = report.fixes.get(key, 0) + 1
    return out, report
