"""Lexicon side: entry loading, syllable inventory and tone counts.

Orthography handling (punctuation, symbol expansion, toned syllable splitting)
comes from :mod:`tujia_nlp.syllable`; this module only keeps the lexicon-specific
assembly.
"""

import json
import os
from collections import Counter, defaultdict

from tujia_nlp.syllable import (  # noqa: F401
    TONE_VALUES,
    clean_ipa,
    expand_symbols,
    phone_seq,
    split_head,
)

# Legacy alias kept for callers of the old engine.
normalize_shape = expand_symbols


def read_entries(path):
    """Read lexicon rows from ``path`` (None, one file, or several files)."""
    if path is None or path == "":
        paths = []
    elif isinstance(path, (str, os.PathLike)):
        paths = [path]
    else:
        paths = list(path)
    entries = []
    for p in paths:
        if not str(p).strip():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


class Lexicon:
    def __init__(self, entries):
        # The same headword may come from several lexicons (built-in pack plus
        # one or more user files). Keep the first occurrence so that rows loaded
        # earlier win, and so a duplicate does not inflate the word count.
        seen, unique = set(), []
        for e in entries:
            head = e.get("w", "")
            if head:
                if head in seen:
                    continue
                seen.add(head)
            unique.append(e)
        entries = unique
        self.entries = entries
        self.tone_counts = defaultdict(Counter)       # shape -> Counter(tone)
        self.word_pairs = defaultdict(int)            # (shape, tone) -> count
        self.word_counts = Counter()                  # word -> entry count
        for e in entries:
            head = e.get("w", "")
            if head:
                self.word_counts[head] += 1
            for shape, tone in split_head(head):
                if not shape or tone is None or tone not in TONE_VALUES:
                    continue
                self.tone_counts[shape][tone] += 1
                self.word_pairs[(shape, tone)] += 1

    @classmethod
    def load(cls, path):
        """Read a lexicon; ``path`` may be None, one file, or several files."""
        return cls(read_entries(path))

    def extend(self, entries):
        """Append rows (e.g. a user lexicon) and refresh the syllable counts."""
        fresh = list(entries)
        if not fresh:
            return self
        self.__init__(self.entries + fresh)
        return self

    def add_variants(self, pairs):
        """Merge grammatical variant (shape, tone) pairs without counting them."""
        for shape, tone in pairs:
            if shape and tone is not None:
                self.tone_counts[shape][tone] += 0
                self.word_pairs[(shape, tone)] += 0

    def shapes(self):
        return set(self.tone_counts)

    def tones_for(self, shape):
        """Tones seen for this shape in the lexicon, most frequent first."""
        return [t for t, _ in self.tone_counts.get(shape, Counter()).most_common()]

    def syllable_count(self):
        return len(self.tone_counts)

    def pair_count(self):
        return sum(len(v) for v in self.tone_counts.values())
