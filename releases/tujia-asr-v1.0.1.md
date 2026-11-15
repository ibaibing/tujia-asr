# tujia-asr 1.0.1

Patch release about *how* v0.3 is decoded, not what it learned. No new weights.

| | |
| --- | --- |
| Package | `tujia-asr` 1.0.1 |
| Model | `tujia-ctc-v0.3` (training run `v03j`), unchanged |
| Demo | <https://modelscope.cn/studios/TujiaNLP/tujia-asr-demo> |

## Fixes

- **Same audio no longer transcribes differently between runs.** `syllabify.build()`
  iterated a set, and the DP kept the first candidate that reached a tie, so every new
  process could pick a different syllable wherever two shapes scored equally. The
  inventory is now sorted, which makes the transcript deterministic.
- **`__version__` matches `pyproject.toml`.** 1.0.0 shipped a wheel whose runtime
  `tujia_asr.__version__` still said `1.0.0` while the package metadata said the same —
  the two are now checked against each other in the test suite.

## New: `--blank-bias`

v0.3 under-generates. On the 353-clip held-out set it deletes 41% of the reference
syllables and emits 62% of the expected length, against 24% deletions and 0.86 length for
v0.2. When a human proofreads the output, a visible wrong syllable is easier to correct
than a gap, so the CTC blank logit can now be lowered:

| Bias | SER | Sub | Del | Ins | Hyp/ref length |
| --- | --- | --- | --- | --- | --- |
| 0 (default) | 0.759 | 0.314 | 0.410 | 0.034 | 0.62 |
| 1.0 | 0.764 | 0.400 | 0.310 | 0.055 | 0.75 |
| 1.5 | 0.775 | 0.429 | 0.273 | 0.073 | 0.80 |

`--blank-bias 1.0` or `TUJIA_BLANK_BIAS=1.0`. The demo Space runs at 1.0; the library
default stays 0 so existing evaluations keep their numbers.

The sweep reads reference labels through the v0.3i vocabulary, which is why its bias-0
row (0.759) sits 2.6 points above the 0.7330 in the README; the rows are comparable to
each other.

## Space build note

The Space installs 1.0.1 from the wheel published at
`TujiaNLP/tujia-ctc-v0.3/wheels/tujia_asr-1.0.1-py3-none-any.whl`, because the PyPI
mirrors used by the builder had not synced the new version yet. Switch
`requirements.txt` back to `tujia-asr==1.0.1` once the mirrors catch up.
