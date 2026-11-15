# Test fixtures

Audio used for end-to-end checks is **not** committed: the recordings are
licensed for research use and must not be redistributed.

`test_pipeline.py` is skipped unless you point it at local material:

```bash
export TUJIA_CTC_CKPT=/path/to/tujia-ctc-v0.2.pt
export TUJIA_XLSR=/path/to/tujia-xlsr53-ft
export TUJIA_SMOKE_WAV=/path/to/clip.wav
export TUJIA_DICTS=/path/to/lexicon.jsonl   # optional
pytest -q
```

`test_smoke.py` needs no model and always runs.
