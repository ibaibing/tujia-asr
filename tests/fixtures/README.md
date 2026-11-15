# Test fixtures

Audio used for end-to-end checks is **not** committed here: the corpus recordings
are licensed for research use and must not be redistributed.

One clip is public and does not live in this directory: `space/examples/sample_field.wav`
is a short field recording whose speaker authorised public demo use, so it is committed
and used by the Space and by the example commands in the README.

`test_pipeline.py` is skipped unless you point it at local material:

```bash
export TUJIA_CTC_CKPT=/path/to/tujia-ctc-v0.2.pt
export TUJIA_XLSR=/path/to/tujia-xlsr53-ft
export TUJIA_SMOKE_WAV=/path/to/clip.wav
export TUJIA_DICTS=/path/to/lexicon.jsonl   # optional
pytest -q
```

`test_smoke.py` needs no model and always runs.
