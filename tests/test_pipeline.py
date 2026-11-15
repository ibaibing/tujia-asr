import os

import pytest


@pytest.mark.skipif(not os.environ.get("TUJIA_CTC_CKPT"), reason="no CTC checkpoint configured")
def test_recognize_one_file():
    wav = os.environ.get("TUJIA_SMOKE_WAV")
    if not wav:
        pytest.skip("TUJIA_SMOKE_WAV not set")
    dicts = [p for p in os.environ.get("TUJIA_DICTS", "").split(os.pathsep) if p.strip()]

    from tujia_asr import recognize

    res = recognize(wav, dict_path=dicts or None)
    assert res["schema"] == "tujia.asr/1"
    assert res["syllables"]
    assert res["text"]
    assert res["audio"]["dur"] > 0
