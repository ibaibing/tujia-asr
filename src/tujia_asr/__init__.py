"""Tujia speech recognition.

    from tujia_asr import recognize

    result = recognize("speech.wav", dict_path="lexicon.jsonl")
    print(result["text"])

Without a lexicon the recogniser returns toned syllables; with one it returns
word-level output.
"""

__version__ = "1.1.0"

from .config import SUPPORTED_MODEL, Settings  # noqa: E402
from .pipeline import recognize  # noqa: E402

__all__ = ["Settings", "SUPPORTED_MODEL", "recognize", "__version__"]
