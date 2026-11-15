"""Audio loading shared by the acoustic front-ends."""

import numpy as np
import soundfile as sf

SR = 16000


def read_mono(path, offset=None, duration=None, sr=SR):
    """Read a time span as mono float32 at ``sr``; resamples when needed."""
    info = sf.info(str(path))
    start = int((offset or 0) * info.samplerate)
    stop = int(((offset or 0) + duration) * info.samplerate) if duration else None
    y, file_sr = sf.read(
        str(path), dtype="float32", always_2d=False, start=start, stop=stop
    )
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        import torch
        import torchaudio

        t = torch.from_numpy(np.ascontiguousarray(y))
        y = torchaudio.functional.resample(t, file_sr, sr).numpy()
    return np.ascontiguousarray(y, dtype=np.float32)
