"""Lightweight RVC v1/v2 runner on top of Applio's inference code.

Applio embeds with ContentVec through `transformers` (no fairseq), which keeps
RVC in the same environment as F5-TTS. Applio resolves its weights relative
to the working directory, so every call runs inside its checkout.
"""
import contextlib
import gc
import os
import sys
import time

import librosa
import numpy as np
import torch

from voice_arena.paths import APPLIO


@contextlib.contextmanager
def _in_applio():
    prev = os.getcwd()
    os.chdir(APPLIO)
    try:
        yield
    finally:
        os.chdir(prev)


with _in_applio():
    sys.path.insert(0, str(APPLIO))
    import faiss  # noqa: E402
    import rvc.infer.pipeline as _applio_pipeline  # noqa: E402
    from rvc.infer.infer import VoiceConverter  # noqa: E402


class _CachedIndex:
    """Applio re-reads the FAISS index and calls reconstruct_n on every pipeline() call."""

    def __init__(self, index):
        self.index, self.ntotal, self._vectors = index, index.ntotal, None

    def reconstruct_n(self, start, n):
        if self._vectors is None:
            self._vectors = self.index.reconstruct_n(start, n)
        return self._vectors

    def search(self, *args, **kwargs):
        return self.index.search(*args, **kwargs)


class _FaissCache:
    def __init__(self):
        self._cache = {}

    def read_index(self, path):
        if path not in self._cache:
            self._cache[path] = _CachedIndex(faiss.read_index(path))
        return self._cache[path]


_applio_pipeline.faiss = _FaissCache()


class RVCEngine:
    def __init__(self):
        with _in_applio():
            self.vc = VoiceConverter()
            self.vc.load_hubert("contentvec")
            self.vc.last_embedder_model = "contentvec"
        self.model_path = None
        self.index_path = None
        self.last_load_ms = 0.0

    @property
    def sr(self) -> int:
        return int(self.vc.tgt_sr)

    def load(self, pth: str, index: str | None):
        if pth == self.model_path:
            return
        t = time.perf_counter()
        with _in_applio():
            self.vc.get_vc(pth, 0)
        # The previous generator sits in reference cycles until a GC pass: without this,
        # every voice switch leaks ~100 MB of VRAM.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model_path, self.index_path = pth, index or ""
        self.last_load_ms = (time.perf_counter() - t) * 1000

    def __call__(self, audio: np.ndarray, sr: int, pitch: int = 0, index_rate: float = 0.6,
                 protect: float = 0.33, volume_envelope: float = 0.25, autotune: float = 0.0) -> np.ndarray:
        x = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else audio
        peak = np.abs(x).max() / 0.95
        if peak > 1:
            x = x / peak
        with _in_applio(), torch.inference_mode():
            out = self.vc.vc.pipeline(
                model=self.vc.hubert_model, net_g=self.vc.net_g, sid=0, audio=x, pitch=pitch,
                f0_method="rmvpe", file_index=self.index_path, index_rate=index_rate,
                pitch_guidance=self.vc.use_f0, volume_envelope=volume_envelope, version=self.vc.version,
                protect=protect, f0_autotune=autotune > 0, f0_autotune_strength=max(autotune, 1e-3),
                proposed_pitch=False, proposed_pitch_threshold=155.0,
            )
        out = np.asarray(out)
        if out.dtype == np.int16:
            out = out.astype(np.float32) / 32768.0
        return out.astype(np.float32)
