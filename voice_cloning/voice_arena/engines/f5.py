"""F5-TTS_RUSSIAN zero-shot cloning (Misha24-10 fine-tune of F5TTS_v1_Base)."""
import contextlib
import io

import numpy as np

from voice_arena.text import add_stress


class _NoProgress:
    @staticmethod
    def tqdm(it, *args, **kwargs):
        return it


class F5Engine:
    sr = 24000

    def __init__(self, ckpt: str, vocab: str, device: str = "cuda", nfe_step: int = 32):
        from f5_tts.api import F5TTS

        self.tts = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab, device=device)
        self.nfe_step = nfe_step

    def __call__(self, text: str, ref_wav: str, ref_text: str, ref_lang: str = "ru", seed: int = 1234,
                 nfe_step: int | None = None, speed: float = 0.9) -> np.ndarray:
        # speed < 1 gives F5's length estimate ~10 % headroom; at 1.0 fast references clip word endings.
        # The v2 checkpoint was trained on fully stress-marked text ("молок+о").
        gen = add_stress(text)
        ref = add_stress(ref_text) if ref_lang == "ru" else ref_text
        with contextlib.redirect_stdout(io.StringIO()):  # infer_process prints ref/gen text per call
            wav, _, _ = self.tts.infer(ref_wav, ref, gen, show_info=lambda *a, **k: None, progress=_NoProgress,
                                       nfe_step=nfe_step or self.nfe_step, seed=seed, remove_silence=False,
                                       speed=speed)
        return np.asarray(wav, dtype=np.float32)
