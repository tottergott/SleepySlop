"""Quality metrics: faster-whisper WER/CER and WavLM speaker similarity."""
from functools import lru_cache

import jiwer
import librosa
import numpy as np
import torch

from voice_arena.text import normalize_for_wer


@lru_cache(maxsize=1)
def whisper(size: str = "small"):
    import torch  # noqa: F401  (registers torch's CUDA/cuDNN DLL directory for ctranslate2 on Windows)
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(size, device="cuda", compute_type="float16")
    except Exception:
        return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(audio: np.ndarray, sr: int, language: str = "ru") -> str:
    x = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else audio
    segments, _ = whisper().transcribe(x.astype(np.float32), language=language, beam_size=5,
                                       vad_filter=False, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segments).strip()


def wer_cer(reference: str, hypothesis: str) -> tuple[float, float]:
    ref, hyp = normalize_for_wer(reference), normalize_for_wer(hypothesis)
    if not ref:
        return 0.0, 0.0
    if not hyp:
        return 100.0, 100.0
    return 100 * jiwer.wer(ref, hyp), 100 * jiwer.cer(ref, hyp)


class SpeakerSim:
    """Cosine similarity of WavLM-base-plus-sv x-vectors (closer to 1 = same speaker)."""

    def __init__(self, path: str, device: str = "cuda"):
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.fe = AutoFeatureExtractor.from_pretrained(path)
        self.model = WavLMForXVector.from_pretrained(path).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def embed(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        x = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else audio
        inputs = self.fe(x, sampling_rate=16000, return_tensors="pt").to(self.device)
        emb = self.model(**inputs).embeddings
        return torch.nn.functional.normalize(emb, dim=-1)[0]

    def centroid(self, paths) -> torch.Tensor:
        """Mean x-vector of many real lines: a steadier target than a single reference clip."""
        embs = []
        for p in paths:
            y, sr = librosa.load(p, sr=16000, mono=True)
            embs.append(self.embed(y, sr))
        return torch.nn.functional.normalize(torch.stack(embs).mean(0), dim=-1)

    def __call__(self, a: np.ndarray, sr_a: int, ref_emb: torch.Tensor) -> float:
        return float((self.embed(a, sr_a) * ref_emb).sum())
