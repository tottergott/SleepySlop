"""Silero TTS (torch.package from models.silero.ai)."""
import numpy as np
import torch


class SileroEngine:
    sr = 24000

    def __init__(self, path: str, device: str = "cuda"):
        imp = torch.package.PackageImporter(path)
        self.model = imp.load_pickle("tts_models", "model")
        self.model.to(torch.device(device))
        self.speakers = list(getattr(self.model, "speakers", []))

    def __call__(self, text: str, speaker: str = "aidar") -> np.ndarray:
        audio = self.model.apply_tts(text=text, speaker=speaker, sample_rate=self.sr, put_accent=True, put_yo=True)
        return audio.detach().cpu().numpy().astype(np.float32)
