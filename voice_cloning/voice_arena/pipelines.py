"""The three competing architectures, each exposed as a chunk-streaming generator.

    Mode 1  f5      F5-TTS_RUSSIAN zero-shot clone from the 7 s character reference
    Mode 2  rvc     Silero base voice -> RVC v2 character model (RMVPE pitch)
    Mode 3  native  native character TTS (TeraTTS / Piper) or Silero, + character DSP
"""
import gc
import json
from dataclasses import dataclass

import numpy as np
import torch

from voice_arena import dsp
from voice_arena.characters import Profile
from voice_arena.paths import MANIFEST, REFERENCES
from voice_arena.text import merge_short, numbers_to_words, split_chunks
from voice_arena.voice_stats import matched_eq

GAP_S = 0.08  # pause inserted between streamed chunks when they are joined


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_references() -> dict:
    p = REFERENCES / "references.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@dataclass
class Variant:
    mode: str          # f5 | rvc | native
    name: str          # unique within a character, e.g. "rvc:bdwind_pudge"
    label: str         # human readable
    info: dict


class Engines:
    """Lazily created engines, so a benchmark phase only holds the weights it uses."""

    def __init__(self, manifest: dict, device: str = "cuda"):
        self.m, self.device, self._e = manifest, device, {}

    def get(self, name: str):
        if name not in self._e:
            eng = self.m["engines"]
            if name == "f5":
                from voice_arena.engines.f5 import F5Engine
                self._e[name] = F5Engine(eng["f5"]["ckpt"], eng["f5"]["vocab"], self.device)
            elif name == "silero":
                from voice_arena.engines.silero import SileroEngine
                self._e[name] = SileroEngine(eng["silero"]["path"], self.device)
            elif name == "rvc":
                from voice_arena.engines.rvc import RVCEngine
                self._e[name] = RVCEngine()
            elif name == "tera":
                from voice_arena.engines.onnx_vits import TeraVits
                t = eng["tera_glados"]
                self._e[name] = TeraVits(t["model.onnx"], t["config.json"], t.get("dictionary.txt"))
            elif name.startswith("piper_"):
                from voice_arena.engines.onnx_vits import PiperVoice
                self._e[name] = PiperVoice(eng[name]["onnx"], eng[name]["json"])
            else:
                raise KeyError(name)
        return self._e[name]

    def unload(self, *names):
        for n in names or list(self._e):
            self._e.pop(n, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def variants_for(profile: Profile, manifest: dict, refs: dict) -> list[Variant]:
    out = []
    if "f5" in manifest["engines"] and profile.key in refs:
        r = refs[profile.key]
        eq = " + EQ" if "f5" in profile.eq else ""
        out.append(Variant("f5", "f5:nfe32", f"F5-TTS Zero-Shot (NFE 32){eq}", {"ref": r, "nfe": 32, "source": r["source"]}))
        out.append(Variant("f5", "f5:nfe16", f"F5-TTS Zero-Shot (NFE 16, быстрый){eq}", {"ref": r, "nfe": 16,
                                                                                        "source": r["source"]}))
    for e in manifest["rvc"].get(profile.key, []):
        name = f"rvc:{e['variant']}"
        eq = " + EQ" if name in profile.eq else ""
        out.append(Variant("rvc", name, f"Silero ({profile.silero_speaker}, {profile.rvc_pitch:+d} пт) → RVC "
                                        f"({e['variant']}){eq}", e))
    dsp_label = "" if profile.native_fx.get("bypass") else " + DSP"
    if profile.native == "tera" and "tera_glados" in manifest["engines"]:
        eq = " + EQ" if "native:tera" in profile.eq else ""
        out.append(Variant("native", "native:tera", f"TeraTTS VITS (ru){dsp_label}{eq}", {"engine": "tera"}))
    elif profile.native.startswith("piper:") and profile.native[6:] in manifest["engines"]:
        key = profile.native[6:]
        eq = " + EQ" if f"native:{key}" in profile.eq else ""
        out.append(Variant("native", f"native:{key}", f"Piper EN-голос + ru-фонемы{dsp_label}{eq}", {"engine": key}))
    elif "silero" in manifest["engines"]:
        out.append(Variant("native", "native:silero_dsp", f"Silero ({profile.silero_speaker}) + DSP",
                           {"engine": "silero"}))
    return out


class Synthesizer:
    def __init__(self, engines: Engines):
        self.E = engines

    def prepare(self, profile: Profile, v: Variant) -> float:
        """Load per-voice weights; returns the switch time in ms (RVC model swap)."""
        if v.mode == "rvc":
            rvc = self.E.get("rvc")
            rvc.load(v.info["pth"], v.info.get("index"))
            return rvc.last_load_ms
        return 0.0

    def sample_rate(self, v: Variant) -> int:
        if v.mode == "f5":
            return 24000
        if v.mode == "rvc":
            return self.E.get("rvc").sr
        eng = self.E.get(v.info["engine"])
        return eng.sr

    def stream(self, profile: Profile, v: Variant, text: str):
        fx = dsp.FX[profile.key]
        chunks = split_chunks(numbers_to_words(text))
        if v.mode == "f5":
            chunks = merge_short(chunks)
        for chunk in chunks:
            if v.mode == "f5":
                r = v.info["ref"]
                y, sr = self.E.get("f5")(chunk, r["wav"], r["text"], r.get("lang", "ru"), nfe_step=v.info["nfe"]), 24000
                if profile.post_fx:
                    y = fx(y, sr)
            elif v.mode == "rvc":
                silero = self.E.get("silero")
                base = silero(chunk, profile.silero_speaker)
                rvc = self.E.get("rvc")
                # Death Knight: the Frostmourne -2 st is rendered by RVC itself (free) instead of a pitch shifter.
                fold = -2 if profile.post_fx and profile.key == "arthas_deathknight" else 0
                y = rvc(base, silero.sr, pitch=profile.rvc_pitch + fold, index_rate=profile.rvc_index_rate,
                        protect=profile.rvc_protect, autotune=profile.rvc_autotune)
                sr = rvc.sr
                if profile.post_fx:
                    y = fx(y, sr, pre_shifted=True) if fold else fx(y, sr)
            else:
                eng = self.E.get(v.info["engine"])
                if v.info["engine"] == "silero":
                    y = eng(chunk, profile.silero_speaker)
                    y = fx(y, eng.sr, base_voice=True) if profile.key in ("pudge", "witch_doctor") else fx(y, eng.sr)
                else:
                    raw = eng(chunk)
                    fx_kw = {k: v for k, v in profile.native_fx.items() if k != "bypass"}
                    y = raw if profile.native_fx.get("bypass") else fx(raw, eng.sr, **fx_kw)
                sr = eng.sr
            eq = profile.eq.get(v.name) or profile.eq.get(v.mode)  # "rvc:<model>", "native:<engine>" or "f5"
            if eq:
                y = matched_eq(y, sr, eq)
            yield np.asarray(y, dtype=np.float32), sr


def join_chunks(chunks, sr: int) -> np.ndarray:
    gap = np.zeros(int(GAP_S * sr), dtype=np.float32)
    parts = []
    for i, c in enumerate(chunks):
        if i:
            parts.append(gap)
        parts.append(c)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
