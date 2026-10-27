"""Switch the assistant's character voice on the fly.

    from voice_assistant_router import VoiceAssistantRouter

    assistant = VoiceAssistantRouter(policy="balanced")
    assistant.set_voice("arthas_deathknight")
    audio, sr = assistant.say("Ледяная Скорбь жаждет душ.")
    for chunk, sr in assistant.stream("Слушаю тебя. Чего опять сломалось?"):
        play(chunk, sr)                       # first chunk arrives after TTFA

The variant used for each voice (F5 / Silero->RVC / native+DSP) is chosen from
results/benchmark.json by policy:
    speed     lowest TTFA among variants with CER <= 20 %
    quality   highest speaker similarity among variants with CER <= 15 %
    balanced  highest similarity among variants with CER <= 15 % and TTFA < 350 ms,
              falling back to "quality"
    (voices without real recordings have no similarity: the RVC character model with the lowest CER)
"""
import json
import time

import numpy as np
import soundfile as sf

from voice_arena.characters import PROFILES
from voice_arena.paths import RESULTS
from voice_arena.pipelines import (Engines, Synthesizer, join_chunks, load_manifest, load_references,
                                   variants_for)


def _summaries(records):
    by = {}
    for r in records:
        if "error" in r or r.get("cer") is None:
            continue
        by.setdefault((r["voice"], r["variant"]), []).append(r)
    out = {}
    for (voice, variant), rs in by.items():
        sims = [r["sim"] for r in rs if r.get("sim") is not None]
        out.setdefault(voice, {})[variant] = {
            "ttfa": float(np.mean([r["ttfa_ms"] for r in rs])),
            "cer": float(np.mean([r["cer"] for r in rs])),
            "sim": float(np.mean(sims)) if sims else 0.0,
        }
    return out


MODE_PRIORITY = {"rvc": 0, "f5": 1, "native": 2}  # character model before generic DSP when similarity is unknown


def pick_variant(stats: dict, policy: str = "balanced") -> str | None:
    if not stats:
        return None
    ok = {k: s for k, s in stats.items() if s["cer"] <= 15}
    if policy != "speed" and not any(s["sim"] for s in stats.values()):
        # No real recordings -> no similarity score: take the best-understood character model.
        pool = ok or stats
        return min(pool, key=lambda k: (MODE_PRIORITY[k.split(":")[0]], pool[k]["cer"]))
    if policy == "speed":
        pool = {k: s for k, s in stats.items() if s["cer"] <= 20} or stats
        return min(pool, key=lambda k: pool[k]["ttfa"])
    if policy == "balanced":
        fast = {k: s for k, s in ok.items() if s["ttfa"] < 350}
        if fast:
            return max(fast, key=lambda k: fast[k]["sim"])
    if ok:
        return max(ok, key=lambda k: ok[k]["sim"])
    return min(stats, key=lambda k: stats[k]["cer"])


class VoiceAssistantRouter:
    VOICES = list(PROFILES)

    def __init__(self, default_voice: str = "glados", policy: str = "balanced", device: str = "cuda",
                 overrides: dict | None = None):
        self.manifest, self.refs = load_manifest(), load_references()
        self.engines = Engines(self.manifest, device)
        self.synth = Synthesizer(self.engines)
        self.policy = policy
        self.overrides = overrides or {}  # {"pudge": "rvc:bdwind_pudge"}
        bench = RESULTS / "benchmark.json"
        self.stats = _summaries(json.loads(bench.read_text(encoding="utf-8"))["records"]) if bench.exists() else {}
        self.voice = self.variant = None
        self.set_voice(default_voice)

    def variants(self, voice: str) -> list[str]:
        return [v.name for v in variants_for(PROFILES[voice], self.manifest, self.refs)]

    def set_voice(self, voice: str, variant: str | None = None) -> float:
        """Activate a character; returns the switch time in ms (RVC weight swap, engine load)."""
        if voice not in PROFILES:
            raise ValueError(f"unknown voice {voice!r}; choose from {self.VOICES}")
        available = {v.name: v for v in variants_for(PROFILES[voice], self.manifest, self.refs)}
        name = (variant or self.overrides.get(voice) or pick_variant(self.stats.get(voice, {}), self.policy)
                or next(iter(available)))
        if name not in available:
            raise ValueError(f"{voice}: variant {name!r} not available; have {list(available)}")
        t = time.perf_counter()
        self.voice, self.variant = PROFILES[voice], available[name]
        self.synth.prepare(self.voice, self.variant)
        # Free engines the new variant does not need (keeps VRAM low when hopping between modes).
        keep = {"f5": {"f5"}, "rvc": {"silero", "rvc"}, "native": {self.variant.info.get("engine")}}[self.variant.mode]
        stale = [n for n in list(self.engines._e) if n not in keep]
        if stale:  # unload() with no names clears everything
            self.engines.unload(*stale)
        return (time.perf_counter() - t) * 1000

    def stream(self, text: str):
        yield from self.synth.stream(self.voice, self.variant, text)

    def say(self, text: str) -> tuple[np.ndarray, int]:
        chunks, sr = [], 24000
        for chunk, sr in self.stream(text):
            chunks.append(chunk)
        return join_chunks(chunks, sr), sr

    def save(self, text: str, path: str) -> str:
        audio, sr = self.say(text)
        sf.write(path, audio, sr)
        return path

    def __repr__(self):
        return f"VoiceAssistantRouter(voice={self.voice.key!r}, variant={self.variant.name!r}, policy={self.policy!r})"


if __name__ == "__main__":
    from voice_arena.paths import OUTPUTS

    out = OUTPUTS / "router"
    out.mkdir(parents=True, exist_ok=True)
    assistant = VoiceAssistantRouter()
    for key in VoiceAssistantRouter.VOICES:
        ms = assistant.set_voice(key)
        path = assistant.save("Слушаю тебя. Чего опять сломалось?", str(out / f"{key}.wav"))
        print(f"{assistant}  switch {ms:.0f} ms -> {path}")
