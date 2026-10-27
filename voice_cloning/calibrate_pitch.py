"""Suggest rvc_pitch per voice so the RVC output's median F0 matches the real character.

RVC keeps the base voice's F0 contour and shifts it by `pitch` semitones, so the
matching shift is 12*log2(real median F0 / Silero speaker median F0).

    .venv/Scripts/python calibrate_pitch.py
"""
import json

import librosa
import numpy as np

from voice_arena.characters import PHRASES, PROFILES, phrase_text
from voice_arena.paths import RESULTS
from voice_arena.pipelines import Engines, load_manifest, load_references
from voice_arena.text import numbers_to_words
from voice_arena.voice_stats import f0_stats

refs, E = load_references(), Engines(load_manifest())
silero = E.get("silero")
out = {}
for key, prof in PROFILES.items():
    if key not in refs:
        print(f"{key:20s} no real clips -> keep rvc_pitch={prof.rvc_pitch}")
        continue
    real = [f0_stats(*librosa.load(p, sr=16000))[0] for p in refs[key]["sim_set"]]
    real_med = float(np.median([x for x in real if x]))
    base = [f0_stats(silero(numbers_to_words(phrase_text(pid, prof)), prof.silero_speaker), silero.sr)[0] for pid, _ in PHRASES]
    base_med = float(np.median([x for x in base if x]))
    shift = 12 * np.log2(real_med / base_med)
    out[key] = {"real_f0": real_med, "silero_speaker": prof.silero_speaker, "silero_f0": base_med,
                "exact_shift": round(float(shift), 2), "suggested_pitch": int(round(shift)), "current_pitch": prof.rvc_pitch}
    print(f"{key:20s} real {real_med:5.0f} Hz  {prof.silero_speaker:8s} {base_med:5.0f} Hz  -> shift {shift:+.1f} st "
          f"(now {prof.rvc_pitch:+d})")
(RESULTS / "pitch_calibration.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
