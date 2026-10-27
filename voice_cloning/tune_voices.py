"""Tune every voice with real Russian clips towards the real character.

WavLM similarity alone saturates (good candidates all score ~0.95), so each
candidate is measured against the character's real lines (references.json sim_set):

    gap   SIM to the character minus SIM to the Silero base voice (how far it moved)
    ltas  long-term spectrum distance, dB (the game's mic/processing colour)
    f0    |median pitch error| in semitones + |intonation range error|
    cer   Whisper CER, must stay <= 5 %

Winner = lowest rank sum of sim, gap, ltas, f0. Searched per voice:
  * Silero -> RVC: every RVC model x base speaker x pitch around the F0-matched shift
  * matched EQ (real LTAS / synth LTAS) per variant, kept only if it helps
  * Native and F5: current settings vs + matched EQ (and raw TeraTTS vs DSP)

Writes results/voice_tuning.json; voice_arena.characters applies it to the profiles.

    .venv/Scripts/python tune_voices.py [--voices glados,pudge]
"""
import argparse
import json

import librosa
import numpy as np
import soundfile as sf
import torch

from voice_arena import dsp
from voice_arena.characters import PHRASES, PROFILES, phrase_text
from voice_arena.metrics import SpeakerSim, transcribe, wer_cer
from voice_arena.paths import OUTPUTS, RESULTS
from voice_arena.pipelines import Engines, load_manifest, load_references
from voice_arena.text import numbers_to_words
from voice_arena.voice_stats import band_mask, f0_stats, ltas, ltas_distance, matched_eq, matched_eq_gains

EXTRA = ["Задача выполнена. Можешь сказать спасибо.", "Я проверил систему: всё работает, но ненадолго."]
SPEAKERS = {"female": ("xenia", "kseniya", "baya"), "male": ("aidar", "eugene")}
OUT = OUTPUTS / "tuning"


class Judge:
    def __init__(self, sim, key, ref, silero):
        self.sim, self.key, self.silero = sim, key, silero
        real = [librosa.load(p, sr=24000) for p in ref["sim_set"]]
        self.target = sim.centroid(ref["sim_set"])
        self.ltas = np.mean([ltas(y, sr) for y, sr in real], axis=0)
        self.mask = band_mask(min(sf.info(p).samplerate for p in ref["sim_set"]))
        f0 = [s for s in (f0_stats(y, sr) for y, sr in real) if s[0]]
        self.f0_med = float(np.median([s[0] for s in f0]))
        self.f0_range = float(np.median([s[1] for s in f0]))
        prof = PROFILES[key]
        self.texts = [phrase_text(pid, prof) for pid, _ in PHRASES] + EXTRA
        self._base, self._base_emb = {}, {}
        self.rows = []

    def base(self, speaker, text):
        if (speaker, text) not in self._base:
            self._base[(speaker, text)] = self.silero(numbers_to_words(text), speaker)
        return self._base[(speaker, text)]

    def base_emb(self, speaker):
        if speaker not in self._base_emb:
            e = [self.sim.embed(self.base(speaker, t), self.silero.sr) for t in self.texts]
            self._base_emb[speaker] = torch.nn.functional.normalize(torch.stack(e).mean(0), dim=-1)
        return self._base_emb[speaker]

    def evaluate(self, name, synth, base_speaker, **meta):
        outs = [synth(t) for t in self.texts]
        d = OUT / self.key
        d.mkdir(parents=True, exist_ok=True)
        for i, (y, sr) in enumerate(outs[:2]):
            sf.write(d / f"{name.replace('|', '_').replace(':', '-').replace(' ', '_')}__{i}.wav", y, sr)
        sims = [self.sim(y, sr, self.target) for y, sr in outs]
        gap = float(np.mean([s - self.sim(y, sr, self.base_emb(base_speaker)) for s, (y, sr) in zip(sims, outs)]))
        cer = float(np.mean([wer_cer(t, transcribe(y, sr, "ru"))[1] for t, (y, sr) in zip(self.texts, outs)]))
        lt = np.mean([ltas(y, sr) for y, sr in outs], axis=0)
        f0 = [s for s in (f0_stats(y, sr) for y, sr in outs) if s[0]]
        med, rng = float(np.median([s[0] for s in f0])), float(np.median([s[1] for s in f0]))
        r = {"name": name, "sim": float(np.mean(sims)), "gap": gap, "cer": cer, "ltas": ltas_distance(lt, self.ltas, self.mask),
             "f0_med": med, "f0_range": rng, "f0": abs(12 * np.log2(med / self.f0_med)) + abs(rng - self.f0_range),
             "ltas_curve": lt.tolist(), **meta}
        self.rows.append(r)
        print(f"  {name:56s} SIM {r['sim']:.3f} gap {gap:+.3f} LTAS {r['ltas']:4.1f} dB F0 {med:4.0f} Hz/{rng:.1f} st "
              f"CER {cer:4.1f}")
        return r


def rank_pick(rows):
    pool = [r for r in rows if r["cer"] <= 5] or rows
    # Absolute SIM is ranked too: "gap" alone favours base speakers that start far from the character.
    for key, sign in (("sim", -1), ("gap", -1), ("ltas", 1), ("f0", 1)):
        for i, r in enumerate(sorted(pool, key=lambda r: sign * r[key])):
            r.setdefault("ranks", {})[key] = i
    # Ties (common with two candidates) go to the lower CER, then the higher SIM.
    return min(pool, key=lambda r: (sum(r["ranks"].values()), round(r["cer"], 1), -r["sim"]))


def eq_helps(before, after):
    return after["cer"] <= max(5.0, before["cer"] + 1.0) and after["ltas"] < before["ltas"] - 0.5 \
        and after["gap"] >= before["gap"] - 0.02


def tune(key, m, refs, E, sim):
    prof = PROFILES[key]
    silero = E.get("silero")
    J = Judge(sim, key, refs[key], silero)
    print(f"\n===== {key}: real F0 {J.f0_med:.0f} Hz, range {J.f0_range:.1f} st, {len(refs[key]['sim_set'])} real lines")
    gender = "female" if key == "glados" else "male"
    chosen = {"eq": {}}

    # ---- Silero -> RVC
    rvc = E.get("rvc")
    per_variant = {}
    for entry in m["rvc"].get(key, []):
        rvc.load(entry["pth"], entry.get("index"))
        rows = []
        for spk in SPEAKERS[gender]:
            spk_f0 = np.median([f0_stats(J.base(spk, t), silero.sr)[0] for t in J.texts[:4]])
            center = int(round(12 * np.log2(J.f0_med / spk_f0)))
            for pitch in sorted({center - 2, center, center + 2}):
                kw = dict(pitch=pitch, index_rate=prof.rvc_index_rate, protect=prof.rvc_protect)
                rows.append(J.evaluate(f"rvc:{entry['variant']}|{spk}|p{pitch:+d}",
                                       lambda t: (rvc(J.base(spk, t), silero.sr, **kw), rvc.sr), spk,
                                       variant=f"rvc:{entry['variant']}", speaker=spk, pitch=pitch))
        best = rank_pick(rows)
        gains = matched_eq_gains(J.ltas, np.array(best["ltas_curve"]), mask=J.mask)
        kw = dict(pitch=best["pitch"], index_rate=prof.rvc_index_rate, protect=prof.rvc_protect)
        spk = best["speaker"]
        eqd = J.evaluate(best["name"] + "|eq", lambda t: (matched_eq(rvc(J.base(spk, t), silero.sr, **kw), rvc.sr, gains),
                                                          rvc.sr), spk, variant=best["variant"], speaker=spk,
                         pitch=best["pitch"])
        use_eq = eq_helps(best, eqd)
        per_variant[best["variant"]] = (eqd if use_eq else best, gains.tolist() if use_eq else None)
    if per_variant:
        # One base speaker + pitch for the voice: taken from the best RVC variant overall.
        top = rank_pick([r for r, _ in per_variant.values()])
        chosen.update(silero_speaker=top["speaker"], rvc_pitch=top["pitch"], rvc_variant=top["variant"])
        for variant, (r, gains) in per_variant.items():
            if gains is not None:
                chosen["eq"][variant] = gains
    E.unload("rvc")

    # ---- Native
    variant_engine = {"tera": "tera", "piper:piper_wheatley": "piper_wheatley",
                      "piper:piper_cave_johnson": "piper_cave_johnson"}.get(prof.native)
    fx = dsp.FX[key]
    if variant_engine:
        eng = E.get(variant_engine)
        nat_name = f"native:{variant_engine}"
        spk0 = prof.silero_speaker
        rows = [J.evaluate(f"{nat_name}|dsp", lambda t: (fx(eng(numbers_to_words(t)), eng.sr), eng.sr), spk0,
                           native_fx={})]
        if key == "glados":  # the procedural GLaDOS DSP vs the bare TeraTTS voice
            rows.append(J.evaluate(f"{nat_name}|raw", lambda t: (eng(numbers_to_words(t)), eng.sr), spk0,
                                   native_fx={"bypass": True}))
        best = rank_pick(rows)
        nfx = best["native_fx"]
        render = (lambda t: eng(numbers_to_words(t))) if nfx.get("bypass") else \
            (lambda t: fx(eng(numbers_to_words(t)), eng.sr))
        gains = matched_eq_gains(J.ltas, np.array(best["ltas_curve"]), mask=J.mask)
        eqd = J.evaluate(best["name"] + "|eq", lambda t: (matched_eq(render(t), eng.sr, gains), eng.sr), spk0,
                         native_fx=nfx)
        chosen["native_fx"] = nfx
        if eq_helps(best, eqd):
            chosen["eq"][nat_name] = gains.tolist()
        E.unload(variant_engine)

    # ---- F5 (NFE 16 to keep tuning quick; the EQ is shared by both NFE settings)
    if key in refs:
        f5 = E.get("f5")
        r = refs[key]
        base = J.evaluate("f5", lambda t: (f5(numbers_to_words(t), r["wav"], r["text"], "ru", nfe_step=16), 24000),
                          prof.silero_speaker)
        gains = matched_eq_gains(J.ltas, np.array(base["ltas_curve"]), mask=J.mask)
        eqd = J.evaluate("f5|eq", lambda t: (matched_eq(f5(numbers_to_words(t), r["wav"], r["text"], "ru", nfe_step=16),
                                                        24000, gains), 24000), prof.silero_speaker)
        if eq_helps(base, eqd):
            chosen["eq"]["f5"] = gains.tolist()
        E.unload("f5")
    print(f"  -> chosen: { {k: v for k, v in chosen.items() if k != 'eq'} }  EQ on: {list(chosen['eq'])}")
    return chosen, J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="")
    args = ap.parse_args()
    m, refs = load_manifest(), load_references()
    voices = [v for v in args.voices.split(",") if v] or [k for k in PROFILES if k in refs]
    E = Engines(m)
    sim = SpeakerSim(m["engines"]["speaker_sv"])
    path = RESULTS / "voice_tuning.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"voices": {}, "trials": {}}
    for key in voices:
        chosen, J = tune(key, m, refs, E, sim)
        saved["voices"][key] = chosen
        saved["trials"][key] = {"real": {"f0_med": J.f0_med, "f0_range": J.f0_range},
                                "rows": [{k: v for k, v in r.items() if k != "ltas_curve"} for r in J.rows]}
        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
