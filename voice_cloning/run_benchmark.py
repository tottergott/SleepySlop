"""Benchmark 7 voices x 3 architectures x 4 phrases on this machine.

Phase 1 (generation) runs one architecture at a time with only its engines in
memory, so Peak VRAM reflects that architecture alone. Every (voice, variant)
gets one unmeasured warm-up run first; each phrase is then measured --repeats
times and the median is reported (other apps may share the GPU). Chunks are
streamed sentence by sentence: TTFA = time until the first chunk is ready to play.

Phase 2 (quality) loads faster-whisper small (ru) and WavLM-SV and scores every
clip: WER/CER against the input text, speaker similarity against the reference.

    .venv/Scripts/python run_benchmark.py [--voices glados,pudge] [--modes f5,rvc,native]
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from voice_arena.characters import PHRASES, PROFILES, phrase_text
from voice_arena.paths import OUTPUTS, RESULTS, ROOT
from voice_arena.pipelines import Engines, Synthesizer, join_chunks, load_manifest, load_references, variants_for

WARMUP_TEXT = "Проверка связи, один, два."


def hardware() -> dict:
    import psutil

    info = {"cpu": platform.processor(), "cpu_threads": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
            "torch": torch.__version__, "cuda": torch.version.cuda, "os": platform.platform()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gb=round(p.total_memory / 2**30, 1))
    try:
        import cpuinfo  # optional
        info["cpu"] = cpuinfo.get_cpu_info()["brand_raw"]
    except Exception:
        pass
    return info


def gpu_background() -> dict:
    """GPU load from other apps (games, browsers) before we load anything -- it skews timings."""
    import subprocess

    try:
        util, mem = [], []
        for _ in range(5):
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, check=True).stdout.strip().split(",")
            util.append(float(out[0]))
            mem.append(float(out[1]))
            time.sleep(0.4)
        return {"gpu_util_pct": round(float(np.median(util))), "gpu_mem_used_mb": round(float(np.median(mem)))}
    except Exception:
        return {}


def run_one(synth, prof, v, text):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    ttfa, chunks, sr = None, [], None
    for y, sr in synth.stream(prof, v, text):
        if ttfa is None:
            ttfa = time.perf_counter() - t0
        chunks.append(y)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = time.perf_counter() - t0
    vram = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    audio = join_chunks(chunks, sr)
    return audio, sr, ttfa * 1000, total * 1000, vram, len(chunks)


def generate(voices, modes, repeats=3):
    manifest, refs = load_manifest(), load_references()
    records = []
    for mode in modes:
        engines = Engines(manifest)
        synth = Synthesizer(engines)
        print(f"\n===== mode: {mode}")
        for key in voices:
            prof = PROFILES[key]
            for v in [v for v in variants_for(prof, manifest, refs) if v.mode == mode]:
                try:
                    switch_ms = synth.prepare(prof, v)
                    run_one(synth, prof, v, WARMUP_TEXT)  # warm-up, not recorded
                except Exception as e:
                    print(f"  !! {key} {v.name}: {type(e).__name__}: {e}")
                    records.append({"voice": key, "variant": v.name, "mode": mode, "label": v.label,
                                    "error": f"{type(e).__name__}: {e}"[:300]})
                    continue
                for pid, _ in PHRASES:
                    text = phrase_text(pid, prof)
                    try:
                        # Median of `repeats` runs: robust to spikes from other GPU users.
                        runs = [run_one(synth, prof, v, text) for _ in range(repeats)]
                        ttfas, totals = [r[2] for r in runs], [r[3] for r in runs]
                        audio, sr, _, _, _, n = runs[int(np.argsort(totals)[len(totals) // 2])]
                        ttfa, total = float(np.median(ttfas)), float(np.median(totals))
                        vram = max(r[4] for r in runs)
                    except Exception as e:
                        print(f"  !! {key} {v.name} {pid}: {type(e).__name__}: {e}")
                        records.append({"voice": key, "variant": v.name, "mode": mode, "label": v.label, "phrase": pid,
                                        "text": text, "error": f"{type(e).__name__}: {e}"[:300]})
                        continue
                    out = OUTPUTS / key / v.name.replace(":", "__") / f"{pid}.wav"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(out, audio, sr, subtype="PCM_16")
                    dur = len(audio) / sr
                    rec = {"voice": key, "variant": v.name, "mode": mode, "label": v.label, "phrase": pid, "text": text,
                           "ttfa_ms": round(ttfa, 1), "total_ms": round(total, 1), "audio_s": round(dur, 2),
                           "rtf": round(total / 1000 / dur, 3) if dur else None, "vram_mb": round(vram, 0),
                           "switch_ms": round(switch_ms, 1), "chunks": n, "sr": sr, "repeats": repeats,
                           "ttfa_runs_ms": [round(x, 1) for x in ttfas], "total_runs_ms": [round(x, 1) for x in totals],
                           "wav": out.relative_to(ROOT).as_posix(), "source": v.info.get("source") or v.info.get("engine"),
                           "source_lang": v.info.get("lang"), "note": v.info.get("note", "")}
                    records.append(rec)
                    print(f"  {key:19s} {v.name:38s} {pid:8s} TTFA {ttfa:7.0f} ms  total {total:7.0f} ms  "
                          f"RTF {rec['rtf']:.3f}  VRAM {vram:6.0f} MB")
        engines.unload()
    return records


def score(records):
    from voice_arena.metrics import SpeakerSim, transcribe, wer_cer

    manifest, refs = load_manifest(), load_references()
    sim = SpeakerSim(manifest["engines"]["speaker_sv"])
    ref_emb = {k: sim.centroid(r.get("sim_set") or [r["wav"]]) for k, r in refs.items()}
    print("\n===== quality pass (faster-whisper small, ru; WavLM-SV)")
    for rec in records:
        if "error" in rec:
            continue
        y, sr = sf.read(ROOT / rec["wav"], dtype="float32")
        hyp = transcribe(y, sr, "ru")
        rec["asr"] = hyp
        rec["wer"], rec["cer"] = (round(x, 1) for x in wer_cer(rec["text"], hyp))
        rec["sim"] = round(sim(y, sr, ref_emb[rec["voice"]]), 3) if rec["voice"] in ref_emb else None
        print(f"  {rec['voice']:19s} {rec['variant']:38s} {rec['phrase']:8s} WER {rec['wer']:5.1f}  CER {rec['cer']:5.1f}  "
              f"SIM {rec['sim']}  | {hyp[:60]}")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default=",".join(PROFILES))
    ap.add_argument("--modes", default="native,rvc,f5")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--repeats", type=int, default=3, help="measured runs per phrase (median reported)")
    ap.add_argument("--score-only", action="store_true", help="re-score the saved clips without regenerating them")
    args = ap.parse_args()
    voices = [v for v in args.voices.split(",") if v]
    modes = [m for m in args.modes.split(",") if m]

    path = RESULTS / "benchmark.json"
    if args.score_only:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["records"] = score(data["records"])
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nre-scored -> {path}")
        return

    background = gpu_background()
    print("GPU background before start:", background)
    records = generate(voices, modes, args.repeats)
    if not args.no_score:
        records = score(records)
    out = {"hardware": {**hardware(), **background}, "generated": time.strftime("%Y-%m-%d %H:%M"), "references": load_references(),
           "manifest_status": load_manifest()["status"], "records": records}
    if path.exists() and (set(voices) != set(PROFILES) or set(modes) != {"native", "rvc", "f5"}):
        # Partial run: merge into the previous results instead of overwriting them.
        old = json.loads(path.read_text(encoding="utf-8"))
        keep = [r for r in old["records"] if not (r["voice"] in voices and r["mode"] in modes)]
        out["records"] = keep + records
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nresults -> {path}")


if __name__ == "__main__":
    main()
