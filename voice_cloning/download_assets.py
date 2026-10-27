"""Download every model, dataset and reference clip used by the voice arena.

Writes models/manifest.json, which the pipelines read to find local files.
Each source is checked independently: a missing repo or file is recorded in
the manifest's "status" list instead of aborting the whole run.

    .venv/Scripts/python download_assets.py            # everything
    .venv/Scripts/python download_assets.py --skip-f5  # without the 1.3 GB F5 checkpoint
"""
import argparse
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

from voice_arena.paths import APPLIO, DATASETS, HF_TOKEN, MANIFEST, MODELS

api = HfApi(token=HF_TOKEN)
status = []  # [{"what", "ok", "note"}]

# Keywords used to search large community packs for each character.
KEYWORDS = {
    "glados": ["glados", "глэдос", "гладос"],
    "wheatley": ["wheatley", "уитли"],
    "cave_johnson": ["cave", "кейв", "джонсон"],
    "pudge": ["pudge", "пудж"],
    "witch_doctor": ["witch", "witchdoctor", "доктор"],
    "arthas_paladin": ["arthas", "артас"],
    "arthas_deathknight": ["lich", "лич", "arthas", "артас"],
}


def log(what, ok, note=""):
    status.append({"what": what, "ok": ok, "note": note})
    print(("  OK  " if ok else "  --  ") + what + (f"  ({note})" if note else ""))


def dl(repo, filename, repo_type="model", subdir=None):
    """hf_hub_download into models/<repo>/ (or datasets/<repo>/) and return the local Path."""
    base = DATASETS if repo_type == "dataset" else MODELS
    local_dir = base / (subdir or repo.replace("/", "__"))
    path = hf_hub_download(repo, filename, repo_type=repo_type, local_dir=local_dir, token=HF_TOKEN)
    return Path(path)


def valid_rvc(pth: Path) -> bool:
    """An RVC inference checkpoint has 'weight' and 'config'; G_/D_ training dumps do not."""
    try:
        cpt = torch.load(pth, map_location="cpu", weights_only=True)
        return isinstance(cpt, dict) and "weight" in cpt and "config" in cpt
    except Exception:
        return False


def rvc_entry(variant, pth, index, source, lang, note=""):
    return {"variant": variant, "pth": str(pth), "index": str(index) if index else None,
            "source": source, "lang": lang, "note": note}


def pick_from_dir(d: Path):
    pths = sorted((p for p in d.rglob("*.pth") if valid_rvc(p)), key=lambda p: p.stat().st_size)
    idx = sorted(d.rglob("*.index"), key=lambda p: ("added" not in p.name, -p.stat().st_size))
    return (pths[-1] if pths else None), (idx[0] if idx else None)


def rvc_from_zip(repo, zip_name, variant, source_lang, note="", repo_type="model"):
    z = dl(repo, zip_name, repo_type=repo_type)
    out = MODELS / "rvc" / variant
    if not out.exists():
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():
                if m.filename.lower().endswith((".pth", ".index")):
                    # Many zips are packed on Windows with cp866 names; keep only the basename.
                    target = out / Path(m.filename).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    pth, index = pick_from_dir(out)
    if not pth:
        raise RuntimeError("no valid RVC inference .pth inside zip")
    return rvc_entry(variant, pth, index, f"{repo}/{zip_name}", source_lang, note)


def rvc_from_files(repo, pth_name, index_name, variant, source_lang, note=""):
    pth = dl(repo, pth_name)
    index = dl(repo, index_name) if index_name else None
    if not valid_rvc(pth):
        raise RuntimeError(f"{pth_name} is not an RVC inference checkpoint")
    return rvc_entry(variant, pth, index, f"{repo}/{pth_name}", source_lang, note)


def try_add(manifest_list, what, fn):
    try:
        entry = fn()
        manifest_list.append(entry)
        log(f"{what}: {entry['source']}", True, entry.get("note", ""))
    except Exception as e:
        log(what, False, f"{type(e).__name__}: {e}"[:200])


def search_pack(repo, prefix):
    """Find character .pth/.index files in a big community pack by keyword."""
    try:
        files = api.list_repo_files(repo)
    except Exception as e:
        log(f"list {repo}", False, str(e)[:120])
        return {}
    hits = {}
    for char, kws in KEYWORDS.items():
        for f in files:
            low = f.lower()
            if prefix in f and low.endswith((".pth", ".index")) and any(k in low for k in kws):
                # "Персонажи Англ/Геральд(Англ)/Geralt_Witcher.pth" matches "witch" -- skip the Witcher.
                if char == "witch_doctor" and "witcher" in low:
                    continue
                hits.setdefault(char, []).append(f)
    for char, fs in hits.items():
        print(f"  {repo}: {char} -> {fs}")
    return hits


def download_silero(manifest):
    out = MODELS / "silero"
    out.mkdir(exist_ok=True)
    # Silero is distributed from models.silero.ai (via torch.hub), not Hugging Face.
    for name in ("v5_6_ru", "v5_ru", "v4_ru"):
        target = out / f"{name}.pt"
        try:
            if not target.exists():
                urllib.request.urlretrieve(f"https://models.silero.ai/models/tts/ru/{name}.pt", target)
            manifest["engines"]["silero"] = {"path": str(target), "version": name}
            log(f"silero {name}", True)
            return
        except Exception as e:
            log(f"silero {name}", False, str(e)[:120])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-f5", action="store_true")
    args = ap.parse_args()

    manifest = {"engines": {}, "rvc": {c: [] for c in KEYWORDS}, "datasets": {}, "status": status}
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["engines"].update(old.get("engines", {}))

    print("== Base engines")
    if not args.skip_f5:
        try:
            ck = dl("Misha24-10/F5-TTS_RUSSIAN", "F5TTS_v1_Base_v2/model_last_inference.safetensors")
            vocab = dl("Misha24-10/F5-TTS_RUSSIAN", "F5TTS_v1_Base/vocab.txt")
            manifest["engines"]["f5"] = {"ckpt": str(ck), "vocab": str(vocab), "variant": "F5TTS_v1_Base_v2"}
            log("F5-TTS_RUSSIAN v2", True)
        except Exception as e:
            log("F5-TTS_RUSSIAN v2", False, str(e)[:200])
    download_silero(manifest)

    # RVC runtime (Applio layout): ContentVec embedder + RMVPE pitch predictor.
    try:
        for f, sub in (("Resources/embedders/contentvec/pytorch_model.bin", "embedders/contentvec"),
                       ("Resources/embedders/contentvec/config.json", "embedders/contentvec"),
                       ("Resources/predictors/rmvpe.pt", "predictors")):
            p = hf_hub_download("IAHispano/Applio", f, local_dir=MODELS / "applio", token=HF_TOKEN)
            target = APPLIO / "rvc" / "models" / sub / Path(f).name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(p, target)
        log("RVC runtime: contentvec + rmvpe", True)
    except Exception as e:
        log("RVC runtime: contentvec + rmvpe", False, str(e)[:200])

    # Speaker-similarity model for the benchmark.
    try:
        from huggingface_hub import snapshot_download
        p = snapshot_download("microsoft/wavlm-base-plus-sv", local_dir=MODELS / "wavlm-base-plus-sv", token=HF_TOKEN)
        manifest["engines"]["speaker_sv"] = p
        log("microsoft/wavlm-base-plus-sv", True)
    except Exception as e:
        log("microsoft/wavlm-base-plus-sv", False, str(e)[:200])

    print("== Portal 2")
    # Native Russian GLaDOS VITS (TeraTTS, ONNX, 22.05 kHz).
    try:
        files = {f: str(dl("TeraTTS/glados2-g2p-vits", f"exported/{f}")) for f in ("model.onnx", "config.json", "dictionary.txt")}
        manifest["engines"]["tera_glados"] = files
        log("TeraTTS/glados2-g2p-vits (native ru VITS)", True)
    except Exception as e:
        log("TeraTTS/glados2-g2p-vits", False, str(e)[:200])

    # Random118 Russian Style-Bert-VITS2 pilot: downloadable, but its Russian
    # text frontend (orange-RAG glados_ru) is not published, so it is not runnable here.
    try:
        for f in ("README.md", "config.json", "latest.safetensors", "style_vectors.npy"):
            dl("Random118/GLaDOS_TTS", f"Models/Russian_StyleBert_Pilot/v3-1000/{f}")
        log("Random118/GLaDOS_TTS Russian_StyleBert_Pilot v3-1000", True,
            "downloaded; needs unpublished orange-RAG Russian frontend -> not benchmarked")
    except Exception as e:
        log("Random118/GLaDOS_TTS Russian pilot", False, str(e)[:200])

    # English Piper voices (en-us espeak); used cross-lingually with Russian phonemes.
    for repo, name, key in (("davet2001/wheatley1", "wheatley1", "piper_wheatley"),
                            ("davet2001/cave_johnson1", "cave_johnson1", "piper_cave_johnson")):
        try:
            manifest["engines"][key] = {"onnx": str(dl(repo, f"{name}.onnx")), "json": str(dl(repo, f"{name}.onnx.json")),
                                        "lang": "en-us"}
            log(f"{repo} (Piper, en-us)", True, "English-only voice")
        except Exception as e:
            log(repo, False, str(e)[:200])

    # Russian game-dub datasets (LJSpeech layout) for zero-shot references.
    for char, repo in (("glados", "RoxasYTB/glados_p2_ru-ljspeech"), ("wheatley", "RoxasYTB/wheatley_ru-ljspeech"),
                       ("cave_johnson", "RoxasYTB/cavejohnson_ru-ljspeech")):
        try:
            z = dl(repo, "data.zip", repo_type="dataset")
            out = DATASETS / f"{char}_ru"
            if not out.exists():
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(out)
            manifest["datasets"][char] = str(out)
            log(repo, True)
        except Exception as e:
            log(repo, False, str(e)[:200])

    # More Russian GLaDOS (Portal 1 + extra lines): voice-similarity target set for tuning.
    manifest["glados_extra"] = []
    for name, repo in (("glados_p1_ru", "RoxasYTB/glados_p1_ru-ljspeech"), ("glados_other_ru", "RoxasYTB/glados_other_ru-ljspeech")):
        try:
            z = dl(repo, "data.zip", repo_type="dataset")
            out = DATASETS / name
            if not out.exists():
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(out)
            manifest["glados_extra"].append(str(out))
            log(repo, True)
        except Exception as e:
            log(repo, False, str(e)[:200])

    print("== Dota 2 Russian voice lines (game dub)")
    manifest["ru_clips"] = {}
    # Pudge: 251 Russian lines (Pud_*_ru.flac) packed as a kNN-VC matching set.
    try:
        tar = dl("DaniilMiskevich/knn-vc-wavlm-pudge", "dataset.tar.gz", repo_type="dataset")
        out = DATASETS / "pudge_ru"
        if not out.exists():
            out.mkdir(parents=True)
            with tarfile.open(tar) as tf:
                for m in tf.getmembers():
                    if m.isfile() and m.name.startswith("./wavs/") and m.name.endswith(".flac"):
                        (out / Path(m.name).name).write_bytes(tf.extractfile(m).read())
        manifest["ru_clips"]["pudge"] = str(out)
        log("DaniilMiskevich/knn-vc-wavlm-pudge (Russian Pudge VO)", True, f"{len(list(out.glob('*.flac')))} lines")
    except Exception as e:
        log("DaniilMiskevich/knn-vc-wavlm-pudge", False, str(e)[:200])
    # Witch Doctor: Russian lines (Wdoc_*_ru) from Dubina/dota2-wd.
    try:
        out = DATASETS / "witch_doctor_ru"
        out.mkdir(parents=True, exist_ok=True)
        for f in api.list_repo_files("Dubina/dota2-wd", repo_type="dataset"):
            if f.startswith("wd dota2/") and f.endswith(".wav") and "(1)" not in f:
                p = dl("Dubina/dota2-wd", f, repo_type="dataset")
                shutil.copy2(p, out / Path(f).name.replace(".mp3.wav", ".wav"))
        manifest["ru_clips"]["witch_doctor"] = str(out)
        log("Dubina/dota2-wd (Russian Witch Doctor VO)", True, f"{len(list(out.glob('*.wav')))} lines")
    except Exception as e:
        log("Dubina/dota2-wd", False, str(e)[:200])

    print("== Community RVC packs")
    pack = "niobures/RVC-Models"
    hits = search_pack(pack, "rvc-big-ru-voice-pack/Персонажи/")
    for char, fs in hits.items():
        pths = [f for f in fs if f.endswith(".pth")]
        for p in pths:
            folder = p.rsplit("/", 1)[0]
            idx = next((f for f in fs if f.endswith(".index") and f.startswith(folder)), None)
            variant = f"niobures_{Path(p).stem}"
            try_add(manifest["rvc"][char], f"{char} RVC",
                    lambda p=p, idx=idx, v=variant: rvc_from_files(pack, p, idx, v, "ru", "rvc-big-ru-voice-pack"))

    print("== Dedicated character RVC models")
    jobs = [
        ("wheatley", lambda: rvc_from_files("bdwind/RVC_models", "wheatley/wheatley.pth", "wheatley/wheatley.index",
                                            "bdwind_wheatley", "ru", "bdwind pack of Russian voices")),
        ("cave_johnson", lambda: rvc_from_zip("DaunIDebil/CaveJohnson-RVCv2", "CaveJohnsonRUS.zip", "daunidebil_cave_rus", "ru",
                                              "zip named CaveJohnsonRUS")),
        ("cave_johnson", lambda: rvc_from_files("bdwind/RVC_models", "cavej/cavej.pth", "cavej/cavej.index",
                                                "bdwind_cavej", "ru", "bdwind pack of Russian voices")),
        ("pudge", lambda: rvc_from_files("bdwind/RVC_models", "pudge/pudge.pth", "pudge/pudge.index",
                                         "bdwind_pudge", "ru", "bdwind pack of Russian voices")),
        ("witch_doctor", lambda: rvc_from_zip("GoodGuideGreg/WitchDoctorRVC", "WitchDoctorRVC.zip", "goodguidegreg_witchdoctor",
                                              "unknown (likely en)")),
        ("witch_doctor", lambda: rvc_from_files("bdwind/RVC_models", "wd/wd.pth", "wd/wd.index",
                                                "bdwind_wd", "ru", "folder 'wd' in a Russian pack; unlabeled")),
        ("witch_doctor", lambda: rvc_from_zip("Dubina/dota2-wd", "wdpack.zip", "dubina_wd_ru", "ru",
                                              "trained next to the Russian Wdoc_*_ru lines", repo_type="dataset")),
        ("glados", lambda: rvc_from_zip("Bababoyich/GLaDOS", "Glados.zip", "bababoyich_glados", "unknown")),
        ("arthas_paladin", lambda: rvc_from_zip("Warlock700/Arthas-Paladin", "arthas-paladin-fp32-no_noise-normal.zip",
                                                "warlock700_arthas_paladin", "ru (author trains Russian dub voices)")),
        ("arthas_paladin", lambda: rvc_from_files("Bohdanio2408/RVC-READY-MODELS", "Arthas/Arthas_500e_13500s.pth",
                                                  "Arthas/Arthas.index", "bohdanio_arthas", "unknown",
                                                  "era (paladin/DK) not stated")),
        ("arthas_deathknight", lambda: rvc_from_zip("Warlock700/Arthas-Knight", "arthas-knight-fp32-no_noise_normal.zip",
                                                    "warlock700_arthas_knight", "ru (author trains Russian dub voices)",
                                                    "Reign of Chaos death knight")),
        ("arthas_deathknight", lambda: rvc_from_zip("Warlock700/Arthas-KnightTFT", "arthas-knightTFT-fp32-no_noise_normal.zip",
                                                    "warlock700_arthas_knight_tft", "ru (author trains Russian dub voices)",
                                                    "The Frozen Throne death knight")),
    ]
    for char, fn in jobs:
        try_add(manifest["rvc"][char], f"{char} RVC", fn)

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(s["ok"] for s in status)
    print(f"\nmanifest -> {MANIFEST}  ({ok}/{len(status)} sources OK)")
    for char, entries in manifest["rvc"].items():
        print(f"  {char:20s} RVC variants: {[e['variant'] for e in entries]}")


if __name__ == "__main__":
    main()
