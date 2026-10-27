"""Build ./references/<character>.wav: clean ~7 s, 24 kHz mono zero-shot references
from real Russian game voice lines only (no synthetic references).

* Portal 2 (RoxasYTB Russian dub datasets): the 6-8.5 s line whose Whisper
  transcript best matches its label (lowest CER) -- proof the clip is clean speech.
* Dota 2 (Russian dub lines: Pudge from DaniilMiskevich/knn-vc-wavlm-pudge,
  Witch Doctor from Dubina/dota2-wd): the clearest lines by Whisper confidence,
  joined with short gaps into a ~7 s reference; the 5 best are kept in references/raw/.
* Warcraft III (Arthas): no Russian Arthas audio is on Hugging Face. Put real game
  clips (any length/format) into references/user/arthas_paladin/ and
  references/user/arthas_deathknight/; long recordings are split at pauses.
  Without them the voice gets no reference (F5 is skipped), never a synthetic one.

Every character also gets a "sim_set": up to 40 real clips whose mean WavLM
embedding is the target for the speaker-similarity metric.
"""
import json
import random
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pedalboard import HighpassFilter, Pedalboard

from voice_arena.metrics import transcribe, wer_cer, whisper
from voice_arena.paths import REFERENCES
from voice_arena.pipelines import load_manifest
from voice_arena.voice_stats import f0_stats, ltas, ltas_distance

SR = 24000
TARGET_S = 7.0
AUDIO_EXT = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".opus")
USER_DIR = REFERENCES / "user"
# Grunts and screams make poor references (no words, extreme timbre).
GRUNTS = re.compile(r"_(attack|pain|laugh|death|deny|happy|anger|nomnom|lose)_", re.IGNORECASE)


def clean(audio: np.ndarray, sr: int) -> np.ndarray:
    y = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR) if sr != SR else audio.astype(np.float32)
    y, _ = librosa.effects.trim(y, top_db=35)
    y = Pedalboard([HighpassFilter(60)])(y[None, :], SR)[0]
    y = y / (np.abs(y).max() + 1e-9) * 0.89  # -1 dBFS
    pad = np.zeros(int(0.15 * SR), dtype=np.float32)
    return np.concatenate([pad, y, pad]).astype(np.float32)


def label_text(t: str) -> str:
    return re.sub(r"[«»\"]", "", t).strip()


def ljspeech_rows(root: Path):
    rows = []
    for line in (root / "metadata.csv").read_text(encoding="utf-8").splitlines():
        name, _, text = line.partition("|")
        text = label_text(text)
        wav = root / "wavs" / name
        if text and "[" not in text and "*" not in text and wav.exists():
            rows.append((wav, text))
    return rows


class Typical:
    """The character's usual voice (median F0, mean spectrum) over many real lines.

    Used to reject atypical lines -- PotatOS GLaDOS (F0 ~720 Hz instead of ~136 Hz),
    screams, whispers -- as references and from the similarity target set.
    """

    def __init__(self, paths, n=60):
        paths = list(paths)
        random.Random(0).shuffle(paths)
        clips = [librosa.load(p, sr=24000, mono=True) for p in paths[:n]]
        self.ltas = np.mean([ltas(y, sr) for y, sr in clips], axis=0)
        self.f0 = float(np.nanmedian([f0_stats(y, sr)[0] or np.nan for y, sr in clips]))

    def distance(self, y, sr) -> float:
        """LTAS distance (dB) + 1 dB per semitone of median-F0 deviation."""
        f0 = f0_stats(y, sr)[0]
        st = abs(12 * np.log2(f0 / self.f0)) if f0 else 12.0
        return ltas_distance(ltas(y, sr), self.ltas) + st

    def keep(self, path, max_dist=9.0) -> bool:
        return self.distance(*librosa.load(path, sr=24000, mono=True)) <= max_dist


def from_ljspeech(char: str, root: Path, typical: Typical, n_candidates: int = 30):
    rows = [(w, t, sf.info(w).duration) for w, t in ljspeech_rows(root)]
    pool = [r for r in rows if 6.0 <= r[2] <= 8.5] or sorted(rows, key=lambda r: abs(r[2] - TARGET_S))[:n_candidates]
    pool = sorted(pool, key=lambda r: abs(r[2] - TARGET_S))[:n_candidates]
    best = None
    for wav, text, dur in pool:
        y, sr = librosa.load(wav, sr=None, mono=True)
        _, cer = wer_cer(text, transcribe(y, sr, "ru"))
        typ = typical.distance(y, sr)
        score = cer + 2 * abs(dur - TARGET_S) + 1.5 * typ
        if best is None or score < best[0]:
            best = (score, wav, text, cer, typ, y, sr)
    _, wav, text, cer, typ, y, sr = best
    print(f"  {char}: {wav.name} '{text[:60]}...' CER={cer:.1f}% typicality {typ:.1f} (typical F0 {typical.f0:.0f} Hz)")
    return clean(y, sr), {"text": text, "lang": "ru", "source": f"{root.name}/{wav.name}", "real": True,
                          "whisper_cer": round(cer, 1), "typicality_db": round(typ, 1)}, [str(wav)]


def split_long(y: np.ndarray, sr: int, max_s: float = 9.0):
    """Cut a long user recording into utterances at pauses."""
    if len(y) / sr <= max_s:
        return [y]
    parts = []
    for a, b in librosa.effects.split(y, top_db=35, frame_length=2048, hop_length=512):
        seg = y[a:b]
        if 0.8 <= len(seg) / sr <= max_s:
            parts.append(seg)
    return parts


def from_clips(char: str, paths: list[Path], source: str, min_s: float = 1.0, max_s: float = 9.0,
               typical: "Typical | None" = None):
    """Pick the clearest, most typical real lines (Whisper confidence), join them into a ~7 s reference."""
    raw_dir = REFERENCES / "raw" / char
    raw_dir.mkdir(parents=True, exist_ok=True)
    for old in raw_dir.glob("*.wav"):
        old.unlink()
    scored = []
    for p in paths:
        y, sr = librosa.load(p, sr=None, mono=True)
        for k, seg in enumerate(split_long(y, sr, max_s)):
            dur = len(seg) / sr
            if not min_s <= dur <= max_s:
                continue
            segs, _ = whisper().transcribe(librosa.resample(seg, orig_sr=sr, target_sr=16000), language="ru",
                                           beam_size=5, condition_on_previous_text=False)
            segs = list(segs)
            text = " ".join(s.text.strip() for s in segs).strip()
            words = len(text.split())
            if words < 2:
                continue
            conf = float(np.mean([s.avg_logprob for s in segs]))
            nospeech = float(np.mean([s.no_speech_prob for s in segs]))
            if conf < -0.6 or nospeech > 0.5:  # unsure transcript -> wrong F5 ref_text
                continue
            typ = typical.distance(seg, sr) if typical else 0.0
            scored.append((conf + 0.04 * words + 0.05 * min(dur, 4) - 0.08 * typ,
                           f"{Path(p).stem}" + (f"#{k}" if k else ""), text, seg, sr))
    if not scored:
        raise RuntimeError(f"{char}: no clean Russian lines among {len(paths)} files")
    scored.sort(key=lambda s: -s[0])
    parts, texts, used = [], [], []
    for i, (_, name, text, y, sr) in enumerate(scored[:5], 1):
        c = clean(y, sr)
        sf.write(raw_dir / f"{i:02d}_{name}.wav", c, SR)
    for _, name, text, y, sr in scored:
        c = clean(y, sr)
        have = sum(len(x) for x in parts) / SR
        if have >= TARGET_S - 0.8:
            break
        if have + len(c) / SR <= 8.5:
            parts.append(c)
            texts.append(text)
            used.append(name)
    ref = np.concatenate(parts)
    print(f"  {char}: {len(scored)} clean lines, reference {len(ref) / SR:.1f}s from {used}")
    for t in texts:
        print(f"      «{t}»")
    return ref, {"text": " ".join(texts), "lang": "ru", "source": f"{source}: " + ", ".join(used), "real": True}


def user_files(char: str) -> list[Path]:
    d = USER_DIR / char
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXT) if d.exists() else []


def sample(paths, n=40, exclude=()):
    paths = [str(p) for p in paths if str(p) not in set(map(str, exclude))]
    random.Random(0).shuffle(paths)
    return paths[:n]


def main():
    manifest = load_manifest()
    for char in ("arthas_paladin", "arthas_deathknight"):
        (USER_DIR / char).mkdir(parents=True, exist_ok=True)
    refs = {}

    def typical_set(typical, pool, exclude=(), n=40):
        """Similarity target: up to n real lines, atypical ones (PotatOS, screams) removed."""
        cands = sample(pool, n=3 * n, exclude=exclude)
        return [p for p in cands if typical.keep(p)][:n]

    for char in ("glados", "wheatley", "cave_johnson"):
        if char in manifest["datasets"]:
            root = Path(manifest["datasets"][char])
            pool = [w for w, _ in ljspeech_rows(root)]
            if char == "glados":  # Portal 1 + extra lines widen the similarity target
                for extra in manifest.get("glados_extra", []):
                    pool += [w for w, _ in ljspeech_rows(Path(extra))]
            typical = Typical(pool)
            y, meta, used = from_ljspeech(char, root, typical)
            refs[char] = (y, meta, typical_set(typical, pool, exclude=used))

    clips = manifest.get("ru_clips", {})
    if clips.get("pudge"):
        files = [p for p in Path(clips["pudge"]).glob("*.flac") if not GRUNTS.search(p.name)]
        typical = Typical(files)
        y, meta = from_clips("pudge", files, "DaniilMiskevich/knn-vc-wavlm-pudge (Russian Dota 2 VO)", typical=typical)
        refs["pudge"] = (y, meta, typical_set(typical, files))
    if clips.get("witch_doctor"):
        files = sorted(p for p in Path(clips["witch_doctor"]).iterdir() if p.suffix.lower() in AUDIO_EXT)
        typical = Typical(files)
        y, meta = from_clips("witch_doctor", files, "Dubina/dota2-wd (Russian Dota 2 VO)", min_s=0.8, typical=typical)
        refs["witch_doctor"] = (y, meta, typical_set(typical, files))

    for char in ("arthas_paladin", "arthas_deathknight"):
        files = user_files(char)
        if not files:
            print(f"  {char}: no real clips in {USER_DIR / char} -> no reference (F5 skipped for this voice)")
            continue
        typical = Typical(files)
        y, meta = from_clips(char, files, "user-provided Warcraft III clips", typical=typical)
        refs[char] = (y, meta, typical_set(typical, files))

    out = {}
    for old in REFERENCES.glob("*.wav"):
        old.unlink()
    for char, (y, meta, sim_set) in refs.items():
        path = REFERENCES / f"{char}.wav"
        sf.write(path, y, SR, subtype="PCM_16")
        out[char] = {"wav": str(path), "duration_s": round(len(y) / SR, 2), **meta, "sim_set": sim_set}
    (REFERENCES / "references.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(out)} references -> {REFERENCES}")


if __name__ == "__main__":
    main()
