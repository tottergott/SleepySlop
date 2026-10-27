"""CPU ONNX VITS voices: TeraTTS native Russian GLaDOS and Piper character voices.

The TeraTTS G2P below is adapted from Tera2Space/TeraTTS (MIT licence).
Piper voices here are English (en-us); we feed them Russian espeak phonemes,
so they speak Russian with the character's English accent.
"""
import json
import os
import re
import unicodedata

import numpy as np
import onnxruntime as ort

from voice_arena.text import add_stress, numbers_to_words


def _session(path: str) -> ort.InferenceSession:
    """CPU session with a bounded thread pool: all-core default stalls badly when a game or browser shares the CPU."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])

# ---------------------------------------------------------------- TeraTTS G2P
_SOFT = set("яёюиье")
_STARTSYL = set("#ъьаяоёуюэеиы-")
_OTHERS = {"#", "+", "-", "ь", "ъ"}
_SOFTHARD = {"б": "b", "в": "v", "г": "g", "д": "d", "з": "z", "к": "k", "л": "l", "м": "m", "н": "n",
             "п": "p", "р": "r", "с": "s", "т": "t", "ф": "f", "х": "h"}
_OTHER_CONS = {"ж": "zh", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "й": "j"}
_VOWELS = {"а": "a", "я": "a", "у": "u", "ю": "u", "о": "o", "ё": "o", "э": "e", "е": "e", "и": "i", "ы": "y"}


def _tera_word(stressword: str) -> list[str]:
    phones, stress = [], 0
    for ch in "#" + stressword + "#":
        if ch == "+":
            stress = 1
        else:
            phones.append((ch, stress))
            stress = 0
    for i, (ch, _) in enumerate(phones[:-1]):
        if ch in _SOFTHARD:
            phones[i] = (_SOFTHARD[ch] + ("j" if phones[i + 1][0] in _SOFT else ""), 0)
        elif ch in _OTHER_CONS:
            phones[i] = (_OTHER_CONS[ch], 0)
    out, prev = [], ""
    for ch, st in phones:
        if prev in _STARTSYL and ch in "яюеё":
            out.append("j")
        out.append(_VOWELS[ch] + str(st) if ch in _VOWELS else ch)
        prev = ch
    return [p for p in out if p not in _OTHERS]


class TeraVits:
    sr = 22050

    def __init__(self, onnx_path: str, config_path: str, dictionary_path: str | None = None):
        self.session = _session(onnx_path)
        self.id_map = json.load(open(config_path, encoding="utf-8"))["phoneme_id_map"]
        self.dic = {}
        if dictionary_path and os.path.exists(dictionary_path):
            with open(dictionary_path, encoding="utf-8") as f:
                for line in f:
                    items = line.split()
                    if items:
                        self.dic[items[0]] = items[1:]

    def _ids(self, text: str) -> list[int]:
        text = re.sub("([!'(),\\-.:;?])", r" \1 ", text.replace("—", "-"))
        phonemes = []
        for word in text.split():
            if re.fullmatch("[!'(),\\-.:;?]", word):
                phonemes.append(word)
                continue
            word = word.lower()
            if phonemes:
                phonemes.append(" ")
            # RUAccent's stress (homograph-aware) wins; the dictionary covers unstressed words.
            phonemes.extend(_tera_word(word) if "+" in word else (self.dic.get(word) or _tera_word(word)))
        ids = [*self.id_map["^"], *self.id_map["_"]]
        for p in phonemes:
            if p in self.id_map:
                ids += [*self.id_map[p], *self.id_map["_"]]
        ids += self.id_map["$"]
        # TeraTTS intersperses the blank (id 0) between every symbol.
        inter = [0] * (len(ids) * 2 + 1)
        inter[1::2] = ids
        return inter

    def __call__(self, text: str, length_scale: float = 1.1) -> np.ndarray:
        text = add_stress(numbers_to_words(text))
        ids = np.array([self._ids(text)], dtype=np.int64)
        audio = self.session.run(None, {
            "input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
            "scales": np.array([0.667, length_scale, 0.8], dtype=np.float32), "sid": None,
        })[0]
        return audio.reshape(-1).astype(np.float32)


# -------------------------------------------------------------- Piper (espeak)
_phonemizer = None


def _espeak_ru():
    global _phonemizer
    if _phonemizer is None:
        import espeakng_loader
        from phonemizer.backend import EspeakBackend
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        EspeakWrapper.set_library(espeakng_loader.get_library_path())
        os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()
        _phonemizer = EspeakBackend("ru", preserve_punctuation=True, with_stress=True)
    return _phonemizer


# Russian-only espeak symbols an en-us voice never heard -> nearest English sounds.
# Picked by Whisper CER on the benchmark phrases (Cave Johnson 54 % -> 32 %).
_RU_TO_EN_IPA = {"ʲ": "j", "ɭ": "l", "y": "ɪ", "ɨ": "ɪ", "ɕ": "ʃ", "ɵ": "o", "x": "h"}


class PiperVoice:
    def __init__(self, onnx_path: str, json_path: str):
        self.session = _session(onnx_path)
        cfg = json.load(open(json_path, encoding="utf-8"))
        self.sr = cfg["audio"]["sample_rate"]
        self.id_map = cfg["phoneme_id_map"]
        inf = cfg.get("inference", {})
        self.scales = [inf.get("noise_scale", 0.667), inf.get("length_scale", 1.0), inf.get("noise_w", 0.8)]

    def __call__(self, text: str) -> np.ndarray:
        ipa = _espeak_ru().phonemize([numbers_to_words(text)], strip=True)[0]
        for ru, en in _RU_TO_EN_IPA.items():
            ipa = ipa.replace(ru, en)
        ids = [*self.id_map["^"], *self.id_map["_"]]
        for ph in unicodedata.normalize("NFD", ipa):
            if ph in self.id_map:
                ids += [*self.id_map[ph], *self.id_map["_"]]
        ids += self.id_map["$"]
        x = np.array([ids], dtype=np.int64)
        feeds = {"input": x, "input_lengths": np.array([x.shape[1]], dtype=np.int64),
                 "scales": np.array(self.scales, dtype=np.float32)}
        if "sid" in {i.name for i in self.session.get_inputs()}:
            feeds["sid"] = np.array([0], dtype=np.int64)
        return self.session.run(None, feeds)[0].reshape(-1).astype(np.float32)
