"""Russian text front-end: numbers to words, stress marks, streaming chunks, WER normalisation."""
import re
from functools import lru_cache

from num2words import num2words

from voice_arena import paths  # noqa: F401  (loads HF_TOKEN from .env for RUAccent's hub downloads)

# Feminine nouns an assistant says after numbers: "42 миллисекунды" -> "сорок две".
_FEMININE = re.compile(r"\s+(милли|микро)?(секунд|минут|недел|ошиб|строк|верси|задач)", re.IGNORECASE)


def numbers_to_words(text: str) -> str:
    # Silero, Piper and TeraTTS cannot read digits; F5 reads them unreliably.
    def repl(m):
        gender = "f" if _FEMININE.match(text, m.end()) else "m"
        return num2words(int(m.group()), lang="ru", gender=gender)

    return re.sub(r"\d+", repl, text)


@lru_cache(maxsize=1)
def _accentizer():
    from ruaccent import RUAccent

    acc = RUAccent()
    acc.load(omograph_model_size="turbo2", use_dictionary=True, device="CPU")
    for model in vars(acc).values():
        if hasattr(model, "session"):
            _feed_token_type_ids(model.session)
    return acc


def _feed_token_type_ids(session):
    """RUAccent's ONNX graphs require token_type_ids, which transformers 5 tokenizers no longer return.

    Only out-of-dictionary words (e.g. "Лордерона") reach the neural accent model, so this
    fails late and only on some phrases.
    """
    if "token_type_ids" not in {i.name for i in session.get_inputs()}:
        return
    run = session.run

    def run_with_types(output_names, feeds, *args, **kwargs):
        if "token_type_ids" not in feeds:
            import numpy as np

            feeds = {**feeds, "token_type_ids": np.zeros_like(feeds["input_ids"])}
        return run(output_names, feeds, *args, **kwargs)

    session.run = run_with_types


@lru_cache(maxsize=512)
def add_stress(text: str) -> str:
    """'молоко' -> 'молок+о' (RUAccent), the format F5-TTS_RUSSIAN and the TeraTTS G2P expect."""
    return _accentizer().process_all(text)


def split_chunks(text: str, max_len: int = 60) -> list[str]:
    """Split into sentences, then split long sentences at commas.

    Each chunk is synthesised and emitted separately, so TTFA is the time to
    the first chunk -- that is how a streaming assistant would play it.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s.strip()]
    chunks = []
    for s in sentences:
        if len(s) <= max_len or "," not in s:
            chunks.append(s)
            continue
        parts = [p.strip() for p in re.split(r"(?<=,)\s+", s)]
        buf = ""
        for p in parts:
            if buf and len(buf) + len(p) > max_len:
                chunks.append(buf)
                buf = p
            else:
                buf = f"{buf} {p}".strip()
        if buf:
            chunks.append(buf)
    return chunks


def merge_short(chunks: list[str], min_len: int = 25) -> list[str]:
    """Glue chunks shorter than min_len onto the next one.

    F5 sizes its output from the reference's speaking rate; very short chunks
    ("Слушаю тебя.") come out clipped. Merging cut its CER 3-6x on the benchmark.
    """
    out = []
    for c in chunks:
        if out and len(out[-1]) < min_len:
            out[-1] = f"{out[-1]} {c}"
        else:
            out.append(c)
    return out


def normalize_for_wer(text: str) -> str:
    text = text.lower().replace("ё", "е").replace("+", "")
    # Whisper writes "42 мс"; expand before number words so gender agreement matches the reference.
    text = re.sub(r"(?<=\d)\s*мс\b", " миллисекунды", text)
    text = numbers_to_words(text)
    text = re.sub(r"[^\w\s]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()
