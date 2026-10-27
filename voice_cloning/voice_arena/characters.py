"""The seven voice profiles and how each synthesis mode realises them."""
from dataclasses import dataclass, field


@dataclass
class Profile:
    key: str
    title: str
    game: str
    silero_speaker: str          # base voice for the cascade and the DSP-only route
    rvc_pitch: int = 0           # semitones applied inside RVC (Silero base -> character range)
    rvc_index_rate: float = 0.6  # how much retrieved character features replace the base voice's
    rvc_protect: float = 0.33    # keeps consonants/breaths from the base voice (0.5 = off)
    rvc_autotune: float = 0.0    # snap F0 to semitones (0 = off, 1 = hard), the GLaDOS "synthetic" contour
    rvc_variant: str = ""        # preferred RVC model when several exist (benchmark still runs all)
    native_fx: dict = field(default_factory=dict)  # kwargs for the character DSP in Native mode; {"bypass": True} = raw
    eq: dict = field(default_factory=dict)  # {"rvc"|"native"|"f5": band gains (dB)} matched EQ to the real voice
    native: str = "silero_dsp"   # "tera" | "piper:<engine key>" | "silero_dsp"
    post_fx: bool = False        # apply the character DSP in every mode, not only in Native/DSP
    lore: str = ""               # phrase 4
    color: str = "#888"
    notes: list[str] = field(default_factory=list)


PROFILES = {p.key: p for p in [
    Profile("glados", "GLaDOS", "Portal 2", "xenia", rvc_pitch=0, native="tera", color="#f5a623",
            lore="Торт будет выдан после завершения всех тестов. Это обещание. Это ложь."),
    Profile("wheatley", "Уитли", "Portal 2", "aidar", rvc_pitch=2, native="piper:piper_wheatley", color="#4aa3ff",
            lore="Так, спокойно, без паники. Я абсолютно точно знаю, какую кнопку нажимать. Ну, почти."),
    Profile("cave_johnson", "Кейв Джонсон", "Portal 2", "aidar", rvc_pitch=-1, native="piper:piper_cave_johnson",
            color="#c8a96e",
            lore="Если жизнь подкидывает вам лимоны, не делайте лимонад. Сожгите ей дом лимонами!"),
    Profile("pudge", "Пудж", "Dota 2", "eugene", rvc_pitch=-2, color="#7fae5a",
            lore="Свежее мясо! Иди сюда, я тебя на крюк насажу."),
    Profile("witch_doctor", "Witch Doctor", "Dota 2", "eugene", rvc_pitch=-1, color="#b46cd8",
            lore="Духи вуду говорят мне: этот баг надо лечить магией и маленькой куклой."),
    Profile("arthas_paladin", "Артас: Паладин", "Warcraft III", "aidar", rvc_pitch=0, color="#e8c547",
            lore="Я принц Лордерона, и я защищу свой народ любой ценой. Свет со мной!"),
    Profile("arthas_deathknight", "Артас: Рыцарь Смерти", "Warcraft III", "eugene", rvc_pitch=-1, post_fx=True,
            color="#7fd3ff",
            lore="Ледяная Скорбь жаждет душ. Склонитесь перед Королём-личом, или умрите."),
]}

def _apply_tuning():
    """Overlay results/voice_tuning.json (written by tune_voices.py) onto the hand-set profiles."""
    import json

    from voice_arena.paths import RESULTS

    path = RESULTS / "voice_tuning.json"
    if not path.exists():
        return
    for key, tuned in json.loads(path.read_text(encoding="utf-8")).get("voices", {}).items():
        prof = PROFILES.get(key)
        if not prof:
            continue
        for attr in ("silero_speaker", "rvc_pitch", "rvc_variant", "native_fx"):
            if attr in tuned:
                setattr(prof, attr, tuned[attr])
        prof.eq = {**prof.eq, **tuned.get("eq", {})}
        prof.notes.append("tuned by tune_voices.py")


_apply_tuning()

PHRASES = [
    ("ping", "Слушаю тебя. Чего опять сломалось?"),
    ("emotion", "Отличная идея! Давай удалим системную папку и посмотрим, что будет дальше!"),
    ("digits", "Сервер запущен на порту 8080, задержка составляет 42 миллисекунды."),
    ("lore", None),  # per-character, Profile.lore
]


def phrase_text(phrase_id: str, profile: Profile) -> str:
    return profile.lore if phrase_id == "lore" else dict(PHRASES)[phrase_id]
