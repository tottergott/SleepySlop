# Voice Arena

Бенчмарк 7 игровых голосов на русском: GLaDOS, Уитли, Кейв Джонсон, Пудж, Witch Doctor,
Артас-паладин и Артас — Рыцарь Смерти. Сравниваются три архитектуры:

| Режим | Пайплайн |
|---|---|
| `f5` | F5-TTS_RUSSIAN (Misha24-10, v2), zero-shot по 7-секундному референсу, NFE 32 и NFE 16 |
| `rvc` | Silero v5 (`aidar`/`eugene`/`xenia`) → RVC v2 (ContentVec + RMVPE, код Applio) |
| `native` | TeraTTS GLaDOS (нативный русский VITS), английские Piper-голоса с русскими фонемами, Silero + DSP персонажа |

## Запуск

```bash
python -m uv venv .venv --python 3.10
python -m uv pip install --python .venv/Scripts/python.exe torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m uv pip install --python .venv/Scripts/python.exe -r requirements.txt -c constraints.txt
git clone --depth 1 https://github.com/IAHispano/Applio.git third_party/Applio
.venv/Scripts/python download_assets.py      # модели, датасеты, реплики → models/manifest.json
.venv/Scripts/python prepare_references.py   # references/<voice>.wav (24 кГц, моно, ~7 с) из реальных реплик
.venv/Scripts/python calibrate_pitch.py      # (справка) F0 настоящих голосов vs базовых голосов Silero
.venv/Scripts/python tune_voices.py          # подбор базы/тона/EQ под настоящий голос -> results/voice_tuning.json
.venv/Scripts/python run_benchmark.py        # outputs/*.wav + results/benchmark.json
.venv/Scripts/python build_report.py         # voice_arena.html (аудио встроено)
```

`HF_TOKEN` читается из `.env`.

## Роутер

```python
from voice_assistant_router import VoiceAssistantRouter

assistant = VoiceAssistantRouter(policy="balanced")   # speed | balanced | quality
assistant.set_voice("arthas_deathknight")
audio, sr = assistant.say("Ледяная Скорбь жаждет душ.")
```

## Что важно знать

- `rvc-python` не установлен намеренно: он требует `fairseq==0.12.2` и `omegaconf==2.0.6`,
  а это несовместимо с F5-TTS (`hydra-core>=1.3`). RVC выполняется кодом Applio,
  ContentVec загружается через `transformers`; FAISS-индекс кэшируется между вызовами.
- `davet2001/wheatley1` и `cave_johnson1` — английские Piper-модели (`en-us`). Им подаются
  русские фонемы espeak, частично переведённые в английские. Получается «акцент», разборчивость низкая.
- Все референсы — настоящие русские реплики из игр: Portal 2 (RoxasYTB), Пудж
  (`DaniilMiskevich/knn-vc-wavlm-pudge`, 251 реплика), Witch Doctor (`Dubina/dota2-wd`, 12 реплик).
  Нетипичные реплики (картофельная ГЛаДОС из Portal 2, крики) отсеиваются по F0 и спектру.
- Русских записей Артаса на Hugging Face нет, синтетических референсов больше нет. Положите
  реплики из игры (любой формат и длина) в `references/user/arthas_paladin/` и
  `references/user/arthas_deathknight/`, затем:
  `prepare_references.py`, `tune_voices.py --voices arthas_paladin,arthas_deathknight`,
  `run_benchmark.py --voices arthas_paladin,arthas_deathknight`, `build_report.py`.
- SIM считается к среднему голосу до 40 реальных реплик. Сам по себе он насыщается (~0,95 у всех
  хороших вариантов), поэтому `tune_voices.py` ранжирует ещё по разрыву SIM относительно базового
  голоса, по спектру (LTAS) и по F0. Подбор идёт на тех же 4 фразах плюс 2 дополнительных.
- `Random118/GLaDOS_TTS` Russian_StyleBert_Pilot скачан, но не запускается: его русский
  фронтенд (`orange-RAG/glados_ru`) не опубликован.
- Лицензии: F5-TTS_RUSSIAN — CC BY-NC 4.0; голоса персонажей принадлежат Valve и Blizzard.
  Только для личного некоммерческого использования.
