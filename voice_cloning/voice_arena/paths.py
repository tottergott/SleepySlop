"""Project paths and .env loading shared by every script."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODELS = ROOT / "models"
DATASETS = ROOT / "datasets"
REFERENCES = ROOT / "references"
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "results"
APPLIO = ROOT / "third_party" / "Applio"
MANIFEST = MODELS / "manifest.json"

HF_TOKEN = os.environ.get("HF_TOKEN")

for d in (MODELS, DATASETS, REFERENCES, OUTPUTS, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

# imageio-ffmpeg ships a static ffmpeg under a versioned name; pydub looks for
# plain "ffmpeg" on PATH, so expose a copy as bin/ffmpeg.exe.
BIN = ROOT / "bin"
try:
    import shutil

    import imageio_ffmpeg

    BIN.mkdir(exist_ok=True)
    _ff = BIN / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not _ff.exists():
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), _ff)
    os.environ["PATH"] = str(BIN) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass
