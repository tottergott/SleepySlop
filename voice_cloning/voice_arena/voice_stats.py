"""Objective "does it sound like the character" statistics beyond speaker embeddings.

* F0: median pitch (Hz) and intonation range (std in semitones) over voiced frames.
* LTAS: long-term average spectrum in 1/3-octave bands (dB, level-normalised); the
  distance to the real character's LTAS captures the game's processing/colouring.
* matched_eq: an FIR filter that moves a synth's LTAS onto the character's.
"""
import librosa
import numpy as np
import pyworld as pw
from scipy import signal

BANDS = np.array([100 * 2 ** (i / 3) for i in range(21)])  # 100 Hz .. ~10 kHz


def f0_stats(y: np.ndarray, sr: int):
    x = librosa.resample(y.astype(np.float64), orig_sr=sr, target_sr=16000) if sr != 16000 else y.astype(np.float64)
    f0, t = pw.dio(x, 16000, frame_period=10.0)
    f0 = pw.stonemask(x, f0, t, 16000)
    v = f0[f0 > 0]
    if len(v) < 10:
        return None, None
    st = 12 * np.log2(v)
    return float(np.median(v)), float(np.std(st))


def ltas(y: np.ndarray, sr: int) -> np.ndarray:
    x = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=24000) if sr != 24000 else y.astype(np.float32)
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=512)) ** 2
    energy = S.sum(0)
    active = energy > np.percentile(energy, 40)  # skip pauses
    p = S[:, active].mean(1) if active.any() else S.mean(1)
    freqs = librosa.fft_frequencies(sr=24000, n_fft=2048)
    lo, hi = BANDS / 2 ** (1 / 6), BANDS * 2 ** (1 / 6)
    bands = np.array([p[(freqs >= a) & (freqs < b)].mean() for a, b in zip(lo, hi)])
    db = 10 * np.log10(bands + 1e-12)
    return db - db.mean()


def band_mask(sr_min: int) -> np.ndarray:
    """Bands the real recordings actually contain: Pudge's clips are 16 kHz, so nothing above ~7.7 kHz."""
    return BANDS * 2 ** (1 / 6) < 0.48 * sr_min


def ltas_distance(a: np.ndarray, b: np.ndarray, mask=None) -> float:
    m = np.ones_like(a, dtype=bool) if mask is None else mask
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()  # re-normalise level over the compared bands
    return float(np.sqrt(np.mean((a - b) ** 2)))


def matched_eq_gains(target_db: np.ndarray, source_db: np.ndarray, limit_db: float = 9.0, mask=None) -> np.ndarray:
    m = np.ones_like(target_db, dtype=bool) if mask is None else mask
    t, s = target_db[m] - target_db[m].mean(), source_db[m] - source_db[m].mean()
    g = np.zeros_like(target_db)
    g[m] = np.clip(t - s, -limit_db, limit_db)  # flat where the real recordings have no information
    return np.convolve(np.pad(g, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")  # gentle smoothing


def matched_eq(y: np.ndarray, sr: int, gains_db, numtaps: int = 511) -> np.ndarray:
    """Linear-phase FIR through the band gains (flat outside 100 Hz-10 kHz)."""
    gains_db = np.asarray(gains_db, dtype=float)
    nyq = sr / 2
    f = np.concatenate([[0], BANDS[BANDS < nyq * 0.95], [nyq]])
    g = np.concatenate([[gains_db[0]], gains_db[: len(f) - 2], [gains_db[len(f) - 3]]])
    taps = signal.firwin2(numtaps, f / nyq, 10 ** (g / 20))
    out = signal.fftconvolve(y.astype(np.float32), taps.astype(np.float32), mode="same")
    peak = np.abs(out).max() or 1.0
    return (out * min(1.0, 0.95 / peak)).astype(np.float32)
