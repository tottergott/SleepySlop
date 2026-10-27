"""Procedural character effects (pedalboard + WORLD vocoder)."""
import numpy as np
import pyworld as pw
from scipy.signal import resample_poly
from pedalboard import (Chorus, Compressor, Distortion, Gain, HighpassFilter, HighShelfFilter, Limiter,
                        LowpassFilter, LowShelfFilter, PeakFilter, Pedalboard, Phaser, PitchShift, Reverb)


def _board(audio: np.ndarray, sr: int, plugins) -> np.ndarray:
    return Pedalboard(plugins)(audio.astype(np.float32)[None, :], sr)[0]


def _peak(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    m = float(np.abs(audio).max()) or 1.0
    return (audio * (target / m)).astype(np.float32)


def glados(audio, sr, quantize=0.7, formant=1.08):
    """Semitone-snapped pitch (sarcastic synthetic AI) + formant warp, then a thin phaser sheen."""
    x = audio.astype(np.float64)
    f0, t = pw.dio(x, sr, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    voiced = f0 > 0
    snapped = 440.0 * 2 ** (np.round(12 * np.log2(f0[voiced] / 440.0)) / 12)
    f0[voiced] = f0[voiced] * (snapped / f0[voiced]) ** quantize
    # Formant shift: read the spectral envelope at bin/formant so peaks move up by `formant`.
    bins = np.arange(sp.shape[1])
    src = np.clip(bins / formant, 0, sp.shape[1] - 1)
    sp = np.stack([np.interp(src, bins, frame) for frame in sp])
    y = pw.synthesize(f0, sp, ap, sr, frame_period=5.0).astype(np.float32)
    y = _board(y, sr, [HighpassFilter(110), Phaser(rate_hz=0.4, depth=0.25, mix=0.18),
                       Compressor(threshold_db=-20, ratio=3), Limiter(-1.0)])
    return _peak(y)


def pudge(audio, sr, base_voice=False):
    # Silero aidar ~118 Hz -> real Russian Pudge ~99 Hz (calibrate_pitch.py)
    pre = [PitchShift(-3)] if base_voice else []
    y = _board(audio, sr, pre + [LowShelfFilter(cutoff_frequency_hz=120, gain_db=6),
                                 Distortion(drive_db=6), Gain(-4), Compressor(threshold_db=-16, ratio=3),
                                 Limiter(-1.0)])
    return _peak(y)


def witch_doctor(audio, sr, base_voice=False):
    # Silero aidar ~118 Hz -> real Russian Witch Doctor ~162 Hz (calibrate_pitch.py)
    pre = [PitchShift(5)] if base_voice else []
    y = _board(audio, sr, pre + [PeakFilter(cutoff_frequency_hz=1100, gain_db=3, q=1.2),
                                 Distortion(drive_db=3), Gain(-2), Compressor(threshold_db=-18, ratio=2.5),
                                 Limiter(-1.0)])
    return _peak(y)


def arthas_paladin(audio, sr):
    """Clean noble voice: no pitch change, gentle warm compression."""
    y = _board(audio, sr, [HighpassFilter(70), LowShelfFilter(cutoff_frequency_hz=220, gain_db=2),
                           Compressor(threshold_db=-18, ratio=2.5, attack_ms=10, release_ms=120),
                           HighShelfFilter(cutoff_frequency_hz=7000, gain_db=1.5), Limiter(-1.0)])
    return _peak(y)


def arthas_deathknight(audio, sr, pre_shifted=False):
    """Frostmourne: -2 st voice, quiet -12 st sub-octave (15 %), cold metallic chorus, dark reverb.

    pre_shifted=True: the -2 st was already applied upstream (RVC renders the voice
    at that pitch for free), so only the sub-octave needs a pitch shifter.
    """
    x = audio.astype(np.float32)
    main = x if pre_shifted else _board(x, sr, [PitchShift(-2)])
    # The sub-octave is low-passed at 900 Hz, so shift it at 8 kHz: ~5x cheaper than at 40-48 kHz.
    low = resample_poly(x, 8000, sr).astype(np.float32)
    sub = _board(low, 8000, [PitchShift(-10 if pre_shifted else -12), LowpassFilter(900)])  # -12 st from the source
    sub = resample_poly(sub, sr, 8000).astype(np.float32)
    n = min(len(main), len(sub))
    y = 0.85 * main[:n] + 0.15 * sub[:n]
    y = _board(y, sr, [
        Chorus(rate_hz=0.6, depth=0.3, centre_delay_ms=8.0, feedback=0.3, mix=0.35),
        PeakFilter(cutoff_frequency_hz=3500, gain_db=2.5, q=2.0),  # icy metallic edge
        Reverb(room_size=0.8, damping=0.8, wet_level=0.22, dry_level=0.85, width=1.0),  # dark hall
        HighShelfFilter(cutoff_frequency_hz=9000, gain_db=-3),
        Compressor(threshold_db=-18, ratio=3), Limiter(-1.0),
    ])
    return _peak(y)


def cave_johnson(audio, sr):
    """Aperture PA / reel-to-reel recording colour."""
    y = _board(audio, sr, [HighpassFilter(180), LowpassFilter(6000), Distortion(drive_db=2), Gain(-1),
                           Compressor(threshold_db=-20, ratio=3), Limiter(-1.0)])
    return _peak(y)


def wheatley(audio, sr):
    y = _board(audio, sr, [HighpassFilter(90), Compressor(threshold_db=-20, ratio=2.5), Limiter(-1.0)])
    return _peak(y)


FX = {
    "glados": glados,
    "wheatley": wheatley,
    "cave_johnson": cave_johnson,
    "pudge": pudge,
    "witch_doctor": witch_doctor,
    "arthas_paladin": arthas_paladin,
    "arthas_deathknight": arthas_deathknight,
}
