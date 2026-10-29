"""Unified realistic channel degradation and noise simulator for LEO satellite signals."""

from typing import Tuple, Dict, Any, Optional
import numpy as np

from src.noise.awgn import add_awgn
from src.noise.impulsive import add_impulsive_noise
from src.noise.burst import add_burst_interference
from src.noise.narrowband import add_narrowband_interference
from src.noise.fading import add_amplitude_fading
from src.noise.frequency_offset import apply_frequency_offset
from src.noise.clipping import apply_clipping


def apply_degradation(
    clean_signal: np.ndarray,
    sr: int,
    noise_type: str = "awgn",
    snr_db: float = 10.0,
    seed: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply realistic synthetic or combined degradation to a clean satellite signal segment.

    $$y(t) = \text{degradation}(x(t), \theta)$$

    Args:
        clean_signal: Clean reference signal numpy array.
        sr: Sample rate in Hz.
        noise_type: Type of degradation ('awgn', 'impulsive', 'burst', 'narrowband',
                    'fading', 'frequency_offset', 'clipping', 'combined').
        snr_db: Target SNR for additive Gaussian component.
        seed: Random seed for complete reproducibility.
        config: Optional parameter override dictionary.

    Returns:
        (noisy_signal, metadata_dict)
    """
    cfg = config or {}
    if "snr_db" in cfg:
        snr_db = float(cfg["snr_db"])
    rng = np.random.default_rng(seed)
    signal = clean_signal.copy()
    meta: Dict[str, Any] = {
        "noise_category": "synthetic",
        "noise_type": noise_type,
        "nominal_snr_db": snr_db,
        "seed": seed,
        "stages": [],
    }

    if noise_type == "awgn":
        noisy, m = add_awgn(signal, snr_db=snr_db, seed=seed)
        meta["stages"].append(m)
        return noisy, meta

    elif noise_type == "impulsive":
        prob = cfg.get("probability", 0.005)
        amp = cfg.get("amplitude_factor", 1.5)
        noisy, m = add_impulsive_noise(signal, probability=prob, amplitude_factor=amp, seed=seed)
        meta["stages"].append(m)
        return noisy, meta

    elif noise_type == "burst":
        b_prob = cfg.get("burst_probability", 1.0)  # Always inject when explicitly selected
        b_snr = cfg.get("burst_snr_db", snr_db)
        noisy, m = add_burst_interference(signal, sr, burst_probability=b_prob, burst_snr_db=b_snr, seed=seed)
        meta["stages"].append(m)
        return noisy, meta

    elif noise_type == "narrowband":
        n_tones = cfg.get("num_tones", rng.integers(1, 3))
        noisy, m = add_narrowband_interference(signal, sr, num_tones=n_tones, seed=seed)
        meta["stages"].append(m)
        return noisy, meta

    elif noise_type == "fading":
        min_g = cfg.get("min_gain", 0.35)
        noisy, m = add_amplitude_fading(signal, sr, min_gain=min_g, seed=seed)
        meta["stages"].append(m)
        return noisy, meta

    elif noise_type == "frequency_offset":
        offset = cfg.get("offset_hz", float(rng.uniform(-10.0, 10.0)))
        noisy, m = apply_frequency_offset(signal, sr, offset_hz=offset)
        meta["stages"].append(m)
        return noisy, meta

    elif noise_type == "clipping":
        thresh = cfg.get("threshold", 0.80)
        mode = cfg.get("mode", "hard")
        noisy, m = apply_clipping(signal, threshold=thresh, mode=mode)
        meta["stages"].append(m)
        return noisy, meta

    elif noise_type == "combined":
        # Realistic composite channel: Fading + AWGN + Burst + Narrowband + Saturation
        current = signal

        # 1. Amplitude fading (Faraday rotation)
        if rng.random() < 0.6:
            current, m_fade = add_amplitude_fading(current, sr, min_gain=0.4, seed=rng.integers(0, 1000000))
            meta["stages"].append(m_fade)

        # 2. Narrowband interference tone
        if rng.random() < 0.4:
            current, m_nb = add_narrowband_interference(current, sr, num_tones=1, seed=rng.integers(0, 1000000))
            meta["stages"].append(m_nb)

        # 3. Burst interference
        if rng.random() < 0.3:
            current, m_burst = add_burst_interference(current, sr, burst_probability=1.0, burst_snr_db=snr_db - 3.0, seed=rng.integers(0, 1000000))
            meta["stages"].append(m_burst)

        # 4. Impulsive atmospheric clicks
        if rng.random() < 0.3:
            current, m_imp = add_impulsive_noise(current, probability=0.003, seed=rng.integers(0, 1000000))
            meta["stages"].append(m_imp)

        # 5. Core AWGN noise background
        current, m_awgn = add_awgn(current, snr_db=snr_db, seed=rng.integers(0, 1000000))
        meta["stages"].append(m_awgn)

        # 6. Occasional ADC saturation
        if rng.random() < 0.25:
            current, m_clip = apply_clipping(current, threshold=0.90, mode="soft")
            meta["stages"].append(m_clip)

        return current, meta

    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
