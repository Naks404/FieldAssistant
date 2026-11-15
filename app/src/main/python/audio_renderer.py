import json
import wave
import numpy as np


class AudioRenderer:
    def __init__(self):
        pass

    def _biquad_coeffs(self, filter_type: str, fc: float, q: float, gain_db: float, fs: int):
        """Exact RBJ Audio EQ Cookbook biquad coefficients matching JUCE/Pedalboard."""
        if abs(gain_db) < 1e-6:
            return (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)

        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * fc / float(fs)
        alpha = np.sin(w0) / (2.0 * max(float(q), 1e-4))
        cos_w0 = np.cos(w0)

        if filter_type == "PeakFilter":
            b0 = 1.0 + alpha * A
            b1 = -2.0 * cos_w0
            b2 = 1.0 - alpha * A
            a0 = 1.0 + alpha / A
            a1 = -2.0 * cos_w0
            a2 = 1.0 - alpha / A
        elif filter_type == "LowShelfFilter":
            sqrt_A = np.sqrt(A)
            b0 = A * ((A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha)
            b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha)
            a0 = (A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha
            a1 = -2.0 * ((A - 1.0) + (A + 1.0) * cos_w0)
            a2 = (A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha
        elif filter_type == "HighShelfFilter":
            sqrt_A = np.sqrt(A)
            b0 = A * ((A + 1.0) + (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha)
            b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cos_w0)
            b2 = A * ((A + 1.0) + (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha)
            a0 = (A + 1.0) - (A - 1.0) * cos_w0 + 2.0 * sqrt_A * alpha
            a1 = 2.0 * ((A - 1.0) - (A + 1.0) * cos_w0)
            a2 = (A + 1.0) - (A - 1.0) * cos_w0 - 2.0 * sqrt_A * alpha
        else:
            return (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)

        return (b0 / a0, b1 / a0, b2 / a0), (1.0, a1 / a0, a2 / a0)

    def _apply_envelope_dynamics(
        self,
        x: np.ndarray,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
        mode: str,
        makeup_db: float,
        fs: int
    ) -> np.ndarray:
        """Vectorized 1 kHz control-rate envelope follower for NoiseGate, Compressor, and Limiter."""
        if len(x) == 0:
            return x

        hop = max(1, int(fs * 0.001))  # 1 ms blocks (44 samples at 44.1 kHz)
        n_blocks = len(x) // hop
        if n_blocks < 2:
            return x * float(10.0 ** (makeup_db / 20.0))

        trimmed = x[:n_blocks * hop].reshape(n_blocks, hop)
        if mode == "limiter":
            env_lin = np.max(np.abs(trimmed), axis=1) + 1e-12
        else:
            env_lin = np.sqrt(np.mean(trimmed ** 2, axis=1) + 1e-12)
        env_db = 20.0 * np.log10(env_lin)

        fs_env = float(fs) / float(hop)
        alpha_a = float(np.exp(-1.0 / max(0.5, (attack_ms * 0.001) * fs_env)))
        alpha_r = float(np.exp(-1.0 / max(1.0, (release_ms * 0.001) * fs_env)))

        smooth_db = np.empty_like(env_db)
        curr = float(env_db[0])
        for i in range(n_blocks):
            val = float(env_db[i])
            if val > curr:
                curr = alpha_a * curr + (1.0 - alpha_a) * val
            else:
                curr = alpha_r * curr + (1.0 - alpha_r) * val
            smooth_db[i] = curr

        if mode == "gate":
            under = np.maximum(0.0, threshold_db - smooth_db)
            gain_db = -under * (max(ratio, 1.0) - 1.0)
            gain_db = np.maximum(gain_db, -80.0)
        elif mode == "compressor":
            over = np.maximum(0.0, smooth_db - threshold_db)
            gain_db = -over * (1.0 - 1.0 / max(ratio, 1.0)) + makeup_db
        elif mode == "limiter":
            over = np.maximum(0.0, smooth_db - threshold_db)
            gain_db = -over
        else:
            gain_db = np.zeros_like(smooth_db)

        gain_lin = 10.0 ** (gain_db / 20.0)
        idx_env = np.arange(n_blocks, dtype=np.float32) * hop + (hop * 0.5)
        gain_full = np.interp(np.arange(len(x), dtype=np.float32), idx_env, gain_lin).astype(np.float32)
        y = x * gain_full

        if mode == "limiter":
            ceiling_lin = float(10.0 ** (threshold_db / 20.0))
            y = np.clip(y, -ceiling_lin, ceiling_lin)

        return y

    def _apply_biquad_cascade_fft(self, x: np.ndarray, filter_list: list, fs: int) -> np.ndarray:
        """Evaluates cascaded biquad filters (EQ + De-esser) via exact frequency-domain transfer function."""
        if len(x) == 0 or not filter_list:
            return x

        n = len(x)
        X_spec = np.fft.rfft(x)
        w = 2.0 * np.pi * np.fft.rfftfreq(n, d=1.0)
        z_inv1 = np.exp(-1j * w)
        z_inv2 = z_inv1 * z_inv1

        H_total = np.ones_like(X_spec, dtype=np.complex128)
        active_any = False
        for ftype, fc, q, gain_db in filter_list:
            if abs(gain_db) < 1e-6:
                continue
            (b0, b1, b2), (a0, a1, a2) = self._biquad_coeffs(ftype, fc, q, gain_db, fs)
            num = b0 + b1 * z_inv1 + b2 * z_inv2
            den = a0 + a1 * z_inv1 + a2 * z_inv2
            H_total *= (num / den)
            active_any = True

        if not active_any:
            return x

        return np.fft.irfft(X_spec * H_total, n=n).astype(np.float32)

    def _apply_reverb(
        self,
        x: np.ndarray,
        room_size: float,
        damping: float,
        wet_level: float,
        dry_level: float,
        fs: int
    ) -> np.ndarray:
        """Applies damped room reverberation via fast FFT convolution."""
        if len(x) == 0 or wet_level <= 1e-5:
            return x

        rt60 = 0.15 + float(room_size) * 1.15
        ir_len = int(fs * min(rt60, 0.85))
        t = np.arange(ir_len, dtype=np.float32) / float(fs)
        decay = np.exp(-6.9078 * t / rt60)

        rng = np.random.RandomState(42)
        noise = rng.randn(ir_len).astype(np.float32)
        k_len = max(1, int(1 + float(damping) * 10))
        if k_len > 1:
            kernel = np.ones(k_len, dtype=np.float32) / float(k_len)
            noise = np.convolve(noise, kernel, mode="same")

        ir = noise * decay
        ir[0] = 1.0
        ir = ir / (np.sqrt(np.sum(ir ** 2)) + 1e-12) * 0.45

        n = len(x)
        n_fft = 1 << (n + ir_len - 1).bit_length()
        wet = np.fft.irfft(np.fft.rfft(x, n=n_fft) * np.fft.rfft(ir, n=n_fft), n=n_fft)[:n].astype(np.float32)
        return (float(dry_level) * x + float(wet_level) * wet).astype(np.float32)

    def render_preset_chain(
        self,
        input_wav_path: str,
        output_wav_path: str,
        json_str: str,
        denoise_idx: int,
        comp_idx: int,
        eq_idx: int,
        deesser_idx: int,
        reverb_idx: int,
        limiter_idx: int
    ) -> str:
        """Applies the 6-stage VoxEffects chain to input_wav_path and saves output_wav_path."""
        with wave.open(input_wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            fs = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        if sampwidth == 2:
            data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            data = np.frombuffer(raw_bytes, dtype=np.float32)

        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)

        if len(data) == 0:
            raise ValueError("Input WAV file is empty.")

        config = json.loads(json_str)
        y = data.copy()

        # 1. Denoise (NoiseGate)
        dn_preset = config["de_noise_presets"][int(denoise_idx)]
        if not dn_preset.get("bypass", True):
            for eff in dn_preset.get("effects", []):
                if eff["type"] == "NoiseGate":
                    p = eff["params"]
                    y = self._apply_envelope_dynamics(
                        y,
                        float(p["threshold_db"]),
                        float(p["ratio"]),
                        float(p["attack_ms"]),
                        float(p["release_ms"]),
                        mode="gate",
                        makeup_db=0.0,
                        fs=fs
                    )

        # 2. Compressor & Makeup Gain
        comp_preset = config["dynamic_presets"][int(comp_idx)]
        if not comp_preset.get("bypass", True):
            effects = comp_preset.get("effects", [])
            comp_params = None
            makeup_db = 0.0
            for eff in effects:
                if eff["type"] == "Compressor":
                    comp_params = eff["params"]
                elif eff["type"] == "Gain":
                    makeup_db += float(eff["params"].get("gain_db", 0.0))
            if comp_params is not None:
                y = self._apply_envelope_dynamics(
                    y,
                    float(comp_params["threshold_db"]),
                    float(comp_params["ratio"]),
                    float(comp_params["attack_ms"]),
                    float(comp_params["release_ms"]),
                    mode="compressor",
                    makeup_db=makeup_db,
                    fs=fs
                )

        # 3 & 4. Equalizer (3-Band) + De-Esser (combined biquad cascade)
        biquad_list = []
        eq_preset = config["eq_presets"][int(eq_idx)]
        if not eq_preset.get("bypass", True):
            for eff in eq_preset.get("effects", []):
                p = eff["params"]
                biquad_list.append((
                    eff["type"],
                    float(p["cutoff_frequency_hz"]),
                    float(p["q"]),
                    float(p["gain_db"])
                ))

        ds_preset = config["de_esser_presets"][int(deesser_idx)]
        if not ds_preset.get("bypass", True):
            for eff in ds_preset.get("effects", []):
                p = eff["params"]
                biquad_list.append((
                    eff["type"],
                    float(p["cutoff_frequency_hz"]),
                    float(p["q"]),
                    float(p["gain_db"])
                ))

        if biquad_list:
            y = self._apply_biquad_cascade_fft(y, biquad_list, fs)

        # 5. Reverb
        rvb_preset = config["reverb_presets"][int(reverb_idx)]
        if not rvb_preset.get("bypass", True):
            for eff in rvb_preset.get("effects", []):
                if eff["type"] == "Reverb":
                    p = eff["params"]
                    y = self._apply_reverb(
                        y,
                        float(p["room_size"]),
                        float(p["damping"]),
                        float(p["wet_level"]),
                        float(p["dry_level"]),
                        fs=fs
                    )

        # 6. Limiter
        lim_preset = config["limiter_presets"][int(limiter_idx)]
        if not lim_preset.get("bypass", True):
            for eff in lim_preset.get("effects", []):
                if eff["type"] == "Limiter":
                    p = eff["params"]
                    y = self._apply_envelope_dynamics(
                        y,
                        float(p["threshold_db"]),
                        ratio=100.0,
                        attack_ms=0.5,
                        release_ms=float(p["release_ms"]),
                        mode="limiter",
                        makeup_db=0.0,
                        fs=fs
                    )

        # Final safety clamp to 16-bit PCM range
        y = np.clip(y, -1.0, 1.0)
        pcm_int16 = (y * 32767.0).astype(np.int16)

        with wave.open(output_wav_path, "wb") as out_wf:
            out_wf.setnchannels(1)
            out_wf.setsampwidth(2)
            out_wf.setframerate(fs)
            out_wf.writeframes(pcm_int16.tobytes())

        return output_wav_path