import wave
import numpy as np

class AudioPreprocessor:
    def __init__(self):
        self.target_frames = 1723
        self.n_fft = 1024
        self.hop_length = 512
        
        # Exact mathematical replica of Librosa's default periodic Hann window
        self.window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n_fft) / self.n_fft)

    def _read_normalized_mono(self, file_path: str) -> np.ndarray:
        with wave.open(file_path, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            
            # Cap reading to 20 seconds to prevent memory overflow on long files
            max_frames = 20 * framerate
            frames_to_read = min(wav_file.getnframes(), max_frames)
            raw_data = wav_file.readframes(frames_to_read)

        if sampwidth == 2:  # 16-bit PCM
            data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            data = np.frombuffer(raw_data, dtype=np.float32)

        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)
        return data

    def process_wav_to_tensor(self, file_path: str) -> bytes:
        data = self._read_normalized_mono(file_path)

        # Explicitly handle digital silence to avoid math domain errors
        if len(data) == 0 or np.max(np.abs(data)) == 0.0:
            return np.full((513, self.target_frames), -80.0, dtype=np.float32).tobytes()

        # 1:1 LIBROSA STFT REPLICA
        pad_width = self.n_fft // 2
        y_padded = np.pad(data, pad_width, mode='reflect')

        n_frames_stft = 1 + (len(y_padded) - self.n_fft) // self.hop_length
        strides = (y_padded.itemsize * self.hop_length, y_padded.itemsize)
        frames = np.lib.stride_tricks.as_strided(
            y_padded, 
            shape=(n_frames_stft, self.n_fft), 
            strides=strides
        )

        stft_matrix = np.fft.rfft(frames * self.window, axis=1).T
        magnitude = np.abs(stft_matrix)
        amin = 1e-5 
        magnitude = np.maximum(amin, magnitude)

        # Relative Scaling matching ML dataset
        ref = np.max(magnitude)
        if ref > 0:
            spectrogram_db = 20 * np.log10(magnitude / ref)
        else:
            spectrogram_db = np.full(magnitude.shape, -80.0)
            
        # Floor at -80 dB and max at 0 dB
        spectrogram_db = np.clip(spectrogram_db, a_min=-80.0, a_max=0.0)

        # Tensor Formatting [513, 1723]
        current_frames = spectrogram_db.shape[1]
        if current_frames < self.target_frames:
            pad_w = self.target_frames - current_frames
            spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, pad_w)), mode='constant', constant_values=-80.0)
        elif current_frames > self.target_frames:
            spectrogram_db = spectrogram_db[:, :self.target_frames]

        return np.ascontiguousarray(spectrogram_db, dtype=np.float32).tobytes()

    def recommend_limiter_idx(self, file_path: str, comp_idx: int, eq_idx: int) -> int:
        """
        Evaluates the True Peak (TP) via O(N) top-10 peaks 4x local oversampling (+-50 samples)
        and checks the Ceiling Violation Penalty P_ceiling = max(0, TP_projected - (-3.0 dBFS)).
        Returns 1 (Active Limiter at -3.0 dB) if P_ceiling > 0, else 0 (Bypassed).
        """
        data = self._read_normalized_mono(file_path)
        if len(data) == 0:
            return 0

        abs_data = np.abs(data)
        max_val = float(np.max(abs_data))
        if max_val <= 1e-9:
            return 0

        # Locate top 10 peaks in O(N) time
        k = min(10, len(abs_data))
        top_indices = np.argpartition(abs_data, -k)[-k:]

        max_tp = max_val
        n = len(data)
        for idx in top_indices:
            start = max(0, int(idx) - 50)
            end = min(n, int(idx) + 51)
            segment = data[start:end]
            if len(segment) < 4:
                continue
            L = len(segment)
            hann = np.hanning(L)
            spec = np.fft.rfft(segment * hann)
            upsampled = np.fft.irfft(spec, n=L * 4) * 4.0
            win_up = np.interp(np.linspace(0, L - 1, L * 4), np.arange(L), hann)
            center_slice = slice(int(L * 4 * 0.25), int(L * 4 * 0.75))
            restored = upsampled[center_slice] / np.maximum(win_up[center_slice], 0.1)
            seg_tp = float(np.max(np.abs(restored)))
            if seg_tp > max_tp:
                max_tp = seg_tp

        tp_raw_db = 20.0 * np.log10(max(max_tp, 1e-9))

        # Account for downstream gain from Compressor Makeup (+3.0 dB) and positive EQ boosts
        chain_gain_db = 3.0 if int(comp_idx) > 0 else 0.0
        if int(eq_idx) in (1, 2, 3):
            chain_gain_db += 1.5

        tp_projected_db = tp_raw_db + chain_gain_db
        p_ceiling = max(0.0, tp_projected_db - (-3.0))

        return 1 if p_ceiling > 0.0 else 0