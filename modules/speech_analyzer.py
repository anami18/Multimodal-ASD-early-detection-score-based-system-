"""
Speech Analysis Module for ASD Screening
=========================================
Analyzes speech features associated with ASD in children aged 2-4.

Key features based on s41598-025-01500-6 (Nature Scientific Reports):
- Mean Length of Utterance (MLU) proxy
- Pitch (F0) variability and range - ASD children often show atypical prosody
- Rhythm irregularity - echolalia, repetition patterns
- Volume (intensity) dynamics
- Speech rate and pause patterns
- Phonation quality (jitter, shimmer)
- Mean Length of Turn (MLT) ratio

ASD Speech Markers:
- Reduced pitch variation (monotone) OR extreme pitch variation
- Atypical rhythm and prosody
- Excessive pausing or very fast speech
- Repetitive phonation patterns (echolalia proxy)
- Reduced phonemic complexity
"""

import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import scipy.signal as signal
from scipy.stats import kurtosis, skew
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import threading
import queue
import time
import os


@dataclass
class SpeechFeatures:
    """Extracted speech features for a single recording."""

    # Prosodic features
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_range: float = 0.0          # max - min F0
    pitch_kurtosis: float = 0.0       # distribution shape
    pitch_contour_slope: float = 0.0  # rising/falling tendency

    # Rhythm features
    speech_rate: float = 0.0           # syllables per second estimate
    pause_ratio: float = 0.0           # fraction of time silent
    pause_count: int = 0               # number of pauses > threshold
    rhythm_regularity: float = 0.0     # autocorrelation of energy envelope

    # Volume/intensity features
    rms_mean: float = 0.0
    rms_std: float = 0.0
    rms_dynamic_range: float = 0.0

    # Spectral features
    mfcc_mean: List[float] = field(default_factory=lambda: [0.0] * 13)
    mfcc_std: List[float] = field(default_factory=lambda: [0.0] * 13)
    spectral_centroid_mean: float = 0.0
    spectral_rolloff_mean: float = 0.0

    # Vocal quality
    jitter: float = 0.0         # cycle-to-cycle F0 variation
    shimmer: float = 0.0        # cycle-to-cycle amplitude variation
    hnr: float = 0.0            # harmonics-to-noise ratio proxy

    # Utterance structure (MLU proxy)
    mlu_proxy: float = 0.0           # mean energy segment duration
    repetition_score: float = 0.0    # echolalia proxy (pattern self-similarity)

    # Derived ASD indicators
    asd_risk_score: float = 0.0      # 0-100
    duration_seconds: float = 0.0
    valid: bool = False


@dataclass
class SpeechSessionResult:
    features: SpeechFeatures
    recordings: List[np.ndarray] = field(default_factory=list)
    prompts_completed: int = 0
    asd_risk_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)


class SpeechAnalyzer:
    """
    Records and analyzes speech for ASD screening.
    Implements acoustic feature extraction aligned with Nature paper findings.
    """

    SAMPLE_RATE = 22050
    HOP_LENGTH = 512
    N_FFT = 2048
    SILENCE_THRESHOLD = 0.01   # RMS threshold for silence detection
    MIN_PAUSE_DURATION = 0.3   # seconds

    # Screening prompts designed for 2-4 year olds (STAT-inspired)
    SCREENING_PROMPTS = [
        {
            "instruction": "Say the name of this: 🐱 (show cat picture)\nClick START then speak",
            "label": "single_word",
            "expected_duration": 3.0
        },
        {
            "instruction": "Tell me what you see (show a simple scene image)\nClick START then speak",
            "label": "description",
            "expected_duration": 8.0
        },
        {
            "instruction": "Count with me: 1, 2, 3...\nClick START then repeat",
            "label": "repetition",
            "expected_duration": 5.0
        },
        {
            "instruction": "Say: 'I want the ball'\nClick START then speak",
            "label": "prompted_utterance",
            "expected_duration": 4.0
        },
    ]

    def __init__(self):
        self.recordings: List[np.ndarray] = []
        self.features_list: List[SpeechFeatures] = []
        self._recording = False
        self._audio_buffer = []
        self._audio_queue = queue.Queue()

    def start_recording(self, duration: float = 8.0) -> np.ndarray:
        """
        Record audio for specified duration.
        Returns numpy array of audio samples.
        """
        print(f"Recording for {duration}s...")
        audio = sd.rec(
            int(duration * self.SAMPLE_RATE),
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        audio = audio.flatten()
        self.recordings.append(audio)
        return audio

    def record_with_callback(self, duration: float = 8.0,
                             on_complete=None) -> threading.Thread:
        """Non-blocking recording with callback."""
        def _record():
            audio = self.start_recording(duration)
            if on_complete:
                on_complete(audio)

        thread = threading.Thread(target=_record, daemon=True)
        thread.start()
        return thread

    def save_recording(self, audio: np.ndarray, path: str):
        """Save recording to file."""
        sf.write(path, audio, self.SAMPLE_RATE)

    def load_recording(self, path: str) -> np.ndarray:
        """Load recording from file."""
        audio, sr = librosa.load(path, sr=self.SAMPLE_RATE)
        return audio

    def extract_features(self, audio: np.ndarray) -> SpeechFeatures:
        """
        Extract comprehensive speech features for ASD analysis.
        Based on prosodic and linguistic markers from s41598-025-01500-6.
        """
        features = SpeechFeatures()
        features.duration_seconds = len(audio) / self.SAMPLE_RATE

        if len(audio) < self.SAMPLE_RATE * 0.5:  # Less than 0.5 seconds
            return features

        try:
            # ─── Pitch (F0) Analysis ───────────────────────────────────
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),   # ~65 Hz - very low
                fmax=librosa.note_to_hz('C7'),   # ~2093 Hz - children's range
                sr=self.SAMPLE_RATE,
                hop_length=self.HOP_LENGTH
            )

            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
            voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

            if len(voiced_f0) > 5:
                features.pitch_mean = float(np.mean(voiced_f0))
                features.pitch_std = float(np.std(voiced_f0))
                features.pitch_range = float(np.max(voiced_f0) - np.min(voiced_f0))
                features.pitch_kurtosis = float(kurtosis(voiced_f0))

                # Pitch contour - linear regression slope (normalized)
                x = np.arange(len(voiced_f0))
                if len(x) > 1:
                    slope = np.polyfit(x, voiced_f0, 1)[0]
                    features.pitch_contour_slope = float(slope)

            # ─── Rhythm & Pause Analysis ───────────────────────────────
            rms_frames = librosa.feature.rms(
                y=audio, frame_length=self.N_FFT, hop_length=self.HOP_LENGTH
            )[0]

            silence_mask = rms_frames < self.SILENCE_THRESHOLD
            features.pause_ratio = float(np.mean(silence_mask))

            # Count pause segments
            pause_count = 0
            in_pause = False
            pause_start = 0
            min_pause_frames = int(self.MIN_PAUSE_DURATION * self.SAMPLE_RATE / self.HOP_LENGTH)

            for i, is_silent in enumerate(silence_mask):
                if is_silent and not in_pause:
                    in_pause = True
                    pause_start = i
                elif not is_silent and in_pause:
                    if i - pause_start >= min_pause_frames:
                        pause_count += 1
                    in_pause = False

            features.pause_count = pause_count

            # Speech rate (syllable-count proxy via energy peaks)
            envelope = np.abs(signal.hilbert(audio))
            envelope_smooth = signal.savgol_filter(envelope, 51, 3)
            peaks, _ = signal.find_peaks(envelope_smooth,
                                          height=np.mean(envelope_smooth) * 0.3,
                                          distance=int(0.08 * self.SAMPLE_RATE))
            speaking_duration = features.duration_seconds * (1 - features.pause_ratio)
            if speaking_duration > 0:
                features.speech_rate = len(peaks) / speaking_duration

            # Rhythm regularity (autocorrelation of RMS envelope)
            if len(rms_frames) > 20:
                autocorr = np.correlate(rms_frames - np.mean(rms_frames),
                                         rms_frames - np.mean(rms_frames), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                if len(autocorr) > 1 and autocorr[0] > 0:
                    autocorr_normalized = autocorr / autocorr[0]
                    # Rhythmic speech has peaks in autocorrelation
                    peaks_ac, _ = signal.find_peaks(autocorr_normalized[1:], height=0.3)
                    features.rhythm_regularity = float(min(len(peaks_ac) / 5.0, 1.0))

            # ─── Volume / Intensity ────────────────────────────────────
            features.rms_mean = float(np.mean(rms_frames))
            features.rms_std = float(np.std(rms_frames))
            rms_nonsilent = rms_frames[~silence_mask]
            if len(rms_nonsilent) > 0:
                features.rms_dynamic_range = float(
                    np.percentile(rms_nonsilent, 95) - np.percentile(rms_nonsilent, 5)
                )

            # ─── MFCCs (Spectral Shape) ────────────────────────────────
            mfccs = librosa.feature.mfcc(
                y=audio, sr=self.SAMPLE_RATE, n_mfcc=13,
                n_fft=self.N_FFT, hop_length=self.HOP_LENGTH
            )
            features.mfcc_mean = mfccs.mean(axis=1).tolist()
            features.mfcc_std = mfccs.std(axis=1).tolist()

            # Spectral centroid
            spec_centroid = librosa.feature.spectral_centroid(
                y=audio, sr=self.SAMPLE_RATE, hop_length=self.HOP_LENGTH
            )[0]
            features.spectral_centroid_mean = float(np.mean(spec_centroid))

            # Spectral rolloff
            spec_rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.SAMPLE_RATE, hop_length=self.HOP_LENGTH
            )[0]
            features.spectral_rolloff_mean = float(np.mean(spec_rolloff))

            # ─── Vocal Quality ─────────────────────────────────────────
            if len(voiced_f0) > 5:
                features.jitter = self._compute_jitter(voiced_f0)
                features.shimmer = self._compute_shimmer(audio, voiced_flag, self.HOP_LENGTH)

            # HNR proxy (via AC of voiced frames)
            features.hnr = self._compute_hnr_proxy(audio)

            # ─── MLU Proxy ────────────────────────────────────────────
            # Estimate utterance length from energy segments
            if len(rms_frames) > 0:
                speech_frames = rms_frames[~silence_mask]
                if len(speech_frames) > 0:
                    features.mlu_proxy = float(len(speech_frames) / max(pause_count + 1, 1))

            # ─── Repetition Score (Echolalia Proxy) ───────────────────
            features.repetition_score = self._compute_repetition_score(mfccs)

            # ─── ASD Risk Score ────────────────────────────────────────
            features.asd_risk_score = self._compute_speech_risk_score(features)
            features.valid = True

        except Exception as e:
            print(f"Feature extraction error: {e}")
            import traceback
            traceback.print_exc()

        return features

    def _compute_jitter(self, f0: np.ndarray) -> float:
        """Compute period jitter (cycle-to-cycle F0 variation)."""
        if len(f0) < 2:
            return 0.0
        periods = 1.0 / (f0 + 1e-8)
        diffs = np.abs(np.diff(periods))
        return float(np.mean(diffs) / (np.mean(periods) + 1e-8))

    def _compute_shimmer(self, audio: np.ndarray, voiced_flag, hop_length: int) -> float:
        """Compute amplitude shimmer proxy."""
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        if voiced_flag is not None and len(voiced_flag) > 0:
            min_len = min(len(rms), len(voiced_flag))
            voiced_rms = rms[:min_len][voiced_flag[:min_len]]
        else:
            voiced_rms = rms

        if len(voiced_rms) < 2:
            return 0.0

        diffs = np.abs(np.diff(voiced_rms))
        return float(np.mean(diffs) / (np.mean(voiced_rms) + 1e-8))

    def _compute_hnr_proxy(self, audio: np.ndarray) -> float:
        """Compute Harmonics-to-Noise Ratio proxy."""
        # Use autocorrelation method
        frame_length = 2048
        if len(audio) < frame_length:
            return 0.0

        frame = audio[:frame_length]
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        if autocorr[0] == 0:
            return 0.0

        # Find first peak after zero crossing
        # (harmonic component vs noise)
        noise_level = np.mean(np.abs(autocorr[autocorr < 0]))
        harmonic_level = np.max(autocorr[1:]) if len(autocorr) > 1 else 0

        if noise_level > 0:
            return float(20 * np.log10((harmonic_level + 1e-8) / (noise_level + 1e-8)))
        return 0.0

    def _compute_repetition_score(self, mfccs: np.ndarray) -> float:
        """
        Echolalia/repetition proxy:
        Measures self-similarity in MFCC sequence.
        Higher score = more repetitive patterns.
        """
        if mfccs.shape[1] < 20:
            return 0.0

        # Divide into segments and compare
        n_segments = min(4, mfccs.shape[1] // 10)
        if n_segments < 2:
            return 0.0

        seg_length = mfccs.shape[1] // n_segments
        segments = [mfccs[:, i*seg_length:(i+1)*seg_length] for i in range(n_segments)]
        seg_means = [s.mean(axis=1) for s in segments]

        # Cosine similarity between segments
        similarities = []
        for i in range(len(seg_means)):
            for j in range(i+1, len(seg_means)):
                a, b = seg_means[i], seg_means[j]
                sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0

    def _compute_speech_risk_score(self, f: SpeechFeatures) -> float:
        """
        Compute ASD speech risk score (0-100).
        Based on markers from s41598-025-01500-6 and clinical literature:
        - Atypical pitch patterns
        - Unusual rhythm/pausing
        - High repetition (echolalia)
        - Reduced prosodic variation
        """
        risk = 0.0

        # 1. Pitch variability (ASD = too low OR too high pitch std)
        # Normal child pitch std ~ 30-80 Hz. ASD often very low or extreme.
        if f.pitch_std > 0:
            if f.pitch_std < 20:        # Very monotone
                risk += 20.0
            elif f.pitch_std > 120:     # Very extreme variation
                risk += 15.0
            else:                        # Normal range - no risk
                risk += 0.0
        else:
            risk += 15.0  # No detected pitch (silence or noise)

        # 2. Excessive pausing (ASD children often pause atypically)
        if f.pause_ratio > 0.6:
            risk += 15.0
        elif f.pause_ratio > 0.45:
            risk += 8.0

        # 3. High repetition score (echolalia proxy)
        # repetition_score is cosine similarity, high = repetitive
        if f.repetition_score > 0.85:
            risk += 20.0
        elif f.repetition_score > 0.70:
            risk += 12.0

        # 4. Abnormal speech rate
        # Normal 2-4yr: ~2-4 syllables/sec
        if f.speech_rate < 0.5 or f.speech_rate > 8.0:
            risk += 15.0
        elif f.speech_rate < 1.0 or f.speech_rate > 6.0:
            risk += 8.0

        # 5. Low rhythm regularity (ASD often have irregular rhythm)
        if f.rhythm_regularity < 0.15:
            risk += 10.0
        elif f.rhythm_regularity < 0.30:
            risk += 5.0

        # 6. Jitter (vocal quality irregularity)
        if f.jitter > 0.05:
            risk += 10.0
        elif f.jitter > 0.03:
            risk += 5.0

        # 7. Low dynamic range (flat affect in speech volume)
        if f.rms_dynamic_range < 0.005:
            risk += 10.0

        return min(max(risk, 0.0), 100.0)

    def analyze_all_recordings(self) -> SpeechSessionResult:
        """Analyze all recorded sessions and compute aggregate result."""
        if not self.recordings:
            return SpeechSessionResult(
                features=SpeechFeatures(),
                asd_risk_score=50.0
            )

        all_features = [self.extract_features(r) for r in self.recordings]
        valid_features = [f for f in all_features if f.valid]

        if not valid_features:
            return SpeechSessionResult(
                features=SpeechFeatures(),
                asd_risk_score=50.0
            )

        # Aggregate features (mean across sessions)
        agg = SpeechFeatures()
        agg.pitch_std = np.mean([f.pitch_std for f in valid_features])
        agg.pitch_range = np.mean([f.pitch_range for f in valid_features])
        agg.pause_ratio = np.mean([f.pause_ratio for f in valid_features])
        agg.speech_rate = np.mean([f.speech_rate for f in valid_features])
        agg.rhythm_regularity = np.mean([f.rhythm_regularity for f in valid_features])
        agg.repetition_score = np.mean([f.repetition_score for f in valid_features])
        agg.jitter = np.mean([f.jitter for f in valid_features])
        agg.rms_dynamic_range = np.mean([f.rms_dynamic_range for f in valid_features])
        agg.valid = True

        avg_risk = np.mean([f.asd_risk_score for f in valid_features])
        agg.asd_risk_score = avg_risk

        # Identify specific risk factors
        risk_factors = []
        if agg.pitch_std < 20:
            risk_factors.append("Reduced pitch variation (monotone speech)")
        if agg.pause_ratio > 0.5:
            risk_factors.append("Excessive pausing in speech")
        if agg.repetition_score > 0.75:
            risk_factors.append("Repetitive speech patterns detected")
        if agg.speech_rate < 1.0:
            risk_factors.append("Very slow speech rate")
        elif agg.speech_rate > 6.0:
            risk_factors.append("Unusually fast speech rate")
        if agg.rhythm_regularity < 0.2:
            risk_factors.append("Irregular speech rhythm")

        return SpeechSessionResult(
            features=agg,
            recordings=self.recordings.copy(),
            prompts_completed=len(valid_features),
            asd_risk_score=avg_risk,
            risk_factors=risk_factors
        )
