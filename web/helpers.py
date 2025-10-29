# web/helpers.py
import os
import tempfile
import shutil
from contextlib import contextmanager

@contextmanager
def atomic_write(dest_path, mode="wb"):
    """
    A context manager for atomic file writes.

    Writes to a temporary file and then atomically moves it to the
    destination, preventing partial or corrupted files.
    """
    d = os.path.dirname(dest_path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, mode) as f:
            yield f
        os.replace(tmp, dest_path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise

import numpy as np
import librosa
from audio_processing import get_audio_features
import logging

logger = logging.getLogger(__name__)

def error_response(message, code="INTERNAL_ERROR", details=None):
    """Creates a standardized JSON error response."""
    return {"ok": False, "error": {"code": code, "message": message, "details": details}}

def process_uploaded_audio(filepath: str, config: dict) -> dict:
    """Process uploaded audio file and extract features."""
    try:
        # Load audio using librosa
        y, sr = librosa.load(filepath, sr=None)

        # Extract basic information
        duration = len(y) / float(sr) if sr and len(y) else 0.0

        # BPM estimation (existing code)
        tempo_val = None
        try:
            est = librosa.beat.tempo(y=y, sr=sr)
            if hasattr(est, '__len__') and len(est) > 0:
                tempo_val = float(est[0])
            else:
                tempo_val = float(est)
        except Exception:
            tempo_val = None

        if tempo_val is None:
            try:
                tempo_bt, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo_val = float(tempo_bt)
            except Exception:
                tempo_val = None

        if tempo_val is not None:
            if tempo_val <= 0 or tempo_val > 1000:
                tempo_val = None

        # ALWAYS extract features and save .npy file
        features = None
        npy_filename = None
        try:
            features = get_audio_features(
                filepath,
                source_resolution_ms=config['data']['source_resolution_ms'],
                frame_duration_ms=config['data']['time_quantization_ms']
            )

            # Save features as .npy file alongside audio
            if features is not None:
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                npy_filename = f"{base_name}.npy"
                npy_path = os.path.join(os.path.dirname(filepath), npy_filename)
                np.save(npy_path, features)
                logger.info(f"Saved features to {npy_path} with shape {features.shape}")

        except Exception as e:
            logger.warning(f"Feature extraction failed for {filepath}: {e}")

        logger.info(f"Processed audio {os.path.basename(filepath)} duration={duration:.2f}s detected_bpm={tempo_val}")

        return {
            'duration': duration,
            'bpm': int(tempo_val) if tempo_val is not None else None,
            'features_extracted': features is not None,
            'feature_shape': features.shape if features is not None else None,
            'npy_filename': npy_filename
        }

    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        return {'error': str(e)}
