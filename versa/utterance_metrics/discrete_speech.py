#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for discrete speech metrics evaluation."""

import logging

import librosa
import numpy as np
import torch

# Lazy import - will be imported when discrete_speech_setup is called
SpeechBERTScore = None
SpeechBLEU = None
SpeechTokenDistance = None

logger = logging.getLogger(__name__)


def _check_torch_version():
    """Check if torch version is compatible with transformers library.

    Due to CVE-2025-32434, transformers requires torch >= 2.6 when loading
    models that use torch.load(). This check provides a clear error message
    if the version is too old.
    """
    try:
        torch_version = torch.__version__.split("+")[0]  # Remove CUDA suffix if present
        version_parts = torch_version.split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0

        # Require torch 2.6+ for CVE-2025-32434 compliance
        if major < 2 or (major == 2 and minor < 6):
            raise ValueError(
                f"torch version {torch.__version__} is too old. "
                "Due to CVE-2025-32434 (torch.load vulnerability), "
                "transformers requires torch >= 2.6 to load models. "
                "Please upgrade torch: pip install 'torch>=2.6' "
                "(Note: torch 2.6.0 is available for CUDA 12.4 or CUDA 11.8)"
            )
    except (IndexError, ValueError) as e:
        # If version parsing fails (IndexError), let transformers handle the error
        # If it's our ValueError about version being too old, re-raise it
        if "torch version" in str(e):
            raise
        # Otherwise, version parsing failed - let transformers handle it
        pass


def _import_discrete_speech_metrics():
    """Lazy import of discrete_speech_metrics to avoid import-time errors."""
    global SpeechBERTScore, SpeechBLEU, SpeechTokenDistance

    if SpeechBERTScore is None:
        try:
            from discrete_speech_metrics import (
                SpeechBERTScore,
                SpeechBLEU,
                SpeechTokenDistance,
            )
        except ImportError as e:
            raise ImportError("Please install discrete_speech_metrics and retry") from e
        except OSError as e:
            # Handle torch/torchaudio compatibility issues
            raise OSError(
                f"Failed to load discrete_speech_metrics due to library compatibility issue: {e}. "
                "This may be due to incompatible torch/torchaudio versions. "
                "Please ensure torch and torchaudio versions are compatible, "
                "or upgrade to torch >= 2.6 to resolve CVE-2025-32434."
            ) from e


def discrete_speech_setup(use_gpu=False):
    """Set up discrete speech metrics.

    Args:
        use_gpu (bool, optional): Whether to use GPU. Defaults to False.

    Returns:
        dict: Dictionary containing the initialized metrics.

    Raises:
        ValueError: If torch version is too old (< 2.6) due to CVE-2025-32434.
        ImportError: If discrete_speech_metrics cannot be imported.
        OSError: If there are library compatibility issues (e.g., torch/torchaudio mismatch).
    """
    # Check torch version before attempting to load models
    _check_torch_version()

    # Lazy import to avoid import-time errors
    _import_discrete_speech_metrics()

    # NOTE(jiatong) existing discrete speech metrics only works for 16khz
    # We keep the paper best setting. To use other settings, please conduct the
    # test on your own.

    speech_bert = SpeechBERTScore(
        sr=16000, model_type="wavlm-large", layer=14, use_gpu=use_gpu
    )
    speech_bleu = SpeechBLEU(
        sr=16000,
        model_type="hubert-base",
        vocab=200,
        layer=11,
        n_ngram=2,
        remove_repetition=True,
        use_gpu=use_gpu,
    )
    speech_token_distance = SpeechTokenDistance(
        sr=16000,
        model_type="hubert-base",
        vocab=200,
        layer=6,
        distance_type="jaro-winkler",
        remove_repetition=False,
        use_gpu=use_gpu,
    )
    return {
        "speech_bert": speech_bert,
        "speech_bleu": speech_bleu,
        "speech_token_distance": speech_token_distance,
    }


def discrete_speech_metric(discrete_speech_predictors, pred_x, gt_x, fs):
    """Calculate discrete speech metrics.

    Args:
        discrete_speech_predictors (dict): Dictionary of speech metrics.
        pred_x (np.ndarray): Predicted audio signal.
        gt_x (np.ndarray): Ground truth audio signal.
        fs (int): Sampling rate.

    Returns:
        dict: Dictionary containing the metric scores.

    Raises:
        NotImplementedError: If an unsupported metric is provided.
    """
    scores = {}

    if fs != 16000:
        gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=16000)
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    for key in discrete_speech_predictors.keys():
        if key == "speech_bert":
            score, _, _ = discrete_speech_predictors[key].score(gt_x, pred_x)
        elif key == "speech_bleu" or key == "speech_token_distance":
            score = discrete_speech_predictors[key].score(gt_x, pred_x)
        else:
            raise NotImplementedError(f"Not supported {key}")
        scores[key] = score
    return scores


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    predictor = discrete_speech_setup()
    print(discrete_speech_metric(predictor, a, b, 16000))
