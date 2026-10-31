"""NumPy inference for the character-level phishing URL CNN.

The model was trained with TensorFlow/Keras (see the Colab notebook), and the
trained weights were exported to ``artifacts/model_weights.npz``. This module
re-implements the forward pass with NumPy only, so the web app does not need
TensorFlow. The NumPy output was verified against the Keras model on 3,000 test
URLs (max probability difference ~1.6e-06, 100% label agreement).

Architecture:
    Embedding(97, 32) -> Conv1D(64, kernel 5, 'same', ReLU)
    -> GlobalMaxPooling1D -> Dense(64, ReLU) -> Dropout(0.3, train only)
    -> Dense(1, sigmoid)
"""

import json
import os
from functools import lru_cache

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(ROOT, "artifacts")

DEFAULT_MAX_LENGTH = 300
DEFAULT_THRESHOLD = 0.5
PAD_INDEX = 0
UNK_INDEX = 1
MAX_INPUT_CHARS = 10000  # reject absurdly long input before processing


class PhishingDetector:
    """Loads the exported artifacts and classifies URLs."""

    def __init__(self, artifact_dir=ARTIFACT_DIR):
        self.artifact_dir = artifact_dir

        with open(os.path.join(artifact_dir, "vocabulary.json"), "r") as f:
            vocabulary = json.load(f)
        self.vocab_index = {token: i for i, token in enumerate(vocabulary)}

        config_path = os.path.join(artifact_dir, "preprocessing_config.json")
        max_length = DEFAULT_MAX_LENGTH
        threshold = DEFAULT_THRESHOLD
        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            max_length = int(config.get("max_length", max_length))
            threshold = float(config.get("prediction_threshold", threshold))
        self.max_length = max_length
        self.threshold = threshold

        weights = np.load(os.path.join(artifact_dir, "model_weights.npz"))
        self.embedding = weights["embedding"]
        self.conv_kernel = weights["conv_kernel"]
        self.conv_bias = weights["conv_bias"]
        self.dense_kernel = weights["dense1_kernel"]
        self.dense_bias = weights["dense1_bias"]
        self.out_kernel = weights["out_kernel"]
        self.out_bias = weights["out_bias"]

        self.kernel_size = self.conv_kernel.shape[0]
        self.vocab_size = len(vocabulary)

    # ------------------------------------------------------------------
    def encode(self, url):
        """Same as TextVectorization(split='character', standardize=None):
        characters -> vocabulary index, unknown -> 1, truncate, pad with 0."""
        ids = [self.vocab_index.get(ch, UNK_INDEX) for ch in url[: self.max_length]]
        ids += [PAD_INDEX] * (self.max_length - len(ids))
        return np.array(ids, dtype=np.int64)

    def predict_proba(self, url):
        """Return the probability that the URL is phishing (float in 0..1)."""
        url = self._validate(url)
        ids = self.encode(url)

        x = self.embedding[ids]                                  # (L, 32)
        total_pad = self.kernel_size - 1                         # 'same' padding
        left = total_pad // 2
        right = total_pad - left
        xp = np.pad(x, ((left, right), (0, 0)))

        conv = np.zeros((self.max_length, self.conv_bias.shape[0]))
        for k in range(self.kernel_size):
            conv += xp[k : k + self.max_length] @ self.conv_kernel[k]
        conv = np.maximum(conv + self.conv_bias, 0.0)            # ReLU

        pooled = conv.max(axis=0)                                # global max pool
        hidden = np.maximum(pooled @ self.dense_kernel + self.dense_bias, 0.0)
        logit = float((hidden @ self.out_kernel + self.out_bias)[0])
        return float(1.0 / (1.0 + np.exp(-logit)))

    def predict(self, url):
        """Return a JSON-friendly dict with label and probabilities."""
        url = self._validate(url)
        probability = self.predict_proba(url)
        label = "phishing" if probability >= self.threshold else "legitimate"
        return {
            "url": url,
            "prediction": label,
            "phishing_probability": round(probability, 4),
            "legitimate_probability": round(1.0 - probability, 4),
            "threshold": self.threshold,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _validate(url):
        if not isinstance(url, str) or not url.strip():
            raise ValueError("Please provide a non-empty URL.")
        url = url.strip()
        if len(url) > MAX_INPUT_CHARS:
            raise ValueError(f"URL is too long (max {MAX_INPUT_CHARS} characters).")
        return url


@lru_cache(maxsize=1)
def get_detector():
    """Load the model once and reuse it between requests."""
    return PhishingDetector()


def predict_url(url):
    """Convenience wrapper: classify one URL with the default artifacts."""
    return get_detector().predict(url)


if __name__ == "__main__":
    import sys

    for item in sys.argv[1:] or ["https://www.wikipedia.org"]:
        print(json.dumps(predict_url(item), indent=2))
