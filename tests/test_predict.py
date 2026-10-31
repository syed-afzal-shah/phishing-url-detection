"""End-to-end tests for the phishing URL detector.

Run from the repository root with either:
    python -m unittest discover -s tests -v
    pytest tests -v

The Keras parity test is skipped automatically when TensorFlow is not installed
(pip install -r requirements-dev.txt to enable it).
"""

import csv
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.phishing_model import PhishingDetector  # noqa: E402

ARTIFACT_DIR = os.path.join(ROOT, "artifacts")
SAMPLE_CSV = os.path.join(ROOT, "tests", "sample_test_urls.csv")


class TestEncoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = PhishingDetector()

    def test_encoding_is_padded_to_fixed_length(self):
        ids = self.detector.encode("http://a.com")
        self.assertEqual(ids.shape, (self.detector.max_length,))
        self.assertEqual(int(ids[-1]), 0)  # padding index

    def test_long_url_is_truncated(self):
        ids = self.detector.encode("a" * 5000)
        self.assertEqual(ids.shape, (self.detector.max_length,))

    def test_unknown_character_maps_to_unk(self):
        ids = self.detector.encode("\u4e2d")  # character not in the vocabulary
        self.assertEqual(int(ids[0]), 1)


class TestPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = PhishingDetector()

    def test_output_format(self):
        result = self.detector.predict("https://www.wikipedia.org")
        for key in ("url", "prediction", "phishing_probability",
                    "legitimate_probability", "threshold"):
            self.assertIn(key, result)
        self.assertIn(result["prediction"], ("phishing", "legitimate"))
        total = result["phishing_probability"] + result["legitimate_probability"]
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_known_legitimate_urls(self):
        for url in ("https://www.wikipedia.org",
                    "https://www.google.com/search?q=python",
                    "https://github.com/tensorflow/tensorflow"):
            with self.subTest(url=url):
                self.assertEqual(self.detector.predict(url)["prediction"], "legitimate")

    def test_known_phishing_urls(self):
        for url in ("http://paypal-secure-login.verify-account.example.ru/webscr.php?cmd=_login",
                    "http://192.168.10.5/bank/login.php?id=8827364",
                    "http://secure-update-amazon.account-verify.example.com/signin"):
            with self.subTest(url=url):
                self.assertEqual(self.detector.predict(url)["prediction"], "phishing")

    def test_invalid_input_raises(self):
        for bad in ("", "   ", None, 123):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    self.detector.predict(bad)

    def test_sample_file_accuracy(self):
        """200 held-out test URLs (100 per class) taken from the real test split."""
        with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertGreaterEqual(len(rows), 100)
        correct = 0
        for row in rows:
            predicted = 1 if self.detector.predict_proba(row["url"]) >= 0.5 else 0
            correct += int(predicted == int(row["label"]))
        accuracy = correct / len(rows)
        print(f"\nSample accuracy: {accuracy:.4f} on {len(rows)} URLs")
        self.assertGreaterEqual(accuracy, 0.93)


try:
    import tensorflow as tf  # noqa: F401
    HAS_TF = True
except Exception:
    HAS_TF = False


@unittest.skipUnless(HAS_TF, "TensorFlow not installed")
class TestKerasParity(unittest.TestCase):
    def test_numpy_matches_keras(self):
        import numpy as np
        import tensorflow as tf

        with open(os.path.join(ARTIFACT_DIR, "vocabulary.json")) as f:
            vocab = json.load(f)
        model = tf.keras.models.load_model(
            os.path.join(ARTIFACT_DIR, "phishing_url_cnn.keras"), compile=False)
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=100, standardize=None, split="character",
            output_mode="int", output_sequence_length=300, vocabulary=vocab)

        with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
            urls = [row["url"] for row in csv.DictReader(f)][:50]

        keras_p = model.predict(vectorizer(tf.constant(urls)), verbose=0).ravel()
        detector = PhishingDetector()
        numpy_p = np.array([detector.predict_proba(u) for u in urls])
        self.assertLess(float(np.abs(keras_p - numpy_p).max()), 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
