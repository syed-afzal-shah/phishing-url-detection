# Phishing URL Detection with a Character-Level CNN

Deep Learning semester project. A convolutional neural network reads a URL character by character and predicts whether it is **phishing** or **legitimate**.

- **Live demo (Vercel):** `YOUR_VERCEL_URL`
- **Project report:** `YOUR_PROJECT_REPORT_LINK`
- **Presentation:** `YOUR_PRESENTATION_LINK`
- **YouTube video:** `YOUR_YOUTUBE_VIDEO_LINK`

## Results

Evaluated on a held-out test set of 33,973 URLs that were never used for training or model selection.

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 96.94% |
| Precision | 97.24% |
| Recall    | 96.62% |
| F1-score  | 96.93% |
| ROC-AUC   | 0.9965 |

Confusion matrix (rows = true class, columns = predicted class):

|                | Pred. legitimate | Pred. phishing |
|----------------|-----------------:|---------------:|
| **Legitimate** | 16,535           | 465            |
| **Phishing**   | 573              | 16,400         |

Plots: `reports/training_curves.png`, `reports/confusion_matrix.png`. Raw numbers: `reports/metrics.json`.

## Dataset

- Balanced phishing/legitimate URL dataset (`balanced_urls.csv`), 340,000 rows, columns `label` and `url`.
- Source: https://zenodo.org/records/19371661
- License / terms: **check the license shown on the Zenodo record and update this line** before redistributing.
- The full CSV is **not** stored in this repository. Download it from the link above.
- Cleaning: 277 exact duplicate URLs removed (339,723 rows left; no URL had conflicting labels).
- Split: stratified 80/10/10 with `random_state=42` (271,777 train / 33,973 validation / 33,973 test).
- `tests/sample_test_urls.csv` contains 200 URLs (100 per class) from the test split, used by the tests.

## Model

| Item | Value |
|------|-------|
| Input | URL as a string, at most 300 characters (longer URLs are truncated, shorter ones padded) |
| Vocabulary | 97 symbols (padding, unknown, 95 characters), learned from the training split only |
| Layers | Embedding(32) → Conv1D(64, kernel 5, ReLU) → GlobalMaxPooling → Dense(64, ReLU) → Dropout(0.3) → Dense(1, sigmoid) |
| Parameters | 17,633 |
| Training | Adam (lr 0.001), binary cross-entropy, batch size 256, early stopping on validation loss (stopped after 6 epochs) |
| Threshold | 0.5 on the phishing probability |

## Repository structure

```
├── api/predict.py            Vercel serverless endpoint (/api/predict)
├── src/phishing_model.py     NumPy inference (no TensorFlow needed)
├── public/index.html         Website: demo, results, dataset and code details, links
├── artifacts/                Trained model files (see below)
├── reports/                  metrics.json and plots
├── tests/                    Unit tests and 200-URL sample set
├── notebooks/                Training notebook (Google Colab)
├── requirements.txt          Runtime dependency (numpy only)
├── requirements-dev.txt      Optional: TensorFlow etc. for Keras checks
└── vercel.json
```

### Model files in `artifacts/`

| File | Purpose |
|------|---------|
| `phishing_url_cnn.keras` | Full Keras model (needs TensorFlow 2.16+ / Keras 3 to load) |
| `best_model.weights.h5` | Best weights by validation loss (Keras) |
| `model_config.json` | Keras architecture (`tf.keras.models.model_from_json`) |
| `model_weights.npz` | The same weights as NumPy arrays, used by the web app |
| `vocabulary.json` | Character vocabulary (index = line position) |
| `preprocessing_config.json` | Max length, threshold, label mapping |
| `training_history.csv` | Per-epoch training and validation metrics |

## Run locally

```bash
pip install -r requirements.txt
python -m src.phishing_model "https://www.wikipedia.org"
```

Output:

```json
{
  "url": "https://www.wikipedia.org",
  "prediction": "legitimate",
  "phishing_probability": 0.0046,
  "legitimate_probability": 0.9954,
  "threshold": 0.5
}
```

### Use as a Python function

```python
from src.phishing_model import predict_url
print(predict_url("http://192.168.10.5/bank/login.php?id=8827364"))
```

### Load the original Keras model

```python
import tensorflow as tf
model = tf.keras.models.load_model("artifacts/phishing_url_cnn.keras", compile=False)
```

The model expects integer character indices of shape `(batch, 300)`. Use `artifacts/vocabulary.json` with a `TextVectorization(split="character", standardize=None, output_sequence_length=300)` layer, exactly as in the training notebook, or call `src/phishing_model.py` which does the same encoding.

## API

```bash
curl -X POST https://YOUR_VERCEL_URL/api/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.wikipedia.org"}'

curl "https://YOUR_VERCEL_URL/api/predict?url=https://www.wikipedia.org"
```

Errors return JSON with an `error` field (HTTP 400 for empty or invalid input).

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests check encoding, output format, known legitimate and phishing URLs, input validation, accuracy on the 200-URL sample (must be at least 93%), and, if TensorFlow is installed (`pip install -r requirements-dev.txt`), that the NumPy model matches the Keras model.

## Deployment (Vercel)

1. Push this repository to GitHub.
2. In Vercel choose **Add New → Project**, import the repository, keep the default settings, and deploy.
3. `public/` is served as the website, `api/predict.py` as the Python function, and `vercel.json` makes sure `src/` and `artifacts/` are bundled with it.
4. Open `https://YOUR_VERCEL_URL/api/predict` to see a health check (`"status": "ok"`).

## Limitations

- The model sees only the text of the URL. It does not fetch the page, check domain age, certificates or reputation lists.
- Phishing and legitimate URLs in this dataset may differ in style (for example `http` vs `https`, length, path structure). The model can learn such dataset-specific shortcuts, so real-world accuracy can be lower than the 96.94% measured on this test set.
- Newer attack styles, URL shorteners and unusual legitimate URLs may be misclassified.
- Use the output as one signal among others, not as the only security decision.

## Author

`YOUR NAME` · `YOUR ROLL NUMBER` · Deep Learning, Semester Project
