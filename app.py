"""
Iris Flower Classification — Flask web application.

Run:
    python train_model.py
    python app.py

Open http://127.0.0.1:5000
"""
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "iris_model.pkl"
SPECIES_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
FEATURE_NAMES = ["Sepal length", "Sepal width", "Petal length", "Petal width"]

app = Flask(__name__)
model = None
model_error = None


def load_model():
    """Load the trained model relative to this source file."""
    global model, model_error
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Run 'python train_model.py' first."
            )
        model = joblib.load(MODEL_PATH)
        model_error = None
    except Exception as exc:
        model = None
        model_error = str(exc)


load_model()


def classify(values):
    """Return prediction details for four validated measurements."""
    data = np.asarray(values, dtype=float).reshape(1, 4)
    prediction = int(model.predict(data)[0])
    probabilities = model.predict_proba(data)[0]
    return {
        "species": SPECIES_NAMES[prediction],
        "confidence": round(float(probabilities[prediction]) * 100, 2),
        "probabilities": {
            name: round(float(probability) * 100, 2)
            for name, probability in zip(SPECIES_NAMES, probabilities)
        },
    }


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_missing=model is None,
        model_error=model_error,
    )


@app.get("/health")
def health():
    return jsonify({
        "status": "ok" if model is not None else "model_not_ready",
        "model_loaded": model is not None,
        "model_file": MODEL_PATH.name,
        "message": model_error,
    }), (200 if model is not None else 503)


@app.post("/predict")
def predict():
    if model is None:
        return render_template(
            "index.html",
            model_missing=True,
            model_error=model_error,
            error="The prediction model is unavailable. Train it and restart the app.",
        ), 503

    values = []
    for field in ("sepal_length", "sepal_width", "petal_length", "petal_width"):
        raw = request.form.get(field, "").strip()
        if not raw:
            return render_template(
                "index.html", model_missing=False,
                error=f"Please enter a value for {field.replace('_', ' ')}."
            )
        try:
            value = float(raw)
        except ValueError:
            return render_template(
                "index.html", model_missing=False,
                error=f"{field.replace('_', ' ').capitalize()} must be a number."
            )
        if not np.isfinite(value) or value <= 0 or value > 30:
            return render_template(
                "index.html", model_missing=False,
                error="Measurements must be finite, greater than 0 and at most 30 cm."
            )
        values.append(value)

    result = classify(values)
    return render_template(
        "index.html",
        model_missing=False,
        species=result["species"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        submitted_values=dict(zip(
            ("sepal_length", "sepal_width", "petal_length", "petal_width"), values
        )),
    )


if __name__ == "__main__":
    # Debug mode is disabled by default; set FLASK_DEBUG=1 for local debugging.
    debug = __import__("os").environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug)
