"""
app.py
======
Iris Flower Classification — Flask Web Application.

This module serves a two-route HTTP interface:

* ``GET /``       — renders the prediction form (index.html).
* ``POST /predict`` — validates the four numeric inputs, invokes the
  serialised scikit-learn model, and renders the Classification_Result.

Run the application::

    python app.py

The application starts on http://localhost:5000 in debug mode.
The serialised model (``iris_model.pkl``) must exist before starting;
run ``python train_model.py`` first if it is absent.
"""

import os
import sys

import joblib
import numpy as np
from flask import Flask, render_template, request

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Ordered species names that match the model's class indices 0, 1, 2.
SPECIES_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

#: Path to the serialised joblib model file.
MODEL_PATH = "iris_model.pkl"

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load the serialised scikit-learn model from disk.

    Returns
    -------
    model : estimator or None
        The fitted model if the file exists and loads successfully.
        Returns ``None`` if the file is absent or cannot be loaded.
    """
    if not os.path.exists(MODEL_PATH):
        print(
            f"WARNING: '{MODEL_PATH}' not found. "
            "Run 'python train_model.py' first.",
            file=sys.stderr,
        )
        return None

    try:
        loaded = joblib.load(MODEL_PATH)
        print(f"Model loaded from '{MODEL_PATH}'.")
        return loaded
    except (OSError, EOFError, Exception) as exc:  # noqa: BLE001
        print(f"ERROR: could not load model: {exc}", file=sys.stderr)
        return None


# Load the model once at module import time.
model = load_model()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Render the prediction form homepage.

    Template variables
    ------------------
    model_missing : bool
        ``True`` when the model file could not be loaded at startup, so
        that the template can display an instructional error banner.
    """
    return render_template("index.html", model_missing=(model is None))


@app.route("/predict", methods=["POST"])
def predict():
    """Handle a prediction request from the HTML form.

    Reads ``sepal_length``, ``sepal_width``, ``petal_length``, and
    ``petal_width`` from the POST body, validates them, runs inference,
    and renders the result page.

    Template variables on success
    ------------------------------
    species : str
        Predicted Iris species name from ``SPECIES_NAMES``.
    confidence : float
        ``round(max(probabilities) * 100, 2)`` — confidence percentage.
    probabilities : dict[str, float]
        Mapping of species name → probability (rounded to 4 d.p.).

    Template variables on error
    ----------------------------
    error : str
        Human-readable error message shown in the error banner.
    model_missing : bool
        ``True`` when the model is unavailable.
    """
    # Reject requests when the model was never loaded.
    if model is None:
        return render_template(
            "index.html",
            model_missing=True,
            error="Model not loaded. Please run train_model.py first.",
        )

    # --- Input validation --------------------------------------------------
    field_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    values = []

    for field in field_names:
        raw = request.form.get(field)

        # Check for missing / empty field.
        if raw is None or raw.strip() == "":
            return render_template(
                "index.html",
                model_missing=False,
                error=(
                    f"Missing input: '{field}'. "
                    "Please fill in all four measurement fields."
                ),
            )

        # Attempt numeric conversion.
        try:
            numeric_value = float(raw)
        except ValueError:
            return render_template(
                "index.html",
                model_missing=False,
                error=(
                    f"Invalid value for '{field}': '{raw}' is not a number. "
                    "Please enter numeric measurements in centimetres."
                ),
            )

        values.append(numeric_value)

    # --- Inference ----------------------------------------------------------
    try:
        input_array = np.array([values], dtype=float)  # shape (1, 4)

        # Predicted class index → species name.
        pred_index = int(model.predict(input_array)[0])
        species = SPECIES_NAMES[pred_index]

        # Per-class probabilities (array of 3 values summing to 1.0).
        proba = model.predict_proba(input_array)[0]

        # Confidence: maximum probability as a percentage, 2 d.p.
        confidence = round(float(max(proba)) * 100, 2)

        # Build a display-friendly probability dict (4 d.p. per class).
        probabilities = {
            name: round(float(p), 4)
            for name, p in zip(SPECIES_NAMES, proba)
        }

    except Exception as exc:  # noqa: BLE001
        print(f"ERROR during prediction: {exc}", file=sys.stderr)
        return render_template(
            "index.html",
            model_missing=False,
            error=(
                "An unexpected error occurred during prediction. "
                "Please check your inputs and try again."
            ),
        )

    return render_template(
        "index.html",
        model_missing=False,
        species=species,
        confidence=confidence,
        probabilities=probabilities,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
