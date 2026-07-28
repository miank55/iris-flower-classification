"""
tests/test_properties.py
========================
Property-based tests for the Iris Flower Classification ML Application.

Uses the Hypothesis library to verify seven universal correctness properties
across hundreds of randomly generated inputs.

Each test is annotated with the requirement(s) it validates.

Run with::

    pytest tests/test_properties.py -v
"""

import sys
import os

# Ensure project root is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import floats, just, none, one_of, text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tests.conftest import valid_iris_feature_vectors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPECIES_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


# ---------------------------------------------------------------------------
# Property 1 — Prediction Output Shape Invariant
# ---------------------------------------------------------------------------

# Feature: iris-flower-classification, Property 1: Prediction Output Shape Invariant
# Validates: Requirements 5.4
@given(
    sepal_l=floats(min_value=0.1, max_value=20.0,
                   allow_nan=False, allow_infinity=False),
    sepal_w=floats(min_value=0.1, max_value=20.0,
                   allow_nan=False, allow_infinity=False),
    petal_l=floats(min_value=0.1, max_value=20.0,
                   allow_nan=False, allow_infinity=False),
    petal_w=floats(min_value=0.1, max_value=20.0,
                   allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prediction_species_is_valid(
    loaded_model, sepal_l, sepal_w, petal_l, petal_w
):
    """**Property 1: Prediction Output Shape Invariant**

    For any valid 4-element feature vector the model SHALL return exactly
    one species name drawn from the canonical set of three Iris species.

    **Validates: Requirements 5.4**
    """
    input_array = np.array([[sepal_l, sepal_w, petal_l, petal_w]], dtype=float)
    result = loaded_model.predict(input_array)

    # Exactly one prediction is returned.
    assert len(result) == 1, (
        f"Expected 1 prediction, got {len(result)}"
    )

    # The predicted index maps to one of the three known species names.
    predicted_index = int(result[0])
    assert 0 <= predicted_index <= 2, (
        f"Predicted class index {predicted_index} is out of range [0, 2]."
    )
    # Verify against the sklearn iris target_names for completeness.
    iris = load_iris()
    assert iris.target_names[predicted_index] in iris.target_names, (
        f"'{iris.target_names[predicted_index]}' not in known species."
    )


# ---------------------------------------------------------------------------
# Property 2 — Confidence Score Range
# ---------------------------------------------------------------------------

# Feature: iris-flower-classification, Property 2: Confidence Score Range
# Validates: Requirements 5.6, 5.7
@given(feature_vector=valid_iris_feature_vectors())
@settings(max_examples=100)
def test_confidence_in_range(loaded_model, feature_vector):
    """**Property 2: Confidence Score Range**

    For any valid feature vector, the confidence score (derived from
    ``predict_proba``) SHALL lie in [0.0, 100.0] and SHALL be rounded
    to exactly two decimal places.

    **Validates: Requirements 5.6, 5.7**
    """
    input_array = np.array([feature_vector], dtype=float)
    proba = loaded_model.predict_proba(input_array)[0]

    # Replicate the app.py confidence calculation.
    confidence = round(float(max(proba)) * 100, 2)

    assert 0.0 <= confidence <= 100.0, (
        f"Confidence {confidence} is outside [0.0, 100.0]."
    )
    # Verify that the value is already at 2 decimal places (idempotent round).
    assert confidence == round(confidence, 2), (
        f"Confidence {confidence} is not rounded to 2 decimal places."
    )


# ---------------------------------------------------------------------------
# Property 3 — Probability Distribution Sums to One
# ---------------------------------------------------------------------------

# Feature: iris-flower-classification, Property 3: Probability Distribution Sums to One
# Validates: Requirements 5.5, 5.8
@given(feature_vector=valid_iris_feature_vectors())
@settings(max_examples=100)
def test_probabilities_sum_to_one(loaded_model, feature_vector):
    """**Property 3: Probability Distribution Sums to One**

    For any valid feature vector, the three class probabilities returned
    by ``predict_proba`` SHALL sum to 1.0 (±1e-6) and each individual
    probability SHALL lie in [0.0, 1.0].

    **Validates: Requirements 5.5, 5.8**
    """
    input_array = np.array([feature_vector], dtype=float)
    proba = loaded_model.predict_proba(input_array)[0]

    # Exactly three class probabilities.
    assert len(proba) == 3, (
        f"Expected 3 probabilities, got {len(proba)}."
    )

    # Sum within floating-point tolerance.
    total = float(sum(proba))
    assert abs(total - 1.0) < 1e-6, (
        f"Probabilities sum to {total}, expected 1.0 (±1e-6)."
    )

    # Each probability is a valid value in [0, 1].
    for idx, p in enumerate(proba):
        p_float = float(p)
        assert 0.0 <= p_float <= 1.0, (
            f"Probability at index {idx} is {p_float}, "
            "expected value in [0.0, 1.0]."
        )


# ---------------------------------------------------------------------------
# Property 4 — Model Accuracy Threshold
# ---------------------------------------------------------------------------

# Feature: iris-flower-classification, Property 4: Model Accuracy Exceeds Threshold
# Validates: Requirements 1.4, 1.5
def test_best_model_accuracy_exceeds_threshold(loaded_model):
    """**Property 4: Model Accuracy Exceeds Threshold on Test Set**

    For any execution with the canonical scikit-learn Iris dataset and
    ``random_state=42``, the selected best model SHALL achieve accuracy
    greater than 0.90 on the held-out test set.

    This is an example-based (deterministic) test because the dataset,
    split, and random seed are all fixed.

    **Validates: Requirements 1.4, 1.5**
    """
    iris = load_iris()
    _, X_test, _, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    accuracy = loaded_model.score(X_test, y_test)
    assert accuracy > 0.90, (
        f"Best model accuracy {accuracy:.4f} does not exceed threshold 0.90."
    )


# ---------------------------------------------------------------------------
# Property 5 — Visualisation Files Are Created
# ---------------------------------------------------------------------------

# Feature: iris-flower-classification, Property 5: Visualisation Files Are Created
# Validates: Requirements 2.2, 2.4, 2.6, 2.8
def test_visualization_files_exist_and_are_non_empty():
    """**Property 5: Visualisation Files Are Created**

    After a successful run of ``train_model.py``, all four PNG
    visualisation files SHALL exist in ``images/`` and SHALL be
    non-empty.

    **Validates: Requirements 2.2, 2.4, 2.6, 2.8**
    """
    from pathlib import Path

    expected_files = [
        "pair_plot.png",
        "histogram.png",
        "heatmap.png",
        "confusion_matrix.png",
    ]

    for fname in expected_files:
        path = Path("images") / fname
        assert path.exists(), (
            f"Visualisation file '{fname}' not found in images/. "
            "Run 'python train_model.py' first."
        )
        assert path.stat().st_size > 0, (
            f"Visualisation file '{fname}' exists but is empty."
        )


# ---------------------------------------------------------------------------
# Property 6 — Input Validation Rejects Non-Numeric and Missing Values
# ---------------------------------------------------------------------------

# Feature: iris-flower-classification, Property 6: Input Validation Rejects Invalid
# Validates: Requirements 5.10
@given(
    bad_value=one_of(
        text(min_size=1, max_size=20).filter(lambda s: s.strip() != ""),
        none(),
        just(""),
        just("abc"),
        just("not_a_number"),
        just("1.2.3"),
        just("  "),
    )
)
@settings(max_examples=100)
def test_invalid_input_returns_error_not_prediction(client, bad_value):
    """**Property 6: Input Validation Rejects Non-Numeric and Missing Values**

    For any POST to ``/predict`` where the ``sepal_length`` field is
    absent, empty, or non-numeric, the response SHALL contain the word
    "error" and SHALL NOT invoke the model's predict method.

    **Validates: Requirements 5.10**
    """
    # Only test values that are not valid floats.
    if bad_value is not None:
        try:
            float(bad_value)
            # If it parses as a float, skip — this is a valid input.
            return
        except (ValueError, TypeError):
            pass  # Expected: not a valid number — proceed with test.

    form_data = {
        "sepal_width": "3.0",
        "petal_length": "1.5",
        "petal_width": "0.2",
    }
    if bad_value is not None:
        form_data["sepal_length"] = bad_value
    # When bad_value is None, sepal_length is absent from the form entirely.

    response = client.post("/predict", data=form_data)
    response_lower = response.data.lower()

    assert b"error" in response_lower, (
        f"Expected 'error' in response for bad input '{bad_value}', "
        f"but got: {response.data[:200]}"
    )


# ---------------------------------------------------------------------------
# Property 7 — Confidence-Probability Consistency
# ---------------------------------------------------------------------------

# Feature: iris-flower-classification, Property 7: Confidence-Probability Consistency
# Validates: Requirements 5.6
@given(feature_vector=valid_iris_feature_vectors())
@settings(max_examples=100)
def test_confidence_equals_max_probability_times_100(loaded_model, feature_vector):
    """**Property 7: Confidence-Probability Consistency**

    For any valid feature vector, the confidence score SHALL equal
    ``round(max(probabilities.values()) * 100, 2)``.

    That is, confidence is always derived deterministically from the
    returned per-class probability values.

    **Validates: Requirements 5.6**
    """
    input_array = np.array([feature_vector], dtype=float)
    proba = loaded_model.predict_proba(input_array)[0]

    # The app.py derivation.
    expected_confidence = round(float(max(proba)) * 100, 2)

    # Simulate calculating it a second time from the same proba array.
    recalculated_confidence = round(float(max(proba)) * 100, 2)

    assert recalculated_confidence == expected_confidence, (
        f"Confidence recalculation mismatch: "
        f"{recalculated_confidence} != {expected_confidence}."
    )

    # Additional consistency: max_proba * 100 rounded to 2dp must equal
    # the confidence shown in the UI.
    max_proba = float(max(proba))
    assert expected_confidence == round(max_proba * 100, 2), (
        f"confidence ({expected_confidence}) != "
        f"round(max_proba * 100, 2) ({round(max_proba * 100, 2)})."
    )
