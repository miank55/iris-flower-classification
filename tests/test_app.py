"""
tests/test_app.py
=================
Unit tests and integration tests for the Iris Flower Classification
ML Application.

Tests
-----
Unit tests:
  - test_model_file_exists
  - test_csv_created
  - test_home_route_status
  - test_predict_setosa
  - test_predict_virginica
  - test_missing_field_error
  - test_non_numeric_error
  - test_model_not_found_error

Integration tests:
  - test_full_training_pipeline
  - test_app_loads_trained_model
  - test_prediction_roundtrip

Run with::

    pytest tests/test_app.py -v
"""

import importlib
import os
import shutil
import sys

import joblib
import pandas as pd
import pytest

# Ensure project root is importable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestModelFileExists:
    """Verify that iris_model.pkl was created by the training pipeline."""

    def test_model_file_exists(self):
        """``iris_model.pkl`` SHALL exist in the project root after training.

        Requirements: 1.6
        """
        pkl_path = os.path.join(PROJECT_ROOT, "iris_model.pkl")
        if not os.path.exists(pkl_path):
            pytest.skip(
                "'iris_model.pkl' not found. "
                "Run 'python train_model.py' first."
            )
        assert os.path.isfile(pkl_path), (
            f"'{pkl_path}' exists but is not a regular file."
        )
        assert os.path.getsize(pkl_path) > 0, (
            f"'{pkl_path}' exists but is empty."
        )


class TestCSVCreated:
    """Verify the dataset CSV artefact."""

    def test_csv_created(self):
        """``dataset/iris_data.csv`` SHALL exist and contain 150 rows.

        Requirements: 3.3, 3.4
        """
        csv_path = os.path.join(PROJECT_ROOT, "dataset", "iris_data.csv")
        if not os.path.exists(csv_path):
            pytest.skip(
                "'dataset/iris_data.csv' not found. "
                "Run 'python train_model.py' first."
            )

        df = pd.read_csv(csv_path)
        assert df.shape[0] == 150, (
            f"Expected 150 rows in iris_data.csv, got {df.shape[0]}."
        )
        expected_columns = {
            "sepal_length", "sepal_width", "petal_length",
            "petal_width", "species",
        }
        assert expected_columns.issubset(set(df.columns)), (
            f"Missing columns. Expected {expected_columns}, "
            f"got {set(df.columns)}."
        )


class TestHomeRoute:
    """Tests for the GET / homepage route."""

    def test_home_route_status(self, client):
        """GET ``/`` SHALL return HTTP 200.

        Requirements: 4.4
        """
        response = client.get("/")
        assert response.status_code == 200, (
            f"Expected 200 from GET /, got {response.status_code}."
        )

    def test_home_route_contains_title(self, client):
        """Homepage HTML SHALL contain the application title.

        Requirements: 6.1
        """
        response = client.get("/")
        assert b"Iris Flower Classification" in response.data, (
            "Expected 'Iris Flower Classification' in homepage HTML."
        )

    def test_home_route_has_form(self, client):
        """Homepage SHALL contain the four measurement input fields.

        Requirements: 6.2
        """
        data = response = client.get("/").data
        for field in [b"sepal_length", b"sepal_width",
                      b"petal_length", b"petal_width"]:
            assert field in data, (
                f"Expected input field '{field.decode()}' in homepage HTML."
            )


class TestPredictionRouteValid:
    """Tests for POST /predict with valid inputs."""

    def test_predict_setosa(self, client):
        """Canonical Setosa values SHALL return 'Iris-setosa' in the response.

        Requirements: 5.1–5.9
        """
        response = client.post(
            "/predict",
            data={
                "sepal_length": "5.1",
                "sepal_width": "3.5",
                "petal_length": "1.4",
                "petal_width": "0.2",
            },
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}."
        )
        assert b"Iris-setosa" in response.data, (
            "Expected 'Iris-setosa' in prediction response for canonical "
            "Setosa inputs (5.1, 3.5, 1.4, 0.2)."
        )

    def test_predict_virginica(self, client):
        """Canonical Virginica values SHALL return 'Iris-virginica' in response.

        Requirements: 5.1–5.9
        """
        response = client.post(
            "/predict",
            data={
                "sepal_length": "6.3",
                "sepal_width": "3.3",
                "petal_length": "6.0",
                "petal_width": "2.5",
            },
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}."
        )
        assert b"Iris-virginica" in response.data, (
            "Expected 'Iris-virginica' in prediction response for canonical "
            "Virginica inputs (6.3, 3.3, 6.0, 2.5)."
        )

    def test_predict_versicolor(self, client):
        """Canonical Versicolor values SHALL return 'Iris-versicolor' in response.

        Requirements: 5.1–5.9
        """
        response = client.post(
            "/predict",
            data={
                "sepal_length": "6.4",
                "sepal_width": "3.2",
                "petal_length": "4.5",
                "petal_width": "1.5",
            },
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}."
        )
        assert b"Iris-versicolor" in response.data, (
            "Expected 'Iris-versicolor' in prediction response for canonical "
            "Versicolor inputs (6.4, 3.2, 4.5, 1.5)."
        )

    def test_predict_response_contains_confidence(self, client):
        """A valid prediction response SHALL contain a confidence percentage.

        Requirements: 5.6, 5.7, 6.10
        """
        response = client.post(
            "/predict",
            data={
                "sepal_length": "5.1",
                "sepal_width": "3.5",
                "petal_length": "1.4",
                "petal_width": "0.2",
            },
        )
        assert b"%" in response.data or b"Confidence" in response.data, (
            "Expected confidence percentage in valid prediction response."
        )


class TestPredictionRouteInvalid:
    """Tests for POST /predict with invalid or missing inputs."""

    def test_missing_field_error(self, client):
        """POST with ``sepal_length`` omitted SHALL return an error.

        Requirements: 5.10
        """
        response = client.post(
            "/predict",
            data={
                # sepal_length intentionally omitted
                "sepal_width": "3.0",
                "petal_length": "1.5",
                "petal_width": "0.2",
            },
        )
        assert response.status_code == 200, (
            f"Expected 200 (error page), got {response.status_code}."
        )
        assert b"error" in response.data.lower(), (
            "Expected 'error' in response when sepal_length is missing."
        )

    def test_non_numeric_error(self, client):
        """POST with ``sepal_length='abc'`` SHALL return an error.

        Requirements: 5.10
        """
        response = client.post(
            "/predict",
            data={
                "sepal_length": "abc",
                "sepal_width": "3.0",
                "petal_length": "1.5",
                "petal_width": "0.2",
            },
        )
        assert response.status_code == 200, (
            f"Expected 200 (error page), got {response.status_code}."
        )
        assert b"error" in response.data.lower(), (
            "Expected 'error' in response when sepal_length is 'abc'."
        )

    def test_empty_string_field_error(self, client):
        """POST with an empty string field SHALL return an error.

        Requirements: 5.10
        """
        response = client.post(
            "/predict",
            data={
                "sepal_length": "",
                "sepal_width": "3.0",
                "petal_length": "1.5",
                "petal_width": "0.2",
            },
        )
        assert b"error" in response.data.lower(), (
            "Expected 'error' in response when sepal_length is empty string."
        )

    def test_all_fields_missing_error(self, client):
        """POST with no fields SHALL return an error.

        Requirements: 5.10
        """
        response = client.post("/predict", data={})
        assert b"error" in response.data.lower(), (
            "Expected 'error' in response when all fields are missing."
        )

    def test_non_numeric_petal_width_error(self, client):
        """POST with non-numeric ``petal_width`` SHALL return an error.

        Requirements: 5.10
        """
        response = client.post(
            "/predict",
            data={
                "sepal_length": "5.1",
                "sepal_width": "3.5",
                "petal_length": "1.4",
                "petal_width": "not_a_float",
            },
        )
        assert b"error" in response.data.lower(), (
            "Expected 'error' in response for non-numeric petal_width."
        )


class TestModelNotFound:
    """Test app behaviour when iris_model.pkl is absent."""

    def test_model_not_found_error(self, tmp_path, monkeypatch):
        """When ``iris_model.pkl`` is absent, GET ``/`` SHALL show error banner.

        Requirements: 4.3
        """
        # Import a fresh copy of app with model forced to None.
        import app as flask_app  # noqa: PLC0415

        # Temporarily patch the module-level ``model`` to None.
        original_model = flask_app.model
        flask_app.model = None

        try:
            flask_app.app.config["TESTING"] = True
            with flask_app.app.test_client() as patched_client:
                response = patched_client.get("/")
                data = response.data.lower()
                # The template shows either "model not found" or "train_model.py".
                assert (
                    b"model not found" in data
                    or b"train_model" in data
                    or b"model_missing" in data
                    or b"error" in data
                ), (
                    "Expected model-missing warning when model is None, "
                    f"got: {response.data[:300]}"
                )
        finally:
            # Restore the original model so other tests are not affected.
            flask_app.model = original_model


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_training_pipeline(self):
        """Running ``main()`` from ``train_model`` SHALL create all artefacts.

        Requirements: 12.2, 12.5
        """
        import train_model  # noqa: PLC0415

        # Run the full pipeline.
        train_model.main()

        # Verify the model file.
        pkl_path = os.path.join(PROJECT_ROOT, "iris_model.pkl")
        assert os.path.isfile(pkl_path), "iris_model.pkl not created."
        assert os.path.getsize(pkl_path) > 0, "iris_model.pkl is empty."

        # Verify the CSV file.
        csv_path = os.path.join(PROJECT_ROOT, "dataset", "iris_data.csv")
        assert os.path.isfile(csv_path), "dataset/iris_data.csv not created."
        df = pd.read_csv(csv_path)
        assert df.shape[0] == 150, (
            f"Expected 150 rows in CSV, got {df.shape[0]}."
        )

        # Verify all four PNG visualisations.
        images_dir = os.path.join(PROJECT_ROOT, "images")
        for fname in ["pair_plot.png", "histogram.png",
                      "heatmap.png", "confusion_matrix.png"]:
            path = os.path.join(images_dir, fname)
            assert os.path.isfile(path), f"'{fname}' not created."
            assert os.path.getsize(path) > 0, f"'{fname}' is empty."

    def test_app_loads_trained_model(self):
        """After training, importing ``app`` SHALL result in a non-None model.

        Requirements: 4.2, 12.3
        """
        pkl_path = os.path.join(PROJECT_ROOT, "iris_model.pkl")
        if not os.path.exists(pkl_path):
            pytest.skip(
                "'iris_model.pkl' not found. "
                "Run 'python train_model.py' first."
            )

        # Force reimport so load_model() is called fresh.
        import app as flask_app  # noqa: PLC0415

        # Reload to pick up any changes.
        importlib.reload(flask_app)

        assert flask_app.model is not None, (
            "Expected app.model to be a fitted estimator, got None. "
            "Ensure 'iris_model.pkl' exists."
        )

    def test_prediction_roundtrip(self, client):
        """POST valid inputs SHALL return 200 with all three species in HTML.

        The result page should list probability values for all three species.

        Requirements: 12.1–12.6, 5.5, 6.11
        """
        pkl_path = os.path.join(PROJECT_ROOT, "iris_model.pkl")
        if not os.path.exists(pkl_path):
            pytest.skip(
                "'iris_model.pkl' not found. "
                "Run 'python train_model.py' first."
            )

        response = client.post(
            "/predict",
            data={
                "sepal_length": "5.1",
                "sepal_width": "3.5",
                "petal_length": "1.4",
                "petal_width": "0.2",
            },
        )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}."
        )

        data = response.data
        # All three species names should appear in the probability display.
        assert b"Iris-setosa" in data, "Iris-setosa not in roundtrip response."
        assert b"Iris-versicolor" in data, (
            "Iris-versicolor not in roundtrip response."
        )
        assert b"Iris-virginica" in data, (
            "Iris-virginica not in roundtrip response."
        )

    def test_training_pipeline_produces_loadable_model(self):
        """The serialised model SHALL be loadable and callable by joblib.

        Requirements: 1.6
        """
        pkl_path = os.path.join(PROJECT_ROOT, "iris_model.pkl")
        if not os.path.exists(pkl_path):
            pytest.skip(
                "'iris_model.pkl' not found. "
                "Run 'python train_model.py' first."
            )

        import numpy as np  # noqa: PLC0415

        loaded = joblib.load(pkl_path)

        # Must implement predict and predict_proba.
        assert hasattr(loaded, "predict"), "Model must have .predict()"
        assert hasattr(loaded, "predict_proba"), (
            "Model must have .predict_proba()"
        )

        # Must return a valid prediction on a sample input.
        sample = np.array([[5.1, 3.5, 1.4, 0.2]])
        pred = loaded.predict(sample)
        proba = loaded.predict_proba(sample)

        assert len(pred) == 1, "Expected 1 prediction for 1 sample."
        assert proba.shape == (1, 3), (
            f"Expected proba shape (1, 3), got {proba.shape}."
        )
