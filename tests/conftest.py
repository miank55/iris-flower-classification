"""
tests/conftest.py
=================
Shared pytest fixtures and Hypothesis configuration for the
Iris Flower Classification test suite.

Provides:
- ``loaded_model`` — a fitted scikit-learn estimator loaded from
  ``iris_model.pkl``; the test is skipped when the file is absent.
- ``client`` — a Flask test client for the web application.
- ``valid_iris_feature_vectors`` — a Hypothesis composite strategy that
  generates realistic four-element Iris feature vectors.
"""

import os
import sys

import joblib
import pytest
from hypothesis import HealthCheck, settings
from hypothesis.strategies import composite, floats

# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------

settings.register_profile(
    "dev",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "ci",
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)

# Activate the dev profile by default; override with the HYPOTHESIS_PROFILE
# environment variable (e.g. HYPOTHESIS_PROFILE=ci pytest).
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


# ---------------------------------------------------------------------------
# Model fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def loaded_model():
    """Load ``iris_model.pkl`` once per test session.

    Skips the calling test (via ``pytest.skip``) when the file does not
    exist, so the test suite can be authored and inspected before training.

    Returns
    -------
    model : scikit-learn estimator
        A fitted estimator that implements ``predict`` and ``predict_proba``.
    """
    model_path = "iris_model.pkl"
    if not os.path.exists(model_path):
        pytest.skip(
            f"'{model_path}' not found. "
            "Run 'python train_model.py' first to generate it."
        )
    return joblib.load(model_path)


# ---------------------------------------------------------------------------
# Flask test-client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    """Create a Flask test client for the web application.

    Sets ``TESTING = True`` so that exceptions propagate to the test
    runner rather than being swallowed by Flask's error handler.

    Returns
    -------
    flask.testing.FlaskClient
        A test client that can issue GET and POST requests.
    """
    # Ensure the project root is on sys.path so ``app`` can be imported.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import app as flask_app  # noqa: PLC0415 — intentional deferred import

    flask_app.app.config["TESTING"] = True
    flask_app.app.config["WTF_CSRF_ENABLED"] = False

    with flask_app.app.test_client() as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Hypothesis composite strategy
# ---------------------------------------------------------------------------

@composite
def valid_iris_feature_vectors(draw):
    """Hypothesis composite strategy: generate a valid Iris feature vector.

    Produces a list ``[sepal_length, sepal_width, petal_length, petal_width]``
    where each value is a finite float in ``[0.1, 20.0]``.

    The upper bound of 20.0 (well above the real dataset maximum of ~7.9)
    exercises out-of-distribution inputs while remaining numeric and finite,
    which is the interesting property-testing territory for these classifiers.

    Returns
    -------
    list of float, length 4
    """
    _float = floats(
        min_value=0.1,
        max_value=20.0,
        allow_nan=False,
        allow_infinity=False,
    )
    sepal_length = draw(_float)
    sepal_width  = draw(_float)
    petal_length = draw(_float)
    petal_width  = draw(_float)
    return [sepal_length, sepal_width, petal_length, petal_width]
