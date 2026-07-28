# Implementation Plan: Iris Flower Classification ML Application

## Overview

Implement a two-stage Python application: an offline training pipeline (`train_model.py`) that trains three classifiers and selects the best, and a Flask web application (`app.py`) that loads the serialised model and serves predictions through a responsive HTML interface. The project includes full property-based and unit/integration test coverage using `hypothesis` and `pytest`.

## Tasks

- [ ] 1. Project scaffolding and dependency definition
  - [ ] 1.1 Create project directory structure and static asset stubs
    - Create the following empty directories: `dataset/`, `images/`, `screenshots/`, `static/`, `templates/`
    - Create `.gitignore` that excludes `__pycache__/`, `*.pyc`, `iris_model.pkl`, `dataset/`, `images/`, `.env`
    - Create `LICENSE` file using the MIT license text
    - _Requirements: 7.1–7.11, 7.10, 7.11_

  - [ ] 1.2 Create `requirements.txt` with pinned dependencies
    - Include Flask, pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, hypothesis, pytest with compatible pinned versions
    - _Requirements: 8.1–8.9_

- [ ] 2. Training pipeline — data loading and splitting
  - [ ] 2.1 Implement `load_data()` and `split_data()` in `train_model.py`
    - Scaffold `train_model.py` with module docstring and PEP8-compliant imports
    - Implement `load_data()`: calls `sklearn.datasets.load_iris()`, builds a pandas DataFrame with column names `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `species` (string label), returns `(df, X, y, label_names)`
    - Implement `split_data(X, y)`: calls `train_test_split(X, y, test_size=0.2, random_state=42)`, returns `(X_train, X_test, y_train, y_test)`
    - Implement CSV export: `os.makedirs("dataset", exist_ok=True)` then `df.to_csv("dataset/iris_data.csv", index=False)`
    - _Requirements: 3.1–3.5_

- [ ] 3. Training pipeline — model training and selection
  - [ ] 3.1 Implement `train_models()`, `evaluate_models()`, `select_best()`, and `save_model()` in `train_model.py`
    - `train_models(X_train, y_train)`: instantiates `LogisticRegression(random_state=42)`, `DecisionTreeClassifier(random_state=42)`, `KNeighborsClassifier()`; fits each; returns list of `(name, fitted_model)`
    - `evaluate_models(models, X_test, y_test)`: calls `model.score(X_test, y_test)` for each; prints name + accuracy to stdout; returns list of `(name, model, accuracy)`
    - `select_best(evaluated)`: returns the tuple with maximum accuracy
    - `save_model(model, path="iris_model.pkl")`: calls `joblib.dump(model, path)` inside a `try/except OSError`
    - _Requirements: 1.1–1.8_

- [ ] 4. Training pipeline — visualisation generation
  - [ ] 4.1 Implement `generate_pair_plot()`, `generate_histogram()`, `generate_heatmap()`, `generate_confusion_matrix()` in `train_model.py`
    - Each function calls `os.makedirs("images", exist_ok=True)` (idempotent)
    - `generate_pair_plot(df, label_names)`: `sns.pairplot(df, hue="species")` → `plt.savefig("images/pair_plot.png", bbox_inches="tight")` → `plt.close()`
    - `generate_histogram(df)`: `df.drop("species", axis=1).hist(bins=20, figsize=(12,8))` → save to `images/histogram.png` → close
    - `generate_heatmap(df)`: `sns.heatmap(df.drop("species",axis=1).corr(), annot=True)` → save to `images/heatmap.png` → close
    - `generate_confusion_matrix(model, X_test, y_test, label_names)`: use `sklearn.metrics.ConfusionMatrixDisplay.from_estimator(...)` → save to `images/confusion_matrix.png` → close; wrap in `try/finally` with `plt.close()`
    - _Requirements: 2.1–2.10_

  - [ ] 4.2 Implement `main()` and wire all training steps in `train_model.py`
    - `main()` orchestrates: `load_data` → `split_data` → CSV save → `train_models` → `evaluate_models` → `select_best` → `save_model` → visualisations (pair plot, histogram, heatmap, confusion matrix) → print success message
    - Add `if __name__ == "__main__": main()` guard
    - _Requirements: 12.2, 12.5, 10.1–10.9_

- [ ] 5. Flask web application — core setup and routing
  - [ ] 5.1 Implement `load_model()`, GET `/`, and POST `/predict` in `app.py`
    - Scaffold `app.py` with module docstring and PEP8 imports: `Flask`, `render_template`, `request`, `joblib`, `numpy`, `os`
    - `load_model()`: attempts `joblib.load("iris_model.pkl")`; catches `FileNotFoundError` and returns `None`; called once at module level and stored in `model` global
    - Define `SPECIES_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]`
    - GET `/` route: `render_template("index.html")` — passes `model_missing=(model is None)`
    - POST `/predict` route: extract four fields via `request.form.get(...)`, wrap conversion in `try/except ValueError`, build `np.array([[...]], dtype=float)`, call `model.predict` and `model.predict_proba`, compute confidence as `round(float(max(proba)) * 100, 2)`, build `probabilities` dict with 4dp values, render template with results; on error render `error` variable
    - `if __name__ == "__main__": app.run(debug=True)` on port 5000
    - _Requirements: 4.1–4.8, 5.1–5.10, 12.3, 12.4, 12.6_

- [ ] 6. Frontend — HTML template
  - [ ] 6.1 Create `templates/index.html` with Jinja2 template
    - Responsive HTML5 page with `<title>Iris Flower Classification</title>`
    - Four labeled `<input type="number" step="0.1">` fields: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
    - `<button type="submit">Predict</button>` styled with rounded corners
    - Jinja2 conditionals: show error banner if `error` is set; show model-missing banner if `model_missing` is true
    - Result card: display `species` prominently, `confidence` as percentage, probability bars for all three species (loop over `probabilities`)
    - Link `static/style.css` in `<head>`
    - _Requirements: 6.1–6.12_

- [ ] 7. Frontend — CSS styling
  - [ ] 7.1 Create `static/style.css` with responsive styles
    - Gradient background on `body`
    - Card-style containers (`border-radius`, `box-shadow`) for the form and result sections
    - Responsive layout using CSS flexbox or grid; media query breakpoints for mobile
    - Button styles with rounded corners, hover transition (`transform`, `background-color`)
    - Professional typography: font-family, sizes, weights
    - Smooth transitions on interactive elements (`transition: all 0.3s ease`)
    - Probability bar visual using CSS width driven by inline style percentage in template
    - _Requirements: 6.4–6.8, 6.12_

- [ ] 8. Checkpoint — validate training pipeline and web app
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Property-based tests
  - [ ] 9.1 Create `tests/conftest.py` with shared fixtures
    - `loaded_model` fixture: loads `iris_model.pkl` via joblib; skips if file absent
    - `client` fixture: imports `app` from `app.py`, sets `app.config["TESTING"] = True`, returns `app.test_client()`
    - `valid_iris_feature_vectors()` Hypothesis composite strategy: four `floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False)`
    - Register Hypothesis profiles `"dev"` (max_examples=100) and `"ci"` (max_examples=200) in conftest
    - _Requirements: (testing infrastructure)_

  - [ ]* 9.2 Write property test for Property 1 — Prediction Output Shape Invariant
    - **Property 1: Prediction Output Shape Invariant**
    - **Validates: Requirements 5.4**
    - Tag comment: `# Feature: iris-flower-classification, Property 1: Prediction Output Shape Invariant`
    - Use `@given` with four independent `floats(min_value=0.1, max_value=20.0)`, `@settings(max_examples=100)`
    - Assert `model.predict([[sl, sw, pl, pw]])[0] in {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}`

  - [ ]* 9.3 Write property test for Property 2 — Confidence Score Range
    - **Property 2: Confidence Score Range**
    - **Validates: Requirements 5.6, 5.7**
    - Tag comment: `# Feature: iris-flower-classification, Property 2: Confidence Score Range`
    - Use `@given(feature_vector=valid_iris_feature_vectors())`, `@settings(max_examples=100)`
    - Assert `0.0 <= confidence <= 100.0` and `confidence == round(confidence, 2)`

  - [ ]* 9.4 Write property test for Property 3 — Probability Distribution Sums to One
    - **Property 3: Probability Distribution Sums to One**
    - **Validates: Requirements 5.5, 5.8**
    - Tag comment: `# Feature: iris-flower-classification, Property 3: Probability Distribution Sums to One`
    - Use `@given(feature_vector=valid_iris_feature_vectors())`, `@settings(max_examples=100)`
    - Assert `abs(sum(proba) - 1.0) < 1e-6` and `all(0.0 <= p <= 1.0 for p in proba)`

  - [ ]* 9.5 Write property test for Property 4 — Model Accuracy Threshold
    - **Property 4: Model Accuracy Exceeds Threshold on Test Set**
    - **Validates: Requirements 1.4, 1.5**
    - Tag comment: `# Feature: iris-flower-classification, Property 4: Model Accuracy Exceeds Threshold`
    - Example-based (deterministic); load iris, split with `random_state=42`, assert `model.score(X_test, y_test) > 0.90`

  - [ ]* 9.6 Write property test for Property 5 — Visualisation Files Are Created
    - **Property 5: Visualisation Files Are Created**
    - **Validates: Requirements 2.2, 2.4, 2.6, 2.8**
    - Tag comment: `# Feature: iris-flower-classification, Property 5: Visualisation Files Are Created`
    - Example-based; assert each of `images/pair_plot.png`, `images/histogram.png`, `images/heatmap.png`, `images/confusion_matrix.png` exists and `stat().st_size > 0`

  - [ ]* 9.7 Write property test for Property 6 — Input Validation Rejects Invalid
    - **Property 6: Input Validation Rejects Non-Numeric and Missing Values**
    - **Validates: Requirements 5.10**
    - Tag comment: `# Feature: iris-flower-classification, Property 6: Input Validation Rejects Invalid`
    - Use `@given(bad_value=one_of(text(), none(), just(""), just("abc")))`, `@settings(max_examples=100)`
    - POST to `/predict` with bad value in `sepal_length` field; assert `b"error"` in `response.data.lower()`

  - [ ]* 9.8 Write property test for Property 7 — Confidence-Probability Consistency
    - **Property 7: Confidence-Probability Consistency**
    - **Validates: Requirements 5.6**
    - Tag comment: `# Feature: iris-flower-classification, Property 7: Confidence-Probability Consistency`
    - Use `@given(feature_vector=valid_iris_feature_vectors())`, `@settings(max_examples=100)`
    - Assert `round(float(max(proba)) * 100, 2) == confidence` for the same proba array

- [ ] 10. Unit and integration tests
  - [ ] 10.1 Create `tests/test_app.py` with example-based unit tests
    - `test_model_file_exists`: assert `Path("iris_model.pkl").exists()`
    - `test_csv_created`: assert `Path("dataset/iris_data.csv").exists()` and `pd.read_csv(...).shape[0] == 150`
    - `test_home_route_status`: GET `/` returns 200 and `b"Iris Flower Classification"` in response data
    - `test_predict_setosa`: POST `/predict` with `sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2` → response contains `Iris-setosa`
    - `test_predict_virginica`: POST `/predict` with `sepal_length=6.3, sepal_width=3.3, petal_length=6.0, petal_width=2.5` → response contains `Iris-virginica`
    - `test_missing_field_error`: POST `/predict` with `sepal_length` omitted → `b"error"` in response
    - `test_non_numeric_error`: POST `/predict` with `sepal_length="abc"` → `b"error"` in response
    - `test_model_not_found_error`: temporarily rename pkl, start fresh app instance, GET `/` → error banner present; restore pkl
    - _Requirements: 4.3, 5.1–5.10_

  - [ ]* 10.2 Write integration tests in `tests/test_app.py`
    - `test_full_training_pipeline`: import and call `main()` from `train_model`; assert pkl, csv, and all four PNG files exist
    - `test_app_loads_trained_model`: after training, import app and assert `model is not None`
    - `test_prediction_roundtrip`: POST valid inputs to `/predict` via test client; assert response status 200 and all three species names appear in page (probability display)
    - _Requirements: 12.1–12.6_

- [ ] 11. README.md and GitHub assets
  - [ ] 11.1 Create `README.md` with full project documentation
    - Sections: project description, features, dataset, technologies, installation, how-to-run, screenshots (placeholders), folder structure, model performance, future improvements, GitHub badges, license, author
    - Use proper Markdown: headers, fenced code blocks, lists, badge shields
    - _Requirements: 9.1–9.14_

  - [ ] 11.2 Create GitHub repository assets file (`github_assets.md`)
    - Repository description (one-liner for GitHub About)
    - Repository tags list: `machine-learning`, `python`, `flask`, `scikit-learn`, `iris-dataset`
    - 10 professional commit messages
    - LinkedIn post template announcing the project
    - Professional internship submission description
    - GitHub banner image design suggestion
    - _Requirements: 11.1–11.6_

- [ ] 12. Final checkpoint — ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints provide incremental validation milestones
- Property tests (9.2–9.8) use `hypothesis` and validate universal correctness properties defined in the design
- Unit and integration tests (10.1–10.2) validate specific examples, edge cases, and end-to-end flows
- `iris_model.pkl` must exist (run `python train_model.py` first) before the Flask app or most tests will run correctly
- The `tests/conftest.py` fixtures skip gracefully when `iris_model.pkl` is absent, so tests can be authored before training

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["3.1"] },
    { "id": 3, "tasks": ["4.1"] },
    { "id": 4, "tasks": ["4.2"] },
    { "id": 5, "tasks": ["5.1", "9.1"] },
    { "id": 6, "tasks": ["6.1", "7.1"] },
    { "id": 7, "tasks": ["9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "9.8", "10.1"] },
    { "id": 8, "tasks": ["10.2", "11.1", "11.2"] }
  ]
}
```
