# 🌸 Iris Flower Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.3-lightgrey?logo=flask&logoColor=black)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hypothesis](https://img.shields.io/badge/Tested%20with-Hypothesis-purple)](https://hypothesis.readthedocs.io/)

A full-stack machine learning web application that classifies Iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — based on four sepal and petal measurements. The project trains three classifiers, automatically selects the most accurate model, and serves predictions through a polished Flask web interface.

---

## ✨ Features

- **Multi-model training** — Logistic Regression, Decision Tree, and K-Nearest Neighbours trained in one command
- **Automatic model selection** — the best-performing model is selected by test-set accuracy and serialised to disk
- **Interactive web UI** — responsive form with real-time probability bars and confidence scores
- **Four visualisation artefacts** — pair plot, histogram, correlation heatmap, confusion matrix
- **Property-based tests** — seven Hypothesis properties validating correctness across thousands of random inputs
- **One-command setup** — `pip install -r requirements.txt` → `python train_model.py` → `python app.py`
- **Reproducible results** — `random_state=42` throughout ensures identical outputs on every run

---

## 📊 Dataset

The [UCI Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris) (Fisher, 1936) is the canonical benchmark for multiclass classification.

| Property      | Value                                    |
|---------------|------------------------------------------|
| Samples       | 150 (50 per class)                       |
| Features      | 4 numeric (sepal/petal length and width) |
| Classes       | 3 (Setosa, Versicolor, Virginica)        |
| Missing values | None                                    |
| Source        | `sklearn.datasets.load_iris()`           |

**Feature ranges (approximate):**

| Feature        | Min  | Max  |
|----------------|------|------|
| Sepal Length   | 4.3  | 7.9  |
| Sepal Width    | 2.0  | 4.4  |
| Petal Length   | 1.0  | 6.9  |
| Petal Width    | 0.1  | 2.5  |

---

## 🛠️ Technologies

| Layer          | Library / Tool             | Version  |
|----------------|----------------------------|----------|
| ML framework   | scikit-learn               | 1.5.0    |
| Data handling  | pandas, numpy              | 2.2.2 / 1.26.4 |
| Visualisation  | matplotlib, seaborn        | 3.9.0 / 0.13.2 |
| Web framework  | Flask                      | 3.0.3    |
| Serialisation  | joblib                     | 1.4.2    |
| Testing        | pytest, hypothesis         | 8.2.2 / 6.103.1 |
| Language       | Python                     | 3.10+    |

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- `pip` (comes with Python)

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/your-username/iris-flower-classification.git
cd iris-flower-classification
```

**2. (Recommended) Create a virtual environment**

```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1 — Train the model

```bash
python train_model.py
```

This will:
- Load the Iris dataset from scikit-learn
- Train Logistic Regression, Decision Tree, and KNN models
- Select the best model by test-set accuracy
- Save `iris_model.pkl` to the project root
- Save `dataset/iris_data.csv`
- Generate four visualisation PNGs in `images/`

Expected output:
```
====================================================
  Iris Flower Classification — Training Pipeline
====================================================

[1/8] Loading Iris dataset …
[2/8] Splitting data (80 / 20, random_state=42) …
...
  Best model : Logistic Regression  (accuracy=1.0000)
====================================================
  Training complete!
====================================================
```

### Step 2 — Start the web application

```bash
python app.py
```

Open your browser at **http://localhost:5000**

### Step 3 — Make a prediction

Enter the four measurements in the form and click **Predict Species**.

**Example inputs:**

| Species           | Sepal L | Sepal W | Petal L | Petal W |
|-------------------|---------|---------|---------|---------|
| Iris setosa       | 5.1     | 3.5     | 1.4     | 0.2     |
| Iris versicolor   | 6.4     | 3.2     | 4.5     | 1.5     |
| Iris virginica    | 6.3     | 3.3     | 6.0     | 2.5     |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only property-based tests
pytest tests/test_properties.py -v

# Run only unit/integration tests
pytest tests/test_app.py -v

# Run with coverage report
pytest tests/ --tb=short
```

> **Note:** Run `python train_model.py` before running tests. Tests that require `iris_model.pkl` will be automatically skipped if the file is absent.

---

## 📸 Screenshots

### Home Page — Input Form

> *(screenshot of the prediction form with gradient background)*
>
> `screenshots/home_page.png`

### Prediction Result — Iris setosa

> *(screenshot showing the result card with species badge, confidence score, and probability bars)*
>
> `screenshots/result_setosa.png`

### Pair Plot Visualisation

> *(screenshot of `images/pair_plot.png` — scatter matrix coloured by species)*
>
> `screenshots/pair_plot.png`

---

## 📁 Folder Structure

```
iris-flower-classification/
├── app.py                      # Flask web application
├── train_model.py              # ML training pipeline
├── requirements.txt            # Pinned Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT licence
├── .gitignore                  # Git exclusions
│
├── templates/
│   └── index.html              # Jinja2 HTML template
│
├── static/
│   └── style.css               # Global CSS styles
│
├── tests/
│   ├── conftest.py             # pytest fixtures & Hypothesis profiles
│   ├── test_properties.py      # 7 property-based tests (Hypothesis)
│   └── test_app.py             # Unit & integration tests
│
├── dataset/                    # Generated: iris_data.csv
├── images/                     # Generated: PNG visualisations
└── screenshots/                # Manual screenshots for README
```

---

## 📈 Model Performance

All three models are evaluated on the same 30-sample test set (20% of 150 samples, `random_state=42`).

| Model                  | Typical Accuracy |
|------------------------|-----------------|
| Logistic Regression    | ~97 – 100%      |
| Decision Tree          | ~93 – 100%      |
| K-Nearest Neighbours   | ~97 – 100%      |

The model with the highest accuracy is automatically selected and saved. With `random_state=42`, Logistic Regression typically achieves 100% on this split.

---

## 🔮 Future Improvements

- [ ] Add a REST API (`/api/predict`) returning JSON for programmatic access
- [ ] Support batch CSV upload for classifying multiple flowers at once
- [ ] Add cross-validation (k-fold) for more robust accuracy estimation
- [ ] Integrate SHAP or LIME for prediction explainability
- [ ] Add a Docker container for one-command deployment
- [ ] Implement a comparison dashboard showing all three model accuracies side-by-side
- [ ] Add support for uploading custom CSV datasets to retrain the model
- [ ] Write end-to-end Selenium browser tests
- [ ] Add CI/CD pipeline with GitHub Actions

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Iris Classification Project**

- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)
- Email: your.email@example.com

---

> Built with ❤️ using Python, Flask, and scikit-learn.
