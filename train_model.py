"""
train_model.py
==============
Iris Flower Classification — Training Pipeline.

This module loads the canonical scikit-learn Iris dataset, trains three
classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours),
evaluates each model on a held-out test set, selects the best performer,
serialises it to ``iris_model.pkl`` via joblib, and generates four
data-visualisation artefacts in the ``images/`` directory.

Usage::

    python train_model.py
"""

import os
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Use the non-interactive Agg backend so the script runs without a display.
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_data():
    """Load the scikit-learn Iris dataset and return a tidy DataFrame.

    Returns
    -------
    df : pandas.DataFrame
        Full dataset with columns sepal_length, sepal_width, petal_length,
        petal_width, and species (string label).
    X : numpy.ndarray, shape (150, 4)
        Feature matrix.
    y : numpy.ndarray, shape (150,)
        Integer target labels (0, 1, 2).
    label_names : list of str
        Ordered class names, e.g. ['setosa', 'versicolor', 'virginica'].
    """
    iris = load_iris()

    # Build a human-readable DataFrame.
    df = pd.DataFrame(iris.data, columns=["sepal_length", "sepal_width",
                                          "petal_length", "petal_width"])
    # Map integer targets to string species names.
    df["species"] = [iris.target_names[t] for t in iris.target]

    X = iris.data
    y = iris.target
    label_names = list(iris.target_names)

    return df, X, y, label_names


def split_data(X, y):
    """Split feature matrix and labels into training and test subsets.

    Uses an 80 / 20 split with ``random_state=42`` for reproducibility.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, 4).
    y : numpy.ndarray
        Integer target labels of shape (n_samples,).

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy.ndarray
        Train and test partitions.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)


def save_dataset(df):
    """Persist the Iris DataFrame as a CSV file in the ``dataset/`` directory.

    Creates the directory if it does not already exist.

    Parameters
    ----------
    df : pandas.DataFrame
        The full Iris dataset (150 rows × 5 columns).
    """
    os.makedirs("dataset", exist_ok=True)
    csv_path = os.path.join("dataset", "iris_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Dataset saved → {csv_path}")


# ---------------------------------------------------------------------------
# Model training, evaluation and persistence
# ---------------------------------------------------------------------------

def train_models(X_train, y_train):
    """Instantiate, fit, and return three candidate classifiers.

    The three models are:
    * Logistic Regression (``max_iter=1000``, ``random_state=42``)
    * Decision Tree (``random_state=42``)
    * K-Nearest Neighbours (default ``k=5``)

    Parameters
    ----------
    X_train : numpy.ndarray
        Training feature matrix.
    y_train : numpy.ndarray
        Training labels.

    Returns
    -------
    models : list of (str, estimator)
        Each element is a ``(name, fitted_model)`` pair.
    """
    candidates = [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000, random_state=42)),
        ("Decision Tree",
         DecisionTreeClassifier(random_state=42)),
        ("K-Nearest Neighbours",
         KNeighborsClassifier()),
    ]

    fitted_models = []
    for name, model in candidates:
        model.fit(X_train, y_train)
        fitted_models.append((name, model))
        print(f"  Trained: {name}")

    return fitted_models


def evaluate_models(models, X_test, y_test):
    """Evaluate each model on the test set and print an accuracy table.

    Parameters
    ----------
    models : list of (str, estimator)
        Fitted ``(name, model)`` pairs from :func:`train_models`.
    X_test : numpy.ndarray
        Test feature matrix.
    y_test : numpy.ndarray
        Test labels.

    Returns
    -------
    evaluated : list of (str, estimator, float)
        Each element is a ``(name, model, accuracy)`` triple.
    """
    print("\n  ┌─────────────────────────────┬───────────┐")
    print("  │ Model                       │ Accuracy  │")
    print("  ├─────────────────────────────┼───────────┤")

    evaluated = []
    for name, model in models:
        accuracy = model.score(X_test, y_test)
        evaluated.append((name, model, accuracy))
        print(f"  │ {name:<27s} │  {accuracy:.4f}   │")

    print("  └─────────────────────────────┴───────────┘")
    return evaluated


def select_best(evaluated):
    """Select the model with the highest test-set accuracy.

    Parameters
    ----------
    evaluated : list of (str, estimator, float)
        Output of :func:`evaluate_models`.

    Returns
    -------
    best : (str, estimator, float)
        The ``(name, model, accuracy)`` triple with maximum accuracy.
    """
    best = max(evaluated, key=lambda t: t[2])
    print(f"\n  Best model: {best[0]}  (accuracy={best[2]:.4f})")
    return best


def save_model(model, path="iris_model.pkl"):
    """Serialise a trained scikit-learn estimator to disk using joblib.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn model that implements ``predict`` and
        ``predict_proba``.
    path : str, optional
        Destination file path. Defaults to ``"iris_model.pkl"``.
    """
    try:
        joblib.dump(model, path)
        print(f"  Model saved → {path}")
    except OSError as exc:
        print(f"  ERROR: could not save model to '{path}': {exc}",
              file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# Visualisation generation
# ---------------------------------------------------------------------------

def generate_pair_plot(df):
    """Generate a pair plot coloured by species and save to PNG.

    Parameters
    ----------
    df : pandas.DataFrame
        The full Iris dataset including the ``species`` column.
    """
    os.makedirs("images", exist_ok=True)
    output_path = os.path.join("images", "pair_plot.png")
    try:
        pair_grid = sns.pairplot(df, hue="species", height=2.5)
        pair_grid.figure.suptitle("Iris Dataset — Pair Plot", y=1.02)
        pair_grid.savefig(output_path, bbox_inches="tight")
        print(f"  Pair plot saved → {output_path}")
    finally:
        plt.close("all")


def generate_histogram(df):
    """Generate feature distribution histograms and save to PNG.

    Drops the categorical ``species`` column before plotting so that only
    the four numeric features are histogrammed.

    Parameters
    ----------
    df : pandas.DataFrame
        The full Iris dataset.
    """
    os.makedirs("images", exist_ok=True)
    output_path = os.path.join("images", "histogram.png")
    try:
        numeric_df = df.drop("species", axis=1)
        axes = numeric_df.hist(bins=20, figsize=(12, 8), edgecolor="black",
                               color="steelblue", alpha=0.8)
        fig = axes.flatten()[0].get_figure()
        fig.suptitle("Iris Dataset — Feature Distributions", fontsize=14,
                     y=1.02)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Histogram saved → {output_path}")
    finally:
        plt.close("all")


def generate_heatmap(df):
    """Generate a correlation heatmap for the numeric features and save to PNG.

    Parameters
    ----------
    df : pandas.DataFrame
        The full Iris dataset.
    """
    os.makedirs("images", exist_ok=True)
    output_path = os.path.join("images", "heatmap.png")
    try:
        numeric_df = df.drop("species", axis=1)
        corr_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax)
        ax.set_title("Iris Dataset — Feature Correlation Heatmap",
                     fontsize=14, pad=12)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Heatmap saved → {output_path}")
    finally:
        plt.close("all")


def generate_confusion_matrix(model, X_test, y_test, label_names):
    """Generate and save a confusion matrix display for the best model.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn estimator.
    X_test : numpy.ndarray
        Test feature matrix.
    y_test : numpy.ndarray
        Test labels.
    label_names : list of str
        Ordered class names used as tick labels.
    """
    os.makedirs("images", exist_ok=True)
    output_path = os.path.join("images", "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(7, 6))
    try:
        disp = ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test,
            display_labels=label_names,
            cmap="Blues",
            ax=ax,
        )
        disp.ax_.set_title("Confusion Matrix — Best Model", fontsize=14,
                            pad=12)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Confusion matrix saved → {output_path}")
    finally:
        plt.close("all")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    """Orchestrate the full Iris classification training pipeline.

    Steps performed:
    1. Load the Iris dataset.
    2. Split data into train / test sets (80 / 20).
    3. Save the dataset to CSV.
    4. Train three classifiers.
    5. Evaluate and compare model accuracies.
    6. Select the best model.
    7. Serialise the best model to disk.
    8. Generate four visualisation artefacts.
    9. Print a success message.
    """
    print("=" * 52)
    print("  Iris Flower Classification — Training Pipeline")
    print("=" * 52)

    # Step 1: Load data.
    print("\n[1/8] Loading Iris dataset …")
    df, X, y, label_names = load_data()
    print(f"  Loaded {len(df)} samples, {X.shape[1]} features, "
          f"{len(label_names)} classes.")

    # Step 2: Split data.
    print("\n[2/8] Splitting data (80 / 20, random_state=42) …")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # Step 3: Save CSV.
    print("\n[3/8] Saving dataset to CSV …")
    save_dataset(df)

    # Step 4: Train models.
    print("\n[4/8] Training models …")
    models = train_models(X_train, y_train)

    # Step 5: Evaluate models.
    print("\n[5/8] Evaluating models on test set …")
    evaluated = evaluate_models(models, X_test, y_test)

    # Step 6: Select best model.
    print("\n[6/8] Selecting best model …")
    best_name, best_model, best_accuracy = select_best(evaluated)

    # Step 7: Save the best model.
    print("\n[7/8] Saving model to disk …")
    save_model(best_model)

    # Step 8: Generate visualisations.
    print("\n[8/8] Generating visualisations …")
    generate_pair_plot(df)
    generate_histogram(df)
    generate_heatmap(df)
    generate_confusion_matrix(best_model, X_test, y_test, label_names)

    # Done.
    print("\n" + "=" * 52)
    print("  Training complete!")
    print(f"  Best model : {best_name}")
    print(f"  Accuracy   : {best_accuracy:.4f} ({best_accuracy * 100:.2f} %)")
    print("  Artefacts  : iris_model.pkl, dataset/iris_data.csv,")
    print("               images/{{pair_plot,histogram,heatmap,")
    print("               confusion_matrix}}.png")
    print("=" * 52)


if __name__ == "__main__":
    main()
