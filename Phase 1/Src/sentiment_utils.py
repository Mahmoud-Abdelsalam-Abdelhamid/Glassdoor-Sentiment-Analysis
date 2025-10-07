import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def load_dataframe(path):
    """Load CSV into dataframe."""
    return pd.read_csv(path)


def evaluate_classification(y_true, y_pred, average="weighted"):
    """
    Print and return classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion(y_true, y_pred, labels=None, figsize=(6, 5)):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()