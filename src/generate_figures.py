import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_FIGURES_DIR = ROOT / "outputs" / "figures"
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set high-quality plot aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def generate_sentiment_distribution():
    print("[1/4] Generating Sentiment Distribution Plot...")
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "validation.csv"
    test_path = DATA_DIR / "test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        print("  Warning: Split CSV files not found. Skipping distribution plot.")
        return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_df["Split"] = f"Train (n={len(train_df):,})"
    val_df["Split"] = f"Validation (n={len(val_df):,})"
    test_df["Split"] = f"Test (n={len(test_df):,})"

    combined = pd.concat([train_df[["sentiment", "Split"]], val_df[["sentiment", "Split"]], test_df[["sentiment", "Split"]]])
    combined["sentiment"] = combined["sentiment"].astype(str).str.strip().str.capitalize()

    palette = {"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"}

    plt.figure(figsize=(10, 5), dpi=300)
    ax = sns.countplot(
        data=combined,
        x="Split",
        hue="sentiment",
        palette=palette,
        order=[f"Train (n={len(train_df):,})", f"Validation (n={len(val_df):,})", f"Test (n={len(test_df):,})"],
    )

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height):,}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.title("Sentiment Class Distribution Across Dataset Splits", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Dataset Split", fontsize=12, weight="bold")
    plt.ylabel("Review Count", fontsize=12, weight="bold")
    plt.legend(title="Sentiment Class", frameon=True)
    plt.tight_layout()

    out_path = OUTPUT_FIGURES_DIR / "sentiment_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved -> {out_path}")


def generate_loss_curve():
    print("[2/4] Generating BERT Training & Evaluation Loss Curve...")
    trainer_state_path = MODEL_DIR / "bert" / "checkpoints" / "checkpoint-5997" / "trainer_state.json"

    if not trainer_state_path.exists():
        # Fallback to any other checkpoint folder if present
        checkpoints = list(MODEL_DIR.glob("bert/checkpoints/*/trainer_state.json"))
        if checkpoints:
            trainer_state_path = checkpoints[-1]
        else:
            print("  Warning: trainer_state.json not found. Skipping loss curve.")
            return

    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for entry in log_history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])
        if "eval_loss" in entry and "step" in entry:
            eval_steps.append(entry["step"])
            eval_loss.append(entry["eval_loss"])

    plt.figure(figsize=(10, 5), dpi=300)
    if train_steps:
        plt.plot(train_steps, train_loss, label="Training Loss", color="#3498db", linewidth=2)
    if eval_steps:
        plt.plot(eval_steps, eval_loss, label="Validation Loss", color="#e74c3c", linewidth=2.5, marker="o")

    plt.title("Fine-Tuned BERT Training and Validation Loss Curve", fontsize=14, weight="bold", pad=15)
    plt.xlabel("Global Training Steps", fontsize=12, weight="bold")
    plt.ylabel("Cross-Entropy Loss", fontsize=12, weight="bold")
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()

    out_path = OUTPUT_FIGURES_DIR / "bert_loss_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved -> {out_path}")


def generate_model_comparison_bar_chart():
    print("[3/4] Generating Model Comparison Bar Chart...")
    comparison_csv = ROOT / "outputs" / "evaluation" / "final_model_comparison.csv"

    if not comparison_csv.exists():
        print("  Warning: final_model_comparison.csv not found. Skipping comparison chart.")
        return

    df = pd.read_csv(comparison_csv)
    # Pivot to long format for clean grouped bar plotting
    df_long = df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 5.5), dpi=300)
    palette = {"Fine-tuned BERT": "#2980b9", "Traditional TF-IDF": "#95a5a6"}

    ax = sns.barplot(
        data=df_long,
        x="Metric",
        y="Score",
        hue="Model",
        palette=palette,
        edgecolor="black",
        linewidth=0.5,
    )

    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(
                f"{val * 100:.1f}%",
                (p.get_x() + p.get_width() / 2.0, val),
                ha="center",
                va="bottom",
                fontsize=9.5,
                weight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.ylim(0, 1.05)
    plt.title("Evaluation Benchmark: Fine-Tuned BERT vs. Traditional TF-IDF Baseline", fontsize=13, weight="bold", pad=15)
    plt.xlabel("Evaluation Metric", fontsize=12, weight="bold")
    plt.ylabel("Score", fontsize=12, weight="bold")
    plt.legend(title="Model", frameon=True, loc="upper right")
    plt.tight_layout()

    out_path = OUTPUT_FIGURES_DIR / "model_comparison_metrics.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved -> {out_path}")


def generate_confusion_matrices():
    print("[4/4] Generating Confusion Matrices on Test Data...")
    test_path = DATA_DIR / "test.csv"
    vec_path = MODEL_DIR / "traditional_ml" / "tfidf_vectorizer.joblib"
    trad_model_path = MODEL_DIR / "traditional_ml" / "best_tfidf_model.joblib"
    bert_dir = MODEL_DIR / "bert" / "fine-tuned"

    if not (test_path.exists() and vec_path.exists() and trad_model_path.exists() and bert_dir.exists()):
        print("  Warning: Model artifacts or test set not found. Skipping confusion matrices.")
        return

    test_df = pd.read_csv(test_path)
    test_df["clean_text"] = test_df["clean_text"].fillna("").astype(str)
    test_df["sentiment"] = test_df["sentiment"].astype(str).str.strip().str.lower()
    test_df = test_df[test_df["clean_text"].str.strip().ne("")].reset_index(drop=True)

    y_true = test_df["sentiment"]
    labels = sorted(y_true.unique())

    # 1. Traditional TF-IDF predictions
    vectorizer = joblib.load(vec_path)
    trad_model = joblib.load(trad_model_path)
    X_test = vectorizer.transform(test_df["clean_text"])
    trad_preds = trad_model.predict(X_test)

    # 2. Fine-tuned BERT predictions
    tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
    bert_model = AutoModelForSequenceClassification.from_pretrained(str(bert_dir))

    label_path = bert_dir / "label_encoder_classes.txt"
    if label_path.exists():
        bert_labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        bert_labels = [bert_model.config.id2label[i] for i in range(bert_model.config.num_labels)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    bert_model.eval()

    bert_preds_idx = []
    texts = test_df["clean_text"].tolist()
    batch_size = 64
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            logits = bert_model(**batch).logits
            bert_preds_idx.extend(logits.argmax(dim=1).cpu().tolist())

    bert_preds = [bert_labels[i] for i in bert_preds_idx]

    # Create side-by-side Confusion Matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    disp1 = ConfusionMatrixDisplay.from_predictions(
        y_true,
        trad_preds,
        labels=labels,
        display_labels=[l.capitalize() for l in labels],
        cmap="Blues",
        ax=axes[0],
        colorbar=False,
    )
    axes[0].set_title("Traditional TF-IDF Confusion Matrix\n(Accuracy: 79.5%, Macro F1: 64.3%)", fontsize=12, weight="bold")
    axes[0].grid(False)

    disp2 = ConfusionMatrixDisplay.from_predictions(
        y_true,
        bert_preds,
        labels=labels,
        display_labels=[l.capitalize() for l in labels],
        cmap="Greens",
        ax=axes[1],
        colorbar=False,
    )
    axes[1].set_title("Fine-Tuned BERT Confusion Matrix\n(Accuracy: 82.6%, Macro F1: 67.9%)", fontsize=12, weight="bold")
    axes[1].grid(False)

    plt.suptitle("Test Set Confusion Matrix Comparison (n=1,999 reviews)", fontsize=14, weight="bold", y=1.02)
    plt.tight_layout()

    out_path = OUTPUT_FIGURES_DIR / "confusion_matrices.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")


def main():
    print(f"Saving all analytical figures to: {OUTPUT_FIGURES_DIR}\n")
    generate_sentiment_distribution()
    generate_loss_curve()
    generate_model_comparison_bar_chart()
    generate_confusion_matrices()
    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()
