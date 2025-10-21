import argparse, pickle
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Processed data directory (test.csv, vectorizer.pkl)")
    ap.add_argument("--model_dir", required=True, help="Directory containing model.pkl")
    args = ap.parse_args()

    data_dir = Path(args.data)
    model_dir = Path(args.model_dir)

    test = pd.read_csv(data_dir / "test.csv")
    with open(data_dir / "vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    with open(model_dir / "model.pkl", "rb") as f:
        clf = pickle.load(f)

    Xte = vec.transform(test["text"].astype(str))
    yte = test["label"].astype(str)
    pred = clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(yte, pred))

    cm = confusion_matrix(yte, pred, labels=sorted(yte.unique()))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(cm))); ax.set_yticks(range(len(cm)))
    labs = sorted(yte.unique())
    ax.set_xticklabels(labs, rotation=45, ha="right")
    ax.set_yticklabels(labs)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png = data_dir / "confusion_matrix.png"
    fig.savefig(out_png, dpi=120)
    print(f"Saved plot: {out_png}")

if __name__ == "__main__":
    main()
