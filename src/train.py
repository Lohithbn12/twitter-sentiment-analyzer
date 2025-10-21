import argparse, pickle
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Processed data directory (train.csv, vectorizer.pkl)")
    ap.add_argument("--model_dir", required=True, help="Where to save the trained model")
    ap.add_argument("--model", choices=["lr", "svm"], default="lr")
    args = ap.parse_args()

    data_dir = Path(args.data)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    with open(data_dir / "vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)

    X = vec.transform(train["text"].astype(str))
    y = train["label"].astype(str)

    if args.model == "lr":
        clf = LogisticRegression(max_iter=200)
    else:
        clf = LinearSVC()

    clf.fit(X, y)

    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(clf, f)

    print(f"Saved model to {model_dir / 'model.pkl'} using {args.model.upper()}")

if __name__ == "__main__":
    main()
