import argparse, pickle
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="Text to classify")
    ap.add_argument("--data", default="data/processed", help="Directory containing vectorizer.pkl")
    ap.add_argument("--model_dir", default="models", help="Directory containing model.pkl")
    args = ap.parse_args()

    vec_path = Path(args.data) / "vectorizer.pkl"
    model_path = Path(args.model_dir) / "model.pkl"

    with open(vec_path, "rb") as f:
        vec = pickle.load(f)
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    X = vec.transform([args.text])
    pred = clf.predict(X)[0]
    print(pred)

if __name__ == "__main__":
    main()
