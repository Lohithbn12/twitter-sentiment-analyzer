import argparse, pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, help="Input directory with train.csv / test.csv")
    ap.add_argument("--out", dest="outdir", required=True, help="Output directory to save vectorizer.pkl")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(indir / "train.csv")
    vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2)
    vectorizer.fit(train["text"].astype(str))

    with open(outdir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved vectorizer: {outdir / 'vectorizer.pkl'} (vocab size: {len(vectorizer.vocabulary_)})")

if __name__ == "__main__":
    main()
