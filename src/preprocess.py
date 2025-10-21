import argparse, re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = URL_RE.sub("", s)
    s = MENTION_RE.sub("", s)
    s = HASHTAG_RE.sub(r"\1", s)  # keep hashtag word
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV with columns: text, label")
    ap.add_argument("--out", dest="outdir", required=True, help="Output directory for processed splits")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.inp)
    if "text" not in df.columns or "label" not in df.columns:
        raise SystemExit("CSV must contain 'text' and 'label' columns.")

    df["text"] = df["text"].apply(clean_text)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.random_state, stratify=df["label"])

    train_path = outdir / "train.csv"
    test_path = outdir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {train_path} ({len(train_df)}) and {test_path} ({len(test_df)})")

if __name__ == "__main__":
    main()
