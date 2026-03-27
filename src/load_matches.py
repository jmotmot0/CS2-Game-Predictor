from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_matches_csv(filename: str = "matches.csv") -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    try:
        df = load_matches_csv()
        print(df.head())
        print(df.shape)
    except FileNotFoundError as e:
        print(e)