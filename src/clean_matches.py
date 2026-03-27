from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # нормализация названий колонок
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # удаление полных дублей
    df = df.drop_duplicates()

    # пример базовой обработки даты
    if "match_date" in df.columns:
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

    return df


def main():
    input_path = RAW_DIR / "matches.csv"
    output_path = INTERIM_DIR / "matches_clean.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    clean_df = clean_matches(df)
    clean_df.to_csv(output_path, index=False)

    print(f"Saved cleaned data to: {output_path}")
    print(clean_df.shape)


if __name__ == "__main__":
    main()