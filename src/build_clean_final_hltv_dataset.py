
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RAW_CSV = PROJECT_ROOT / "data" / "raw" / "matches_raw.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "hltv_final_clean"

DEFAULT_ENRICHED_DIRS = [
    PROJECT_ROOT / "data" / "interim" / "hltv_enriched",
    PROJECT_ROOT / "data" / "interim" / "hltv_enriched_latest_1000",
    PROJECT_ROOT / "data" / "interim" / "hltv_enriched_rest",
]

CS2_LAN_CUTOFF = "2023-10-16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple HLTV enriched folders, remove duplicates, "
            "cut everything before the chosen CS2 cutoff date, and save "
            "clean relational tables for feature engineering."
        )
    )
    parser.add_argument(
        "--raw-csv",
        type=str,
        default=str(DEFAULT_RAW_CSV),
        help=f"Path to matches_raw.csv. Default: {DEFAULT_RAW_CSV}",
    )
    parser.add_argument(
        "--enriched-dirs",
        nargs="+",
        default=[str(p) for p in DEFAULT_ENRICHED_DIRS],
        help=(
            "One or more enriched folders that contain "
            "matches_enriched.csv / match_lineups.csv / veto_steps.csv / "
            "match_maps.csv / map_player_stats.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for cleaned output tables. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=CS2_LAN_CUTOFF,
        help=(
            "Keep only matches on or after this date (YYYY-MM-DD). "
            f"Default: {CS2_LAN_CUTOFF}"
        ),
    )
    return parser.parse_args()


def safe_str(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(safe_str)
    return df


def read_csv_if_exists(path: Path, dtype: dict[str, str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=dtype)
    return normalize_text_columns(df)


def load_raw_matches(path: Path, cutoff_date: str) -> pd.DataFrame:
    df = read_csv_if_exists(path, dtype={"match_id": str, "event_id": str})
    if df is None:
        raise FileNotFoundError(f"Raw matches file not found: {path}")

    required = {
        "match_id",
        "match_date",
        "event_name",
        "team1",
        "team2",
        "team1_score",
        "team2_score",
        "winner",
        "source_url",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw matches file is missing columns: {sorted(missing)}")

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date"]).copy()

    cutoff_ts = pd.Timestamp(cutoff_date)
    df = df[df["match_date"] >= cutoff_ts].copy()

    df = (
        df.sort_values(["match_date", "match_id"])
        .drop_duplicates(subset=["match_id"])
        .reset_index(drop=True)
    )

    df["match_date"] = df["match_date"].dt.strftime("%Y-%m-%d")
    df["game_version_policy"] = "CS2_from_first_LAN_cutoff"
    df["game_version_cutoff_date"] = cutoff_date

    return df


def concat_tables(enriched_dirs: Iterable[Path], filename: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    id_dtype = {
        "match_id": str,
        "event_id": str,
        "team_id": str,
        "team1_id": str,
        "team2_id": str,
        "player_id": str,
        "mapstatsid": str,
    }

    for directory in enriched_dirs:
        file_path = directory / filename
        df = read_csv_if_exists(file_path, dtype=id_dtype)
        if df is not None and not df.empty:
            df["__source_dir"] = directory.name
            parts.append(df)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out = normalize_text_columns(out)
    return out


def dedupe_matches_enriched(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "match_datetime_utc" in df.columns:
        dt = pd.to_datetime(df["match_datetime_utc"], errors="coerce")
        df = df.assign(__match_datetime_sort=dt)
        df = df.sort_values(["match_id", "__match_datetime_sort", "__source_dir"], na_position="last")
        df = df.drop(columns=["__match_datetime_sort"], errors="ignore")

    df = df.drop_duplicates(subset=["match_id"], keep="last").reset_index(drop=True)
    return df


def dedupe_lineups(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    subset = [c for c in ["match_id", "team_ordinal", "player_id"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="last")
    return df.reset_index(drop=True)


def dedupe_veto(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    subset = [c for c in ["match_id", "step_number", "action", "map_name", "team_name"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="last")
    return df.reset_index(drop=True)


def dedupe_match_maps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    subset = [c for c in ["match_id", "map_no"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="last")
    return df.reset_index(drop=True)


def dedupe_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    subset = [c for c in ["match_id", "map_no", "player_id"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="last")
    return df.reset_index(drop=True)


def filter_by_valid_matches(df: pd.DataFrame, valid_match_ids: set[str]) -> pd.DataFrame:
    if df.empty or "match_id" not in df.columns:
        return df
    out = df[df["match_id"].isin(valid_match_ids)].copy()
    return out.reset_index(drop=True)


def merge_match_level(raw_df: pd.DataFrame, matches_enriched_df: pd.DataFrame) -> pd.DataFrame:
    if matches_enriched_df.empty:
        return raw_df.copy()

    merged = raw_df.merge(
        matches_enriched_df.drop(columns=["__source_dir"], errors="ignore"),
        on="match_id",
        how="left",
        suffixes=("_raw", "_enriched"),
    )

    coalesce_pairs = [
        ("match_date_raw", "match_date_enriched", "match_date"),
        ("event_name_raw", "event_name_enriched", "event_name"),
        ("team1_raw", "team1_enriched", "team1"),
        ("team2_raw", "team2_enriched", "team2"),
        ("source_url_raw", "source_url_enriched", "source_url"),
    ]
    for raw_col, enr_col, out_col in coalesce_pairs:
        if raw_col in merged.columns and enr_col in merged.columns:
            merged[out_col] = merged[enr_col].where(merged[enr_col].astype(str) != "", merged[raw_col])

    drop_cols = [
        "match_date_raw",
        "match_date_enriched",
        "event_name_raw",
        "event_name_enriched",
        "team1_raw",
        "team1_enriched",
        "team2_raw",
        "team2_enriched",
        "source_url_raw",
        "source_url_enriched",
    ]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors="ignore")

    preferred_order = [
        "match_id",
        "match_date",
        "match_datetime_utc",
        "event_id",
        "event_name",
        "team1",
        "team2",
        "team1_id",
        "team2_id",
        "team1_score",
        "team2_score",
        "winner",
        "team1_rank",
        "team2_rank",
        "bo",
        "lan_online",
        "match_context",
        "forfeit_note",
        "team1_roster_hash",
        "team2_roster_hash",
        "source",
        "source_url",
        "game",
        "game_version_policy",
        "game_version_cutoff_date",
        "completed",
        "is_professional_proxy",
        "professional_filter_note",
    ]
    front = [c for c in preferred_order if c in merged.columns]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + rest]

    merged = merged.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return merged


def enrich_match_maps_with_match_date(match_maps: pd.DataFrame, matches_final: pd.DataFrame) -> pd.DataFrame:
    if match_maps.empty:
        return match_maps
    if "match_date" in match_maps.columns:
        return match_maps

    lookup = matches_final[["match_id", "match_date"]].drop_duplicates()
    out = match_maps.merge(lookup, on="match_id", how="left")
    cols = ["match_id", "match_date"] + [c for c in out.columns if c not in {"match_id", "match_date"}]
    return out[cols]


def enrich_player_stats_with_match_date(player_stats: pd.DataFrame, matches_final: pd.DataFrame) -> pd.DataFrame:
    if player_stats.empty:
        return player_stats
    if "match_date" in player_stats.columns:
        return player_stats

    lookup = matches_final[["match_id", "match_date"]].drop_duplicates()
    out = player_stats.merge(lookup, on="match_id", how="left")
    cols = ["match_id", "match_date"] + [c for c in out.columns if c not in {"match_id", "match_date"}]
    return out[cols]


def write_summary(
    output_dir: Path,
    cutoff_date: str,
    raw_df: pd.DataFrame,
    matches_final: pd.DataFrame,
    lineups: pd.DataFrame,
    veto: pd.DataFrame,
    match_maps: pd.DataFrame,
    player_stats: pd.DataFrame,
) -> None:
    summary_lines = []
    summary_lines.append("HLTV final clean dataset summary")
    summary_lines.append("")
    summary_lines.append(f"CS2 cutoff policy: keep match_date >= {cutoff_date}")
    summary_lines.append("")
    summary_lines.append(f"matches_raw_clean.csv rows: {len(raw_df)}")
    summary_lines.append(f"matches_final.csv rows: {len(matches_final)}")
    summary_lines.append(f"match_lineups.csv rows: {len(lineups)}")
    summary_lines.append(f"veto_steps.csv rows: {len(veto)}")
    summary_lines.append(f"match_maps.csv rows: {len(match_maps)}")
    summary_lines.append(f"map_player_stats.csv rows: {len(player_stats)}")
    summary_lines.append("")

    if not matches_final.empty and "match_date" in matches_final.columns:
        summary_lines.append(
            f"Final match date range: {matches_final['match_date'].min()} -> {matches_final['match_date'].max()}"
        )
        summary_lines.append("")

    summary_lines.append("Generated files:")
    summary_lines.append("- matches_raw_clean.csv")
    summary_lines.append("- matches_final.csv")
    summary_lines.append("- match_lineups.csv")
    summary_lines.append("- veto_steps.csv")
    summary_lines.append("- match_maps.csv")
    summary_lines.append("- map_player_stats.csv")

    (output_dir / "README_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    raw_csv = Path(args.raw_csv)
    enriched_dirs = [Path(p) for p in args.enriched_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw matches...")
    raw_df = load_raw_matches(raw_csv, cutoff_date=args.cutoff_date)
    valid_match_ids = set(raw_df["match_id"].astype(str))

    print("Loading enriched tables...")
    matches_enriched = dedupe_matches_enriched(concat_tables(enriched_dirs, "matches_enriched.csv"))
    match_lineups = dedupe_lineups(concat_tables(enriched_dirs, "match_lineups.csv"))
    veto_steps = dedupe_veto(concat_tables(enriched_dirs, "veto_steps.csv"))
    match_maps = dedupe_match_maps(concat_tables(enriched_dirs, "match_maps.csv"))
    map_player_stats = dedupe_player_stats(concat_tables(enriched_dirs, "map_player_stats.csv"))

    print("Filtering all enriched tables by valid post-cutoff match ids...")
    matches_enriched = filter_by_valid_matches(matches_enriched, valid_match_ids)
    match_lineups = filter_by_valid_matches(match_lineups, valid_match_ids)
    veto_steps = filter_by_valid_matches(veto_steps, valid_match_ids)
    match_maps = filter_by_valid_matches(match_maps, valid_match_ids)
    map_player_stats = filter_by_valid_matches(map_player_stats, valid_match_ids)

    print("Building final match-level table...")
    matches_final = merge_match_level(raw_df, matches_enriched)

    print("Backfilling match_date into map-level and player-level tables...")
    match_maps = enrich_match_maps_with_match_date(match_maps, matches_final)
    map_player_stats = enrich_player_stats_with_match_date(map_player_stats, matches_final)

    print("Writing cleaned outputs...")
    raw_df.to_csv(output_dir / "matches_raw_clean.csv", index=False)
    matches_final.to_csv(output_dir / "matches_final.csv", index=False)
    match_lineups.to_csv(output_dir / "match_lineups.csv", index=False)
    veto_steps.to_csv(output_dir / "veto_steps.csv", index=False)
    match_maps.to_csv(output_dir / "match_maps.csv", index=False)
    map_player_stats.to_csv(output_dir / "map_player_stats.csv", index=False)

    write_summary(
        output_dir=output_dir,
        cutoff_date=args.cutoff_date,
        raw_df=raw_df,
        matches_final=matches_final,
        lineups=match_lineups,
        veto=veto_steps,
        match_maps=match_maps,
        player_stats=map_player_stats,
    )

    print("\nDone.")
    print(f"Output directory: {output_dir}")
    print(f"matches_raw_clean.csv: {len(raw_df)} rows")
    print(f"matches_final.csv: {len(matches_final)} rows")
    print(f"match_lineups.csv: {len(match_lineups)} rows")
    print(f"veto_steps.csv: {len(veto_steps)} rows")
    print(f"match_maps.csv: {len(match_maps)} rows")
    print(f"map_player_stats.csv: {len(map_player_stats)} rows")

    if not matches_final.empty and "match_date" in matches_final.columns:
        print(
            f"Final date range: {matches_final['match_date'].min()} -> {matches_final['match_date'].max()}"
        )


if __name__ == "__main__":
    main()
