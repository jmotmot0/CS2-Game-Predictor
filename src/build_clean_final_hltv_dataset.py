# src/build_clean_final_hltv_dataset.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


RAW_REQUIRED_COLS = [
    "match_id",
    "match_date",
    "event_name",
    "team1",
    "team2",
    "team1_score",
    "team2_score",
    "winner",
    "source_url",
]

ENRICHED_FILES = [
    "matches_enriched.csv",
    "match_lineups.csv",
    "veto_steps.csv",
    "match_maps.csv",
    "map_player_stats.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean final HLTV dataset.")
    parser.add_argument("--raw-csv", required=True, type=Path)
    parser.add_argument("--enriched-dirs", required=True, nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cutoff-date", required=True, type=str)
    return parser.parse_args()


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def ensure_columns(df: pd.DataFrame, required_cols: Iterable[str], df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def normalize_str_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    return df


def normalize_raw(raw: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    raw = raw.copy()
    ensure_columns(raw, RAW_REQUIRED_COLS, "raw")

    raw = normalize_str_columns(raw)

    raw["match_id"] = pd.to_numeric(raw["match_id"], errors="coerce").astype("Int64")
    raw["match_date"] = pd.to_datetime(raw["match_date"], errors="coerce").dt.normalize()
    raw["team1_score"] = pd.to_numeric(raw.get("team1_score"), errors="coerce")
    raw["team2_score"] = pd.to_numeric(raw.get("team2_score"), errors="coerce")

    raw = raw.dropna(subset=["match_id", "match_date", "team1", "team2"])
    raw = raw[raw["match_date"] >= pd.Timestamp(cutoff_date)].copy()

    # dedup: prefer latest row if duplicates exist
    dedup_subset = ["match_id"]
    raw = raw.sort_values(["match_date", "match_id"]).drop_duplicates(
        subset=dedup_subset, keep="last"
    )

    return raw.reset_index(drop=True)


def concat_enriched_csvs(enriched_dirs: list[Path], filename: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for d in enriched_dirs:
        path = d / filename
        if not path.exists():
            print(f"[WARN] Missing enriched file, skipping: {path}")
            continue
        df = pd.read_csv(path)
        df["__source_dir"] = str(d)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No files found for {filename} in {enriched_dirs}")

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = normalize_str_columns(out)
    return out


def normalize_matches_enriched(df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    df = df.copy()

    required = [
        "match_id",
        "match_date",
        "team1",
        "team2",
        "team1_id",
        "team2_id",
    ]
    ensure_columns(df, required, "matches_enriched")

    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").astype("Int64")
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce").dt.normalize()

    if "match_datetime_utc" in df.columns:
        df["match_datetime_utc"] = pd.to_datetime(df["match_datetime_utc"], errors="coerce", utc=True)
    else:
        df["match_datetime_utc"] = pd.NaT

    numeric_cols = [
        "event_id",
        "team1_id",
        "team2_id",
        "team1_rank",
        "team2_rank",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["match_id", "match_date", "team1", "team2"])
    df = df[df["match_date"] >= pd.Timestamp(cutoff_date)].copy()

    # Dedup enriched matches by match_id.
    # Prefer rows with non-null datetime, ranks, ids.
    score_cols = ["match_datetime_utc", "team1_id", "team2_id", "team1_rank", "team2_rank"]
    for col in score_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df["__quality_score"] = df[score_cols].notna().sum(axis=1)

    df = (
        df.sort_values(["match_id", "__quality_score"], ascending=[True, False])
          .drop_duplicates(subset=["match_id"], keep="first")
          .drop(columns=["__quality_score"], errors="ignore")
          .reset_index(drop=True)
    )

    return df


def dedup_generic(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    existing_subset = [c for c in subset if c in df.columns]
    if not existing_subset:
        return df.drop_duplicates().reset_index(drop=True)
    return df.drop_duplicates(subset=existing_subset, keep="first").reset_index(drop=True)


def normalize_and_filter_child_tables(
    match_lineups: pd.DataFrame,
    veto_steps: pd.DataFrame,
    match_maps: pd.DataFrame,
    map_player_stats: pd.DataFrame,
    valid_match_ids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # lineups
    if "match_id" in match_lineups.columns:
        match_lineups["match_id"] = pd.to_numeric(match_lineups["match_id"], errors="coerce").astype("Int64")
        for col in ["team_ordinal", "team_id", "player_id"]:
            if col in match_lineups.columns:
                match_lineups[col] = pd.to_numeric(match_lineups[col], errors="coerce")
        match_lineups = match_lineups[match_lineups["match_id"].isin(valid_match_ids)].copy()
        match_lineups = dedup_generic(
            match_lineups,
            ["match_id", "team_id", "player_id"],
        )

    # veto
    if "match_id" in veto_steps.columns:
        veto_steps["match_id"] = pd.to_numeric(veto_steps["match_id"], errors="coerce").astype("Int64")
        if "step_number" in veto_steps.columns:
            veto_steps["step_number"] = pd.to_numeric(veto_steps["step_number"], errors="coerce")
        veto_steps = veto_steps[veto_steps["match_id"].isin(valid_match_ids)].copy()
        veto_steps = dedup_generic(
            veto_steps,
            ["match_id", "step_number", "team_name", "action", "map_name"],
        )

    # maps
    if "match_id" in match_maps.columns:
        match_maps["match_id"] = pd.to_numeric(match_maps["match_id"], errors="coerce").astype("Int64")
        numeric_cols = [
            "map_no", "team1_map_score", "team2_map_score", "mapstatsid",
            "team1_ct_rounds", "team1_t_rounds", "team2_ct_rounds", "team2_t_rounds",
        ]
        for col in numeric_cols:
            if col in match_maps.columns:
                match_maps[col] = pd.to_numeric(match_maps[col], errors="coerce")
        match_maps = match_maps[match_maps["match_id"].isin(valid_match_ids)].copy()
        match_maps = dedup_generic(
            match_maps,
            ["match_id", "map_no", "map_name"],
        )

    # player stats
    if "match_id" in map_player_stats.columns:
        map_player_stats["match_id"] = pd.to_numeric(map_player_stats["match_id"], errors="coerce").astype("Int64")
        numeric_cols = [
            "map_no", "mapstatsid", "team_id", "player_id", "kills", "hs_kills", "assists",
            "flash_assists", "deaths", "traded_deaths", "adr", "kast", "opening_kills",
            "opening_deaths", "rating", "multi_kills", "clutch_wins", "round_swing",
        ]
        for col in numeric_cols:
            if col in map_player_stats.columns:
                map_player_stats[col] = pd.to_numeric(map_player_stats[col], errors="coerce")
        map_player_stats = map_player_stats[map_player_stats["match_id"].isin(valid_match_ids)].copy()
        map_player_stats = dedup_generic(
            map_player_stats,
            ["match_id", "map_no", "team_id", "player_id"],
        )

    return (
        match_lineups.reset_index(drop=True),
        veto_steps.reset_index(drop=True),
        match_maps.reset_index(drop=True),
        map_player_stats.reset_index(drop=True),
    )


def merge_raw_and_enriched(raw: pd.DataFrame, enriched_matches: pd.DataFrame) -> pd.DataFrame:
    raw_cols = [c for c in raw.columns if c not in {"match_id", "match_date"}]

    matches_final = enriched_matches.merge(
        raw[["match_id", "match_date"] + raw_cols],
        on="match_id",
        how="left",
        suffixes=("", "_raw"),
    )

    # preserve enriched match_date primarily
    if "match_date_raw" in matches_final.columns:
        matches_final["match_date"] = matches_final["match_date"].fillna(matches_final["match_date_raw"])
        matches_final = matches_final.drop(columns=["match_date_raw"])

    # canonical target
    matches_final["team1_score"] = pd.to_numeric(matches_final.get("team1_score"), errors="coerce")
    matches_final["team2_score"] = pd.to_numeric(matches_final.get("team2_score"), errors="coerce")

    winner_norm = matches_final.get("winner")
    if winner_norm is not None:
        winner_norm = winner_norm.astype("string").str.strip().str.lower()
    else:
        winner_norm = pd.Series(pd.NA, index=matches_final.index, dtype="string")

    team1_norm = matches_final["team1"].astype("string").str.strip().str.lower()
    team2_norm = matches_final["team2"].astype("string").str.strip().str.lower()

    matches_final["team1_win"] = pd.NA
    matches_final.loc[winner_norm == team1_norm, "team1_win"] = 1
    matches_final.loc[winner_norm == team2_norm, "team1_win"] = 0

    # fallback from scores
    score_mask = (
        matches_final["team1_win"].isna()
        & matches_final["team1_score"].notna()
        & matches_final["team2_score"].notna()
    )
    matches_final.loc[
        score_mask & (matches_final["team1_score"] > matches_final["team2_score"]),
        "team1_win"
    ] = 1
    matches_final.loc[
        score_mask & (matches_final["team1_score"] < matches_final["team2_score"]),
        "team1_win"
    ] = 0

    matches_final["team1_win"] = pd.to_numeric(matches_final["team1_win"], errors="coerce").astype("Int64")
    matches_final["is_valid_result"] = matches_final["team1_win"].isin([0, 1]).astype(int)

    return matches_final


def build_summary(
    cutoff_date: str,
    raw_clean: pd.DataFrame,
    matches_final: pd.DataFrame,
    match_lineups: pd.DataFrame,
    veto_steps: pd.DataFrame,
    match_maps: pd.DataFrame,
    map_player_stats: pd.DataFrame,
) -> str:
    lines = [
        "HLTV final clean dataset summary",
        f"cutoff_date: {cutoff_date}",
        "",
        f"matches_raw_clean rows: {len(raw_clean):,}",
        f"matches_final rows: {len(matches_final):,}",
        f"match_lineups rows: {len(match_lineups):,}",
        f"veto_steps rows: {len(veto_steps):,}",
        f"match_maps rows: {len(match_maps):,}",
        f"map_player_stats rows: {len(map_player_stats):,}",
        "",
    ]

    if not matches_final.empty:
        lines.extend([
            f"matches_final min_date: {matches_final['match_date'].min()}",
            f"matches_final max_date: {matches_final['match_date'].max()}",
            f"matches_final valid_result_rows: {int(matches_final['is_valid_result'].sum())}",
        ])

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_csv_safe(args.raw_csv)
    raw_clean = normalize_raw(raw, cutoff_date=args.cutoff_date)

    matches_enriched = concat_enriched_csvs(args.enriched_dirs, "matches_enriched.csv")
    match_lineups = concat_enriched_csvs(args.enriched_dirs, "match_lineups.csv")
    veto_steps = concat_enriched_csvs(args.enriched_dirs, "veto_steps.csv")
    match_maps = concat_enriched_csvs(args.enriched_dirs, "match_maps.csv")
    map_player_stats = concat_enriched_csvs(args.enriched_dirs, "map_player_stats.csv")

    matches_enriched = normalize_matches_enriched(matches_enriched, cutoff_date=args.cutoff_date)

    # Keep only enriched matches that exist in cleaned raw too.
    valid_raw_ids = set(raw_clean["match_id"].dropna().astype(int).tolist())
    matches_enriched = matches_enriched[matches_enriched["match_id"].isin(valid_raw_ids)].copy()

    valid_match_ids = set(matches_enriched["match_id"].dropna().astype(int).tolist())

    match_lineups, veto_steps, match_maps, map_player_stats = normalize_and_filter_child_tables(
        match_lineups=match_lineups,
        veto_steps=veto_steps,
        match_maps=match_maps,
        map_player_stats=map_player_stats,
        valid_match_ids=valid_match_ids,
    )

    matches_final = merge_raw_and_enriched(raw_clean, matches_enriched)

    raw_clean.to_csv(args.output_dir / "matches_raw_clean.csv", index=False)
    matches_final.to_csv(args.output_dir / "matches_final.csv", index=False)
    match_lineups.to_csv(args.output_dir / "match_lineups.csv", index=False)
    veto_steps.to_csv(args.output_dir / "veto_steps.csv", index=False)
    match_maps.to_csv(args.output_dir / "match_maps.csv", index=False)
    map_player_stats.to_csv(args.output_dir / "map_player_stats.csv", index=False)

    summary = build_summary(
        cutoff_date=args.cutoff_date,
        raw_clean=raw_clean,
        matches_final=matches_final,
        match_lineups=match_lineups,
        veto_steps=veto_steps,
        match_maps=match_maps,
        map_player_stats=map_player_stats,
    )
    (args.output_dir / "README_summary.txt").write_text(summary, encoding="utf-8")

    print(summary)


if __name__ == "__main__":
    main()