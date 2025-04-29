from collections import defaultdict
from pathlib import Path

import pandas as pd

import csv, torch

GLOBAL_TOURNAMENTS = ["Masters", "Valorant Champions"]
REGIONAL_TOURNAMENTS = ["Americas", "EMEA", "Pacific", "China"]


def load_year_thunderbird_match_odds_from_csv(year: str) -> dict[str, dict[str, float]]:
    in_file = (
        Path("scraped_data/thunderbird_match_odds")
        / f"{year}_thunderbird_match_odds.csv"
    )

    match_odds = defaultdict(dict)
    with in_file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            match_url = row["match_url"]
            team_a, team_b = row["team_a"], row["team_b"]
            odd_a, odd_b = float(row["odd_a"]), float(row["odd_b"])
            match_odds[match_url] = {team_a: odd_a, team_b: odd_b}

    return match_odds


def load_scraped_teams_matchups_stats_from_csv():
    return pd.read_csv(Path("scraped_data/matches/teams_matchups_stats.csv"))


def load_scraped_teams_players_stats_from_csv():
    return pd.read_csv(Path("scraped_data/players_stats/team_players_stats.csv"))


def set_display_options():
    torch.set_printoptions(
        threshold=int(1e8),
        edgeitems=3,
        linewidth=200,
        precision=4,
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_rows", None)
