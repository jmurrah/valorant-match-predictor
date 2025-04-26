from collections import defaultdict
from pathlib import Path

import csv


def load_year_match_odds_from_csv(year: str) -> dict[str, dict[str, float]]:
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
