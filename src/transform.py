import pandas as pd
import numpy as np

from collections import defaultdict


DATAFRAME_BY_YEAR_TYPE = dict[str, dict[str, dict[str, pd.DataFrame]]]
GLOBAL_TOURNAMENTS = ["Masters", "Valorant Champions"]


def filter_out_global_tournaments(dataframe: pd.DateFrame) -> pd.DataFrame:
    return dataframe[
        ~dataframe["Tournament"].str.contains(
            "|".join(GLOBAL_TOURNAMENTS), case=False, na=False
        )
    ]


def transform_players_stats(players_stats_df: pd.DateFrame) -> pd.DataFrame:
    USEFUL_PLAYER_COLUMNS = [
        "Tournament",
        "Teams",
        "Rating",
        "Average Combat Score",
        "Kills:Deaths",
        "Kill, Assist, Trade, Survive %",
        "First Kills Per Round",
        "First Deaths Per Round",
    ]
    players_stats_filtered = (
        filter_out_global_tournaments(players_stats_df)[USEFUL_PLAYER_COLUMNS]
        .copy()
        .dropna()
    )
    players_stats_filtered["Kill, Assist, Trade, Survive %"] = (
        players_stats_filtered["Kill, Assist, Trade, Survive %"]
        .str.rstrip("%")
        .astype(float)
        / 100
    )  # converts 73% -> 0.73

    players_stats_transformed = (
        players_stats_filtered[USEFUL_PLAYER_COLUMNS]
        .groupby(USEFUL_PLAYER_COLUMNS[:2])[USEFUL_PLAYER_COLUMNS[2:]]
        .mean()
        .reset_index()
    )
    return players_stats_transformed


def transform_maps_scores(maps_scores_df: pd.DataFrame) -> pd.DataFrame:
    USEFUL_MAPS_SCORES_COLUMNS = [
        "Tournament",
        "Match Name",
        "Map",
        "Team A",
        "Team A Score",
        "Team A Overtime Score",
        "Team B",
        "Team B Score",
        "Team B Overtime Score",
    ]
    maps_scores_filtered = filter_out_global_tournaments(maps_scores_df)[
        USEFUL_MAPS_SCORES_COLUMNS
    ].copy()
    maps_scores_filtered["Team A Map Score"] = maps_scores_filtered[
        "Team A Score"
    ] + maps_scores_filtered["Team A Overtime Score"].fillna(0)
    maps_scores_filtered["Team B Map Score"] = maps_scores_filtered[
        "Team B Score"
    ] + maps_scores_filtered["Team B Overtime Score"].fillna(0)

    all_teams = pd.concat(
        [maps_scores_filtered["Team A"], maps_scores_filtered["Team B"]]
    ).unique()

    for team in all_teams:
        team_a_matches = maps_scores_filtered[maps_scores_filtered["Team A"] == team]
        team_b_matches = maps_scores_filtered[maps_scores_filtered["Team B"] == team]
        total_games = len(team_a_matches) + len(team_b_matches)

        total_round_wins = sum(team_a_matches["Team A Map Score"]) + sum(
            team_b_matches["Team B Map Score"]
        )
        total_round_losses = sum(team_a_matches["Team B Map Score"]) + sum(
            team_b_matches["Team A Map Score"]
        )
        total_wins = sum(
            team_a_matches["Team A Map Score"] > team_a_matches["Team B Map Score"]
        ) + sum(team_b_matches["Team B Map Score"] > team_b_matches["Team A Map Score"])
        total_losses = total_games - total_wins


def transform_data(
    dataframes_by_year: DATAFRAME_BY_YEAR_TYPE,
) -> DATAFRAME_BY_YEAR_TYPE:
    # NOTE: ALL DATA IS GROUPED BY: (Year, Tournament, Team)
    transformed_dataframes_by_year = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    for year in dataframes_by_year:
        print(f"Transforming data for {year}!")
        transformed_dataframes_by_year[year]["players_stats"]["players_stats"] = (
            transform_players_stats(
                dataframes_by_year[year]["players_stats"]["players_stats"]
            )
        )
        transformed_dataframes_by_year[year]["matches"]["maps_scores"] = (
            transform_maps_scores(dataframes_by_year[year]["matches"]["maps_scores"])
        )

    return transformed_dataframes_by_year
