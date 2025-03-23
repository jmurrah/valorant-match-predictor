import os

import pandas as pd

from collections import defaultdict


DATAFRAME_BY_YEAR_TYPE = dict[str, dict[str, dict[str, pd.DataFrame]]]
GLOBAL_TOURNAMENTS = ["Masters", "Valorant Champions"]
REGIONAL_TOURNAMENTS = ["Americas", "EMEA", "Pacific", "China"]


def filter_out_non_regional_tournaments(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe[
        dataframe["Tournament"].str.contains(
            "|".join(REGIONAL_TOURNAMENTS), case=False, na=False
        )
    ]


def transform_players_stats(players_stats_df: pd.DataFrame) -> pd.DataFrame:
    USEFUL_PLAYER_COLUMNS = [
        "Teams",
        "Rating",
        "Average Combat Score",
        "Kills:Deaths",
        "Kill, Assist, Trade, Survive %",
        "First Kills Per Round",
        "First Deaths Per Round",
    ]
    players_stats_filtered = (
        filter_out_non_regional_tournaments(players_stats_df)[USEFUL_PLAYER_COLUMNS]
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
        .groupby(USEFUL_PLAYER_COLUMNS[:1])[USEFUL_PLAYER_COLUMNS[1:]]
        .mean()
        .reset_index()
    )
    return players_stats_transformed


def calculate_team_stats(
    matches_as_team_a: pd.DataFrame, matches_as_team_b: pd.DataFrame
) -> pd.DataFrame:
    total_maps = len(matches_as_team_a) + len(matches_as_team_b)
    total_round_wins = sum(matches_as_team_a["Team A Map Score"]) + sum(
        matches_as_team_b["Team B Map Score"]
    )
    total_round_losses = sum(matches_as_team_a["Team B Map Score"]) + sum(
        matches_as_team_b["Team A Map Score"]
    )
    total_map_wins = sum(
        matches_as_team_a["Team A Map Score"] > matches_as_team_a["Team B Map Score"]
    ) + sum(
        matches_as_team_b["Team B Map Score"] > matches_as_team_b["Team A Map Score"]
    )
    total_map_losses = total_maps - total_map_wins
    round_win_pct = (
        total_round_wins / (total_round_wins + total_round_losses)
        if (total_round_wins + total_round_losses) > 0
        else 0
    )
    map_win_pct = total_map_wins / total_maps if total_maps > 0 else 0

    return pd.DataFrame(
        {
            "Total Maps": [total_maps],
            "Total Round Wins": [total_round_wins],
            "Total Round Losses": [total_round_losses],
            "Total Map Wins": [total_map_wins],
            "Total Map Losses": [total_map_losses],
            "Round Win Pct": [round_win_pct],
            "Map Win Pct": [map_win_pct],
        }
    )


def get_matchup_stats(
    maps_scores_filtered: pd.DataFrame, team_a: str, team_b: str
) -> pd.DataFrame:
    # Team A statistics
    team_a_as_a = maps_scores_filtered[maps_scores_filtered["Team A"] == team_a]
    team_a_as_b = maps_scores_filtered[maps_scores_filtered["Team B"] == team_a]

    # Team A vs Team B statistics
    team_a_vs_b = team_a_as_a[team_a_as_a["Team B"] == team_b]
    team_b_vs_a = team_a_as_b[team_a_as_b["Team A"] == team_b]
    team_a_vs_b_stats = calculate_team_stats(team_a_vs_b, team_b_vs_a)

    # Team A vs ALL OTHER TEAMS (excluding Team B) statistics
    team_a_vs_others = team_a_as_a[team_a_as_a["Team B"] != team_b]
    others_vs_team_a = team_a_as_b[team_a_as_b["Team A"] != team_b]
    team_a_vs_others_stats = calculate_team_stats(team_a_vs_others, others_vs_team_a)

    # Team B statistics
    team_b_as_a = maps_scores_filtered[maps_scores_filtered["Team A"] == team_b]
    team_b_as_b = maps_scores_filtered[maps_scores_filtered["Team B"] == team_b]

    # Team B vs Team A statistics
    team_b_vs_a_stats = calculate_team_stats(team_b_vs_a, team_a_vs_b)

    # Team B vs ALL OTHER TEAMS (excluding Team A) statistics
    team_b_vs_others = team_b_as_a[team_b_as_a["Team B"] != team_a]
    others_vs_team_b = team_b_as_b[team_b_as_b["Team A"] != team_a]
    team_b_vs_others_stats = calculate_team_stats(team_b_vs_others, others_vs_team_b)

    return pd.DataFrame(
        {
            "Matchup": [f"{team_a}_vs_{team_b}"] * 4,
            "Team": ["A", "A", "B", "B"],
            "Opponent": ["B", "Others", "A", "Others"],
            "Total Maps": [
                team_a_vs_b_stats["Total Maps"].values[0],
                team_a_vs_others_stats["Total Maps"].values[0],
                team_b_vs_a_stats["Total Maps"].values[0],
                team_b_vs_others_stats["Total Maps"].values[0],
            ],
            "Total Round Wins": [
                team_a_vs_b_stats["Total Round Wins"].values[0],
                team_a_vs_others_stats["Total Round Wins"].values[0],
                team_b_vs_a_stats["Total Round Wins"].values[0],
                team_b_vs_others_stats["Total Round Wins"].values[0],
            ],
            "Total Round Losses": [
                team_a_vs_b_stats["Total Round Losses"].values[0],
                team_a_vs_others_stats["Total Round Losses"].values[0],
                team_b_vs_a_stats["Total Round Losses"].values[0],
                team_b_vs_others_stats["Total Round Losses"].values[0],
            ],
            "Total Map Wins": [
                team_a_vs_b_stats["Total Map Wins"].values[0],
                team_a_vs_others_stats["Total Map Wins"].values[0],
                team_b_vs_a_stats["Total Map Wins"].values[0],
                team_b_vs_others_stats["Total Map Wins"].values[0],
            ],
            "Total Map Losses": [
                team_a_vs_b_stats["Total Map Losses"].values[0],
                team_a_vs_others_stats["Total Map Losses"].values[0],
                team_b_vs_a_stats["Total Map Losses"].values[0],
                team_b_vs_others_stats["Total Map Losses"].values[0],
            ],
            "Round Win Pct": [
                team_a_vs_b_stats["Round Win Pct"].values[0],
                team_a_vs_others_stats["Round Win Pct"].values[0],
                team_b_vs_a_stats["Round Win Pct"].values[0],
                team_b_vs_others_stats["Round Win Pct"].values[0],
            ],
            "Map Win Pct": [
                team_a_vs_b_stats["Map Win Pct"].values[0],
                team_a_vs_others_stats["Map Win Pct"].values[0],
                team_b_vs_a_stats["Map Win Pct"].values[0],
                team_b_vs_others_stats["Map Win Pct"].values[0],
            ],
        }
    )


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
    maps_scores_filtered = (
        filter_out_non_regional_tournaments(maps_scores_df)[USEFUL_MAPS_SCORES_COLUMNS]
        .copy()
        .dropna()
    )
    maps_scores_filtered["Team A Map Score"] = maps_scores_filtered[
        "Team A Score"
    ] + maps_scores_filtered["Team A Overtime Score"].fillna(0)
    maps_scores_filtered["Team B Map Score"] = maps_scores_filtered[
        "Team B Score"
    ] + maps_scores_filtered["Team B Overtime Score"].fillna(0)

    all_teams = pd.concat(
        [maps_scores_filtered["Team A"], maps_scores_filtered["Team B"]]
    ).unique()

    all_matchup_stats = {}
    for team_a in all_teams[:10]:
        team_a_as_a = maps_scores_filtered[maps_scores_filtered["Team A"] == team_a]
        team_a_as_b = maps_scores_filtered[maps_scores_filtered["Team B"] == team_a]

        opponents = pd.concat([team_a_as_a["Team B"], team_a_as_b["Team A"]]).unique()
        for team_b in opponents:
            matchup_stats = get_matchup_stats(maps_scores_filtered, team_a, team_b)
            all_matchup_stats[f"{team_a}_vs_{team_b}"] = matchup_stats

    combined_df = pd.concat(all_matchup_stats.values()).reset_index()
    repeat_count = len(combined_df) // len(all_matchup_stats)

    matchup_identifiers = []
    for key in all_matchup_stats.keys():
        matchup_identifiers.extend([key] * repeat_count)

    combined_df["Matchup"] = matchup_identifiers
    return combined_df


def transform_data(
    dataframes_by_year: DATAFRAME_BY_YEAR_TYPE,
) -> DATAFRAME_BY_YEAR_TYPE:
    # NOTE: ALL DATA IS GROUPED BY: (Year, Tournament, Team)
    transformed_dataframes_by_year = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    for year in dataframes_by_year:
        print(f"Transforming data for {year}!")
        transformed_dataframes_by_year[year]["players_stats"]["team_players_stats"] = (
            transform_players_stats(
                dataframes_by_year[year]["players_stats"]["players_stats"]
            )
        )
        transformed_dataframes_by_year[year]["matches"]["teams_matchups_stats"] = (
            transform_maps_scores(dataframes_by_year[year]["matches"]["maps_scores"])
        )

    return transformed_dataframes_by_year


def read_in_data(folder_name: str = "data") -> DATAFRAME_BY_YEAR_TYPE:
    # Dataset -> SEE README.md
    USEFUL_CSVS = {
        "players_stats": ["players_stats"],
        "matches": [
            "scores",
            "overview",
            "maps_scores",
            "win_loss_methods_round_number",
            "eco_stats",
            "maps_played",
        ],
    }
    base_path = os.path.join(os.path.abspath(os.getcwd()), folder_name)
    subfolders = [
        subfolder
        for subfolder in os.listdir(base_path)
        if subfolder.startswith("vct_20") and subfolder != "vct_2021"
    ]
    csv_folders_and_basenames = [
        [data_folder, csv_basename]
        for data_folder, csv_basenames in USEFUL_CSVS.items()
        for csv_basename in csv_basenames
    ]
    dataframes_by_year = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for subfolder in subfolders:
        year = subfolder.split("_")[-1]  # NOTE: subfolder = vct_20XX
        print(f"Reading in {year} data!")

        for data_folder, csv_basename in csv_folders_and_basenames:
            full_path = os.path.join(
                base_path, subfolder, data_folder, csv_basename + ".csv"
            )
            dataframes_by_year[year][data_folder][csv_basename] = pd.read_csv(
                full_path, low_memory=False
            )

    return dataframes_by_year
