import pandas as pd

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
            "Total Maps": total_maps,
            "Total Round Wins": total_round_wins,
            "Total Round Losses": total_round_losses,
            "Total Map Wins": total_map_wins,
            "Total Map Losses": total_map_losses,
            "Round Win Pct": round_win_pct,
            "Map Win Pct": map_win_pct,
        }
    )


def get_matchup_stats(maps_scores_filtered, team_a, team_b) -> pd.DataFrame:
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

    return pd.DataFrame(
        {
            f"{team_a}_vs_{team_b}": {
                # Team A vs Team B stats
                "Total Maps vs B": team_a_vs_b_stats["Total Maps"],
                "Total Round Wins vs B": team_a_vs_b_stats["Total Round Wins"],
                "Total Round_Losses vs B": team_a_vs_b_stats["Total Round Losses"],
                "Total Map_Wins vs B": team_a_vs_b_stats["Total Map Wins"],
                "Total Map_Losses vs B": team_a_vs_b_stats["Total Map Losses"],
                "Round Win Pct vs B": team_a_vs_b_stats["Round Win Pct"],
                "Map Win Pct vs B": team_a_vs_b_stats["Map Win Pct"],
                # Team A vs ALL OTHER TEAMS (excluding Team B) stats
                "Total Maps vs Others": team_a_vs_others_stats["Total Maps"],
                "Total Round Wins vs Others": team_a_vs_others_stats[
                    "Total Round Wins"
                ],
                "Total Round Losses vs Others": team_a_vs_others_stats[
                    "Total Round Losses"
                ],
                "Total Map_Wins vs Others": team_a_vs_others_stats["Total Map Wins"],
                "Total Map_Losses vs Others": team_a_vs_others_stats[
                    "Total Map Losses"
                ],
                "Round Win Pct vs Others": team_a_vs_others_stats["Round Win Pct"],
                "Map_Win Pct vs Others": team_a_vs_others_stats["Map Win Pct"],
            }
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

    for team_a in all_teams:
        team_a_as_a = maps_scores_filtered[maps_scores_filtered["Team A"] == team_a]
        team_a_as_b = maps_scores_filtered[maps_scores_filtered["Team B"] == team_a]

        opponents = pd.concat([team_a_as_a["Team B"], team_a_as_b["Team A"]]).unique()
        for team_b in opponents:
            if team_a == team_b:
                continue
            matchup_stats = get_matchup_stats(maps_scores_filtered, team_a, team_b)


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
        transformed_dataframes_by_year[year]["matches"]["teams_matchup_stats"] = (
            transform_maps_scores(dataframes_by_year[year]["matches"]["maps_scores"])
        )

    return transformed_dataframes_by_year
