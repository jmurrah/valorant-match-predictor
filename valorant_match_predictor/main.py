import torch
import pandas as pd

from typing import Callable
from collections import defaultdict

from valorant_match_predictor import PowerRatingNeuralNetwork
from valorant_match_predictor import read_in_data, transform_data
from valorant_match_predictor import print_transformed_data_structure


def set_pandas_options() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.expand_frame_repr", False)


def display_features(
    feature_names: list[str], tensor: torch.Tensor, team_name="Team"
) -> None:
    print(f"{team_name} features:")
    for i, (name, value) in enumerate(zip(feature_names, tensor)):
        print(f"{i+1}. {name}: {value.item():.4f}")


def create_team_feature(
    team_players_stats: pd.DataFrame,
    team_vs_others_stats: pd.DataFrame,
) -> list[float]:
    return [
        team_players_stats["Rating"].mean(),
        team_players_stats["Average Combat Score"].mean(),
        team_players_stats["Kills:Deaths"].mean(),
        team_players_stats["Kill, Assist, Trade, Survive %"].mean(),
        team_players_stats["First Kills Per Round"].mean(),
        team_players_stats["First Deaths Per Round"].mean(),
        team_vs_others_stats["Round Win Pct"].values[0],
        team_vs_others_stats["Map Win Pct"].values[0],
    ]


def create_input_tensors(
    players_stats: pd.DataFrame, matchups_data: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_names = [
        "Rating",
        "Average Combat Score",
        "Kills:Deaths",
        "Kill, Assist, Trade, Survive %",
        "First Kills Per Round",
        "First Deaths Per Round",
        "Head-to-Head Round Win %",
        "Head-to-Head Map Win %",
        "Team-vs-Others Round Win %",
        "Team-vs-Others Map Win %",
    ]
    team_a_features = []
    team_b_features = []
    win_probabilities = []

    for matchup in matchups_data["Matchup"].unique():
        matchup_data = matchups_data[matchups_data["Matchup"] == matchup]
        team_a, team_b = matchup.split("_vs_")
        team_a_players_stats = players_stats[players_stats["Teams"] == team_a]
        team_b_players_stats = players_stats[players_stats["Teams"] == team_b]

        team_a_vs_b_stats = matchup_data[
            (matchup_data["Team"] == "A") & (matchup_data["Opponent"] == "B")
        ]
        team_a_vs_others_stats = matchup_data[
            (matchup_data["Team"] == "A") & (matchup_data["Opponent"] == "Others")
        ]
        team_b_vs_a_stats = matchup_data[
            (matchup_data["Team"] == "B") & (matchup_data["Opponent"] == "A")
        ]
        team_b_vs_others_stats = matchup_data[
            (matchup_data["Team"] == "B") & (matchup_data["Opponent"] == "Others")
        ]

        team_a_feature = create_team_feature(
            team_a_players_stats, team_a_vs_b_stats, team_a_vs_others_stats
        )
        team_b_feature = create_team_feature(
            team_b_players_stats, team_b_vs_a_stats, team_b_vs_others_stats
        )

        team_a_features.append(team_a_feature)
        team_b_features.append(team_b_feature)
        win_probabilities.append(
            team_a_vs_b_stats["Round Win Pct"].values[0]
        )  # could also try 'Map Win Pct'

    team_a_tensor = torch.tensor(team_a_features, dtype=torch.float32)
    team_b_tensor = torch.tensor(team_b_features, dtype=torch.float32)
    win_probabilities_tensor = torch.tensor(
        win_probabilities, dtype=torch.float32
    ).unsqueeze(1)

    return team_a_tensor, team_b_tensor, win_probabilities_tensor, feature_names


def create_pr_input_tensors(
    players_stats: pd.DataFrame, matchups_data: pd.DataFrame
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    feature_names = [
        "Rating",
        "Average Combat Score",
        "Kills:Deaths",
        "Kill, Assist, Trade, Survive %",
        "First Kills Per Round",
        "First Deaths Per Round",
        "Team-vs-Others Round Win %",
        "Team-vs-Others Map Win %",
    ]
    team_a_features = []
    team_b_features = []
    team_stats = defaultdict(int)

    # print(matchups_data["Matchup"])
    # return
    unique_matches = matchups_data["Matchup"].unique()
    for matchup in unique_matches:
        matchup_data = matchups_data[matchups_data["Matchup"] == matchup]
        print(matchup_data)
        # break
        # Filter to only include head-to-head stats
        team_a_vs_b = matchup_data[
            (matchup_data["Team"] == "A") & (matchup_data["Opponent"] == "B")
        ]
        team_b_vs_a = matchup_data[
            (matchup_data["Team"] == "B") & (matchup_data["Opponent"] == "A")
        ]

        # Get values (these are single values, not sums)
        team_a_round_wins = team_a_vs_b["Total Round Wins"].values[0]
        team_a_map_wins = team_a_vs_b["Total Map Wins"].values[0]

        team_b_round_wins = team_b_vs_a["Total Round Wins"].values[0]
        team_b_map_wins = team_b_vs_a["Total Map Wins"].values[0]

        # Total rounds and maps
        total_rounds = team_a_round_wins + team_b_round_wins
        total_maps = team_a_map_wins + team_b_map_wins

        team_a_players_stats = players_stats[players_stats["Teams"] == team_a]
        team_b_players_stats = players_stats[players_stats["Teams"] == team_b]

        team_a_vs_others_stats = matchup_data[
            (matchup_data["Team"] == "A") & (matchup_data["Opponent"] == "Others")
        ]
        team_b_vs_others_stats = matchup_data[
            (matchup_data["Team"] == "B") & (matchup_data["Opponent"] == "Others")
        ]

        team_a_feature = create_team_feature(
            team_a_players_stats, team_a_vs_others_stats
        )
        team_b_feature = create_team_feature(
            team_b_players_stats, team_b_vs_others_stats
        )

        team_a_features.append(team_a_feature)
        team_b_features.append(team_b_feature)

    team_a_tensor = torch.tensor(team_a_features, dtype=torch.float32)
    team_b_tensor = torch.tensor(team_b_features, dtype=torch.float32)

    return team_a_tensor, team_b_tensor, feature_names, len(unique_matches)


def create_final_pr_model(
    models: list[PowerRatingNeuralNetwork],
) -> Callable[[torch.Tensor], float]:
    def final_power_rating(team_features: torch.Tensor) -> float:
        for m in models:
            m.eval()
        with torch.no_grad():
            preds = torch.stack([m(team_features) for m in models])
            return torch.mean(preds, dim=0).item()

    return final_power_rating


def train(years) -> None:
    dataframes_by_year = read_in_data("data", years)
    transformed_data = transform_data(dataframes_by_year)

    # get power ratings for all the teams
    yearly_team_a_pr_models = []
    yearly_team_b_pr_models = []
    # print_transformed_data_structure(transformed_data)
    # return
    for year, year_data in transformed_data.items():
        print(f"Training for year: {year}")
        players_stats = year_data["players_stats"]["team_players_stats"]
        matchups_data = year_data["matches"]["teams_matchups_stats"]

        team_a_pr_tensors, team_b_pr_tensors, feature_names, num_matches = (
            create_pr_input_tensors(players_stats, matchups_data)
        )
        break
        print(team_a_pr_tensors)
        team_a_pr_nn = PowerRatingNeuralNetwork(input_size=len(feature_names))
        team_b_pr_nn = PowerRatingNeuralNetwork(input_size=len(feature_names))

        team_a_pr_nn.train_model(team_a_pr_tensors)
        team_b_pr_nn.train_model(team_b_pr_tensors)

        # all_pr_a, all_pr_b = []
        # # Iterate through every matchup
        # for i in range(num_matchups):
        #     single_a = team_a_pr_tensors[i].unsqueeze(0)  # shape (1, feature_dim)
        #     single_b = team_b_pr_tensors[i].unsqueeze(0)

        #     pr_a = final_team_a_pr_model(single_a)         # float
        #     pr_b = final_team_b_pr_model(single_b)

        #     all_pr_a.append(pr_a)
        #     all_pr_b.append(pr_b)

        # # Convert to tensors (or leave as lists if you prefer)
        # all_pr_a = torch.tensor(all_pr_a)  # shape (num_matchups,)
        # all_pr_b = torch.tensor(all_pr_b)

        # print("All Team A power ratings:", all_pr_a)
        # print("All Team B power ratings:", all_pr_b)

        yearly_team_a_pr_models.append(team_a_pr_nn)
        yearly_team_b_pr_models.append(team_b_pr_nn)

    final_team_a_pr_model = create_final_pr_model(yearly_team_a_pr_models)
    final_team_b_pr_model = create_final_pr_model(yearly_team_b_pr_models)

    # pred_win_probability = final_model(pred_team_a, pred_team_b)
    # print(f"Predicted win probability for Team A: {pred_win_probability:.4f}")
    # print(f"Actual historical win rate: {expected_win_probabilities[0].item():.4f}")


def test(years):
    pass


if __name__ == "__main__":
    set_pandas_options()
    train(years=["2022", "2023"])
    test(years=["2025"])
