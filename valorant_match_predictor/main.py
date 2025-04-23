import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from typing import Callable
from collections import defaultdict, Counter

from valorant_match_predictor import PowerRatingNeuralNetwork
from valorant_match_predictor import read_in_data, transform_data
from valorant_match_predictor import print_transformed_data_structure


def set_pandas_options() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_rows", None)


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
    team_a_pr_features = []
    team_b_pr_features = []

    unique_matches = matchups_data["Matchup"].unique()
    for matchup in unique_matches:
        matchup_data = matchups_data[matchups_data["Matchup"] == matchup]
        team_a, team_b = matchup.split("_vs_")

        team_a_players_stats = players_stats[players_stats["Teams"] == team_a]
        team_b_players_stats = players_stats[players_stats["Teams"] == team_b]
        team_a_vs_others_stats = matchup_data[
            (matchup_data["Team"] == "A") & (matchup_data["Opponent"] == "Others")
        ]
        team_b_vs_others_stats = matchup_data[
            (matchup_data["Team"] == "B") & (matchup_data["Opponent"] == "Others")
        ]

        team_a_pr_features.append(
            create_team_feature(team_a_players_stats, team_a_vs_others_stats)
        )
        team_b_pr_features.append(
            create_team_feature(team_b_players_stats, team_b_vs_others_stats)
        )

    team_a_pr_tensor = torch.tensor(team_a_pr_features, dtype=torch.float32)
    team_b_pr_tensor = torch.tensor(team_b_pr_features, dtype=torch.float32)

    return team_a_pr_tensor, team_b_pr_tensor, feature_names


def create_final_pr_model(
    models: list[PowerRatingNeuralNetwork],
) -> Callable[[torch.Tensor], float]:
    def final_power_rating(team_features: torch.Tensor) -> float:
        for m in models:
            m.eval()
        with torch.no_grad():
            preds = torch.stack([m.encode(team_features) for m in models])
            return torch.mean(preds, dim=0).item()

    return final_power_rating


def train(years) -> None:
    scaler = StandardScaler()
    dataframes_by_year = read_in_data("data", years)
    transformed_data = transform_data(dataframes_by_year)

    yearly_team_pr_models = []
    all_team_features = []
    year_data_cache = {}

    for year, year_data in transformed_data.items():
        players_stats = year_data["players_stats"]["team_players_stats"]
        matchups_data = year_data["matches"]["teams_matchups_stats"]

        team_a_pr_tensor, team_b_pr_tensor, feature_names = create_pr_input_tensors(
            players_stats, matchups_data
        )
        year_data_cache[year] = {
            "team_a_pr_tensor": team_a_pr_tensor,
            "team_b_pr_tensor": team_b_pr_tensor,
            "feature_names": feature_names,
        }
        all_team_features.append(team_a_pr_tensor.numpy())
        all_team_features.append(team_b_pr_tensor.numpy())

    combined_features = np.vstack(all_team_features)
    scaler.fit(combined_features)
    for year in transformed_data.keys():
        print(f"Training for year: {year}")
        team_a_pr_tensor = year_data_cache[year]["team_a_pr_tensor"]
        team_b_pr_tensor = year_data_cache[year]["team_b_pr_tensor"]
        feature_names = year_data_cache[year]["feature_names"]

        team_a_pr_features_scaled = scaler.transform(team_a_pr_tensor.numpy())
        team_b_pr_features_scaled = scaler.transform(team_b_pr_tensor.numpy())

        team_a_pr_tensors = torch.tensor(team_a_pr_features_scaled, dtype=torch.float32)
        team_b_pr_tensors = torch.tensor(team_b_pr_features_scaled, dtype=torch.float32)

        team_a_pr_nn = PowerRatingNeuralNetwork(input_size=len(feature_names))
        team_b_pr_nn = PowerRatingNeuralNetwork(input_size=len(feature_names))
        team_a_pr_nn.train_model(team_a_pr_tensors)
        team_b_pr_nn.train_model(team_b_pr_tensors)

        yearly_team_pr_models.extend([team_a_pr_nn, team_b_pr_nn])

    final_team_pr_model = create_final_pr_model(yearly_team_pr_models)

    # Sample output using the last year's data
    last_year = max(years)
    sample_tensor_a = torch.tensor(
        [
            scaler.transform(
                year_data_cache[last_year]["team_a_pr_tensor"][0:1].numpy()
            )[0]
        ],
        dtype=torch.float32,
    )
    sample_tensor_b = torch.tensor(
        [
            scaler.transform(
                year_data_cache[last_year]["team_b_pr_tensor"][0:1].numpy()
            )[0]
        ],
        dtype=torch.float32,
    )

    # Display the input features
    print("\nSample input features for Team A (scaled):")
    display_features(feature_names, sample_tensor_a[0])

    print("\nSample input features for Team B (scaled):")
    display_features(feature_names, sample_tensor_b[0])

    # Get and display the power ratings
    pr_a = final_team_pr_model(sample_tensor_a)
    pr_b = final_team_pr_model(sample_tensor_b)

    print(f"\nPower Rating for sample Team A: {pr_a:.4f}")
    print(f"Power Rating for sample Team B: {pr_b:.4f}")


def test(years):
    pass


if __name__ == "__main__":
    set_pandas_options()
    train(years=["2023"])
    test(years=["2025"])
