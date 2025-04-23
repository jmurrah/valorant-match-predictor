import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from typing import Callable
from collections import defaultdict, Counter

from valorant_match_predictor import (
    DATAFRAME_BY_YEAR_TYPE,
    PowerRatingNeuralNetwork,
    MatchPredictorNeuralNetwork,
    read_in_data,
    transform_data,
    print_transformed_data_structure,
)


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
    head_to_head_stats: pd.DataFrame,
) -> list[float]:
    return [
        head_to_head_stats["Round Win Pct"].values[0],
        head_to_head_stats["Map Win Pct"].values[0],
    ]


def create_team_pr_feature(
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


def create_match_input_tensors(
    pr_nn: PowerRatingNeuralNetwork,
    matchups_data: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_names = [
        "Head-to-Head Round Win %",
        "Head-to-Head Map Win %",
    ]
    team_a_features = []
    team_b_features = []
    win_probabilities = []

    for matchup in matchups_data["Matchup"].unique():
        matchup_data = matchups_data[matchups_data["Matchup"] == matchup]
        team_a, team_b = matchup.split("_vs_")

        team_a_vs_b_stats = matchup_data[
            (matchup_data["Team"] == "A") & (matchup_data["Opponent"] == "B")
        ]
        team_b_vs_a_stats = matchup_data[
            (matchup_data["Team"] == "B") & (matchup_data["Opponent"] == "A")
        ]

        # pr_nn is your trained PowerRatingNeuralNetwork
        with torch.no_grad():
            pr_vector = pr_nn.encode(team_tensor)  # shape (1, latent_dim)
        # or, if latent_dim=1:
        pr_score = pr_vector.item()  # a float

        team_a_feature = create_team_feature(team_a_vs_b_stats)
        team_b_feature = create_team_feature(team_b_vs_a_stats)

        team_a_features.append(team_a_feature)
        team_b_features.append(team_b_feature)

        win_probabilities.append(team_a_vs_b_stats["Map Win Pct"].values[0])

    team_a_tensor = torch.tensor(team_a_features, dtype=torch.float32)
    team_b_tensor = torch.tensor(team_b_features, dtype=torch.float32)
    win_probabilities_tensor = torch.tensor(
        win_probabilities, dtype=torch.float32
    ).unsqueeze(1)

    return team_a_tensor, team_b_tensor, win_probabilities_tensor, feature_names


def create_pr_input_tensor(
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
    team_pr_tensor = torch.cat([team_a_pr_tensor, team_b_pr_tensor], dim=0)

    return team_pr_tensor, feature_names


def create_final_pr_feature_vector(
    models: list[PowerRatingNeuralNetwork],
) -> Callable[[torch.Tensor], torch.Tensor]:
    def final_pr_vector(team_features: torch.Tensor) -> torch.Tensor:
        for m in models:
            m.eval()
        with torch.no_grad():
            preds = torch.stack([m.encode(team_features) for m in models], dim=0)
            return preds

    return final_pr_vector


def get_power_rating_model_feature_vector(transformed_data: DATAFRAME_BY_YEAR_TYPE):
    scaler = StandardScaler()
    yearly_team_pr_models = []
    all_team_features = []
    year_data_cache = {}

    for year, year_data in transformed_data.items():
        players_stats = year_data["players_stats"]["team_players_stats"]
        matchups_data = year_data["matches"]["teams_matchups_stats"]

        team_pr_tensor, feature_names = create_pr_input_tensor(
            players_stats, matchups_data
        )

        mask = ~(torch.isnan(team_pr_tensor).any(dim=1))
        team_pr_tensor = team_pr_tensor[mask]

        year_data_cache[year] = {
            "team_pr_tensor": team_pr_tensor,
            "feature_names": feature_names,
        }
        all_team_features.append(team_pr_tensor.numpy())

    combined_features = np.vstack(all_team_features)
    scaler.fit(combined_features)
    for year in transformed_data.keys():
        print(f"Training for year: {year}")
        team_pr_tensor = year_data_cache[year]["team_pr_tensor"]
        feature_names = year_data_cache[year]["feature_names"]

        team_pr_features_scaled = scaler.transform(team_pr_tensor.numpy())
        team_pr_tensors = torch.tensor(team_pr_features_scaled, dtype=torch.float32)

        team_pr_nn = PowerRatingNeuralNetwork(input_size=len(feature_names))
        team_pr_nn.train_model(team_pr_tensors)

        yearly_team_pr_models.append(team_pr_nn)

    final_team_pr_feature_vector = create_final_pr_feature_vector(yearly_team_pr_models)
    return final_team_pr_feature_vector


def train_match_predictor_model(pr_feature_vector, transformed_data):
    yearly_team_pr_models = []
    all_team_features = []
    year_data_cache = {}

    for year, year_data in transformed_data.items():
        players_stats = year_data["players_stats"]["team_players_stats"]
        matchups_data = year_data["matches"]["teams_matchups_stats"]

        team_a_tensor, team_b_tensor, win_probabilities_tensor, feature_names = (
            create_match_input_tensors(matchups_data)
        )

        team_a_features = 1
        team_b_features = 1
        match_predictor_nn = MatchPredictorNeuralNetwork(
            len(team_a_features) + len(team_b_features)
        )
        match_predictor_nn.train_model(team_a_tensor, team_b_tensor)


def train(years):
    dataframes_by_year = read_in_data("data", years)
    transformed_data = transform_data(dataframes_by_year)
    pr_feature_vector = get_power_rating_model_feature_vector(transformed_data)


def test(years):
    pass


if __name__ == "__main__":
    torch.set_printoptions(
        threshold=int(1e8),  # max number of elements before truncating
        edgeitems=3,  # how many items to show at beginning/end of each dimension
        linewidth=200,  # wrap line length
        precision=4,  # decimal precision
    )
    set_pandas_options()
    train(years=["2023"])
    test(years=["2025"])
