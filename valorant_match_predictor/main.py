import torch
import pandas as pd

from typing import Callable

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

    for matchup in matchups_data["Matchup"].unique():
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

    return team_a_tensor, team_b_tensor, feature_names


def create_final_pr_model(
    models_a: list[PowerRatingNeuralNetwork], models_b: list[PowerRatingNeuralNetwork]
) -> Callable[[torch.Tensor, torch.Tensor], tuple[float, float]]:
    def final_prediction(
        team_a_features: torch.Tensor, team_b_features: torch.Tensor
    ) -> tuple[float, float]:
        for m in models_a + models_b:
            m.eval()

        with torch.no_grad():
            preds_a = torch.stack([m(team_a_features) for m in models_a])
            preds_b = torch.stack([m(team_b_features) for m in models_b])
            avg_a = torch.mean(preds_a, dim=0).item()
            avg_b = torch.mean(preds_b, dim=0).item()
        return avg_a, avg_b

    return final_prediction


def train(years) -> None:
    dataframes_by_year = read_in_data("data", years)
    transformed_data = transform_data(dataframes_by_year)

    yearly_team_a_pr_models = []
    yearly_team_b_pr_models = []
    pred_pr_a = pred_pr_b = 0.0
    for year, year_data in transformed_data.items():
        print(f"Training for year: {year}")
        players_stats = year_data["players_stats"]["team_players_stats"]
        matchups_data = year_data["matches"]["teams_matchups_stats"]

        team_a_pr_tensors, team_b_pr_tensors, feature_names = create_pr_input_tensors(
            players_stats, matchups_data
        )

        print(team_a_pr_tensors)
        break
        team_a_pr_nn = PowerRatingNeuralNetwork(input_size=len(feature_names))
        team_b_pr_nn = PowerRatingNeuralNetwork(input_size=len(feature_names))

        yearly_team_a_pr_models.append(team_a_pr_nn)
        yearly_team_b_pr_models.append(team_b_pr_nn)
        break

    final_model = create_final_model(yearly_team_a_pr_models)
    # pred_win_probability = final_model(pred_team_a, pred_team_b)
    # print(f"Predicted win probability for Team A: {pred_win_probability:.4f}")
    # print(f"Actual historical win rate: {expected_win_probabilities[0].item():.4f}")


def test(years):
    pass


if __name__ == "__main__":
    set_pandas_options()
    train(years=["2022", "2023"])
    test(years=["2025"])
