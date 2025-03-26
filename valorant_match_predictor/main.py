import torch
import pandas as pd

from typing import Callable

from valorant_match_predictor import NeuralNetwork
from valorant_match_predictor import read_in_data, transform_data


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
    head_to_head_stats: pd.DataFrame,
    team_vs_others_stats: pd.DataFrame,
) -> list[float]:
    return [
        team_players_stats["Rating"].mean(),
        team_players_stats["Average Combat Score"].mean(),
        team_players_stats["Kills:Deaths"].mean(),
        team_players_stats["Kill, Assist, Trade, Survive %"].mean(),
        team_players_stats["First Kills Per Round"].mean(),
        team_players_stats["First Deaths Per Round"].mean(),
        head_to_head_stats["Round Win Pct"].values[0],
        head_to_head_stats["Map Win Pct"].values[0],
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


def create_final_model(
    models: list[NeuralNetwork],
) -> Callable[[torch.Tensor, torch.Tensor], float]:

    def final_prediction(
        team_a_features: torch.Tensor, team_b_features: torch.Tensor
    ) -> float:
        for model in models:
            model.eval()

        with torch.no_grad():
            predictions = torch.stack(
                [model(team_a_features, team_b_features) for model in models]
            )
            return torch.mean(predictions, dim=0).item()

    return final_prediction


def main() -> None:
    set_pandas_options()
    dataframes_by_year = read_in_data()
    transformed_data = transform_data(dataframes_by_year)

    yearly_models = []
    pred_team_a, pred_team_b = None, None
    for year, year_data in transformed_data.items():
        print(f"Training for year: {year}")
        players_stats = year_data["players_stats"]["team_players_stats"]
        matchups_data = year_data["matches"]["teams_matchups_stats"]

        team_a_tensors, team_b_tensors, expected_win_probabilities, feature_names = (
            create_input_tensors(players_stats, matchups_data)
        )

        pred_team_a = team_a_tensors[0].unsqueeze(0)
        pred_team_b = team_b_tensors[0].unsqueeze(0)
        nn = NeuralNetwork(input_size=len(feature_names))
        nn.train_model(
            team_a_tensors,
            team_b_tensors,
            expected_win_probabilities,
        )

        yearly_models.append(nn)
        break

    final_model = create_final_model(yearly_models)
    pred_win_probability = final_model(pred_team_a, pred_team_b)
    print(f"Predicted win probability for Team A: {pred_win_probability:.4f}")
    print(f"Actual historical win rate: {expected_win_probabilities[0].item():.4f}")


if __name__ == "__main__":
    main()
