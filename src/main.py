import torch
import os
import pandas as pd

from collections import defaultdict

from neural_network import NeuralNetwork
from transform import DATAFRAME_BY_YEAR_TYPE, transform_data


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
        if subfolder.startswith("vct_20")
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
                base_path, subfolder, data_folder, csv_filename + ".csv"
            )
            dataframes_by_year[year][data_folder][csv_basename] = pd.read_csv(
                full_path, low_memory=False
            )

    return dataframes_by_year


def create_final_model(models: list[NeuralNetwork]):
    def final_prediction(team_a_features, team_b_features):
        for model in models:
            model.eval()

        with torch.no_grad():
            predictions = torch.stack(
                [model(team_a_features, team_b_features) for model in models]
            )
            return torch.mean(predictions, dim=0)

    return final_prediction


def main() -> None:
    dataframes_by_year = read_in_data()
    transformed_data = transform_data(dataframes_by_year)

    yearly_models = []
    for year, year_data in transformed_data.items():
        print(year_data)
        matchups = year_data["matches"]["maps_scores"]["Match Name"]
        # print(matchups)
        nn = NeuralNetwork()
        # train with all the matches this year
        yearly_models.append(nn)

    final_model = create_final_model(yearly_models)


if __name__ == "__main__":
    main()
