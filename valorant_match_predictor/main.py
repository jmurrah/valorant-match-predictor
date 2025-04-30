import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import bootstrap

import warnings
from numpy.polynomial.polyutils import RankWarning

from valorant_match_predictor import ScaledPRModel
import joblib


from typing import Callable

from helper import (
    load_year_thunderbird_match_odds_from_csv,
    set_display_options,
    load_scraped_teams_matchups_stats_from_csv,
    load_scraped_teams_players_stats_from_csv,
)

from valorant_match_predictor import (
    DATAFRAME_BY_YEAR_TYPE,
    PowerRatingNeuralNetwork,
    MatchPredictorNeuralNetwork,
    read_in_data,
    transform_data,
    print_transformed_data_structure,
)


warnings.simplefilter("ignore", RankWarning)


def set_pandas_options() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_rows", None)


def display_features(
    feature_names: list[str], tensor: torch.Tensor, team_name="Teams"
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
    pr_vector: Callable[[torch.Tensor], torch.Tensor],
    players_stats: pd.DataFrame,
    matchups_data: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    team_a_features = []
    team_b_features = []
    win_probabilities = []

    using_matchup_url = False
    matchups = matchups_data["Matchup"].unique()
    if "Matchup URL" in matchups_data:
        using_matchup_url = True
        matchups = matchups_data["Matchup URL"].unique()

    for matchup in matchups:
        if using_matchup_url:
            matchup_data = matchups_data[matchups_data["Matchup URL"] == matchup]
            team_a, team_b = matchup_data["Matchup"].iloc[0].split("_vs_")
        else:
            matchup_data = matchups_data[matchups_data["Matchup"] == matchup]
            team_a, team_b = matchup.split("_vs_")

        team_a_players_stats = players_stats[players_stats["Teams"] == team_a]
        team_b_players_stats = players_stats[players_stats["Teams"] == team_b]
        team_a_vs_b_stats = matchup_data[
            (matchup_data["Teams"] == "A") & (matchup_data["Opponent"] == "B")
        ]
        team_b_vs_a_stats = matchup_data[
            (matchup_data["Teams"] == "B") & (matchup_data["Opponent"] == "A")
        ]
        team_a_vs_others_stats = matchup_data[
            (matchup_data["Teams"] == "A") & (matchup_data["Opponent"] == "Others")
        ]
        team_b_vs_others_stats = matchup_data[
            (matchup_data["Teams"] == "B") & (matchup_data["Opponent"] == "Others")
        ]

        team_a_pr_feature = create_team_pr_feature(
            team_a_players_stats, team_a_vs_others_stats
        )
        team_b_pr_feature = create_team_pr_feature(
            team_b_players_stats, team_b_vs_others_stats
        )

        with torch.no_grad():
            team_a_pr_encoded = pr_vector(
                torch.tensor(
                    np.array(team_a_pr_feature, dtype=np.float32).reshape(1, -1)
                )
            )
            team_b_pr_encoded = pr_vector(
                torch.tensor(
                    np.array(team_b_pr_feature, dtype=np.float32).reshape(1, -1)
                )
            )

        team_a_pr_vector = team_a_pr_encoded.detach().cpu().numpy().ravel()
        team_b_pr_vector = team_b_pr_encoded.detach().cpu().numpy().ravel()

        team_a_feature = np.array(create_team_feature(team_a_vs_b_stats))
        team_b_feature = np.array(create_team_feature(team_b_vs_a_stats))
        team_a_features.append(
            np.concatenate([team_a_pr_vector, team_a_feature], axis=0)
        )
        team_b_features.append(
            np.concatenate([team_b_pr_vector, team_b_feature], axis=0)
        )

        win_probabilities.append(team_a_vs_b_stats["Map Win Pct"].values[0])

    team_a_tensor = torch.from_numpy(
        np.stack(team_a_features, axis=0).astype(np.float32)
    )
    team_b_tensor = torch.from_numpy(
        np.stack(team_b_features, axis=0).astype(np.float32)
    )
    win_probabilities_tensor = torch.tensor(
        win_probabilities, dtype=torch.float32
    ).unsqueeze(1)

    return team_a_tensor, team_b_tensor, win_probabilities_tensor


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
            (matchup_data["Teams"] == "A") & (matchup_data["Opponent"] == "Others")
        ]
        team_b_vs_others_stats = matchup_data[
            (matchup_data["Teams"] == "B") & (matchup_data["Opponent"] == "Others")
        ]

        team_a_pr_features.append(
            create_team_pr_feature(team_a_players_stats, team_a_vs_others_stats)
        )
        team_b_pr_features.append(
            create_team_pr_feature(team_b_players_stats, team_b_vs_others_stats)
        )

    team_a_pr_tensor = torch.tensor(team_a_pr_features, dtype=torch.float32)
    team_b_pr_tensor = torch.tensor(team_b_pr_features, dtype=torch.float32)
    team_pr_tensor = torch.cat([team_a_pr_tensor, team_b_pr_tensor], dim=0)

    return team_pr_tensor, feature_names


def get_power_rating_model(
    transformed_data: DATAFRAME_BY_YEAR_TYPE,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], StandardScaler]:
    scaler = StandardScaler()
    yearly_team_pr_models: list[PowerRatingNeuralNetwork] = []
    all_team_features: list[np.ndarray] = []
    year_cache = {}

    for year, data in transformed_data.items():
        players_stats = data["players_stats"]["team_players_stats"]
        matchups_stats = data["matches"]["teams_matchups_stats"]

        pr_tensor, feature_names = create_pr_input_tensor(players_stats, matchups_stats)
        mask = ~torch.isnan(pr_tensor).any(1)
        pr_tensor = pr_tensor[mask]

        year_cache[year] = dict(raw=pr_tensor, names=feature_names)
        all_team_features.append(pr_tensor.numpy())

    scaler.fit(np.vstack(all_team_features))

    for year, cached in year_cache.items():
        print("Training PR NN for year", year)
        raw_tensor = cached["raw"]
        names = cached["names"]

        X_scaled = scaler.transform(raw_tensor.numpy())
        X_scaled = torch.tensor(X_scaled, dtype=torch.float32)

        net = PowerRatingNeuralNetwork(input_size=len(names))
        net.train_model(X_scaled)
        yearly_team_pr_models.append(net)

    pr_model = ScaledPRModel(yearly_team_pr_models, scaler)
    return pr_model


def create_final_pr_model(
    models: list[PowerRatingNeuralNetwork],
) -> Callable[[torch.Tensor], torch.Tensor]:
    def final_predication(team_features: torch.Tensor) -> torch.Tensor:
        for m in models:
            m.eval()
        with torch.no_grad():
            preds = torch.stack([m.encode(team_features) for m in models], dim=0)
            return preds

    return final_predication


def create_final_match_model(
    models: list[MatchPredictorNeuralNetwork],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    def final_prediction(
        team_a_features: torch.Tensor, team_b_features: torch.Tensor
    ) -> torch.Tensor:
        for model in models:
            model.eval()

        with torch.no_grad():
            logits = torch.stack(
                [model(team_a_features, team_b_features) for model in models], dim=0
            )
            probs = torch.sigmoid(logits)
            return torch.mean(probs, dim=0)

    return final_prediction


def get_match_predictor_model(
    pr_model: Callable[[torch.Tensor], torch.Tensor],
    transformed_data: DATAFRAME_BY_YEAR_TYPE,
) -> Callable[[torch.Tensor], torch.Tensor]:
    yearly_match_models = []
    for year, year_data in transformed_data.items():
        players_stats = year_data["players_stats"]["team_players_stats"]
        matchups_data = year_data["matches"]["teams_matchups_stats"]

        team_a_tensor, team_b_tensor, win_probabilities = create_match_input_tensors(
            pr_model, players_stats, matchups_data
        )

        combined_mask = ~(torch.isnan(team_a_tensor).any(dim=1)) & ~(
            torch.isnan(team_b_tensor).any(dim=1)
        )
        team_a_tensor = team_a_tensor[combined_mask]
        team_b_tensor = team_b_tensor[combined_mask]
        win_probabilities = win_probabilities[combined_mask]

        input_size = len(team_a_tensor[0]) + len(team_b_tensor[0])
        match_predictor_nn = MatchPredictorNeuralNetwork(input_size=input_size)
        match_predictor_nn.train_model(team_a_tensor, team_b_tensor, win_probabilities)

        yearly_match_models.append(match_predictor_nn)

    final_match_model = create_final_match_model(yearly_match_models)
    return final_match_model


def compute_odds(yearly_probabilities):
    odds = []
    for probabilities in yearly_probabilities:
        eps = 1e-6
        probs = np.clip(probabilities.squeeze(1).cpu().numpy(), eps, 1 - eps)
        odds_a = 1.0 / probs
        odds_b = 1.0 / (1.0 - probs)

        df = pd.DataFrame(
            {
                "Team A Odds": odds_a,
                "Team B Odds": odds_b,
                "Original Team A Win Probability": probs,
            }
        )

        pd.set_option("display.float_format", "{:.2f}".format)
        odds.append(df)

    return odds


def match_decimal_odds(
    team_a: str, team_b: str, prob_a: float, eps: float = 1e-6
) -> tuple[float, float]:
    prob_a = max(min(prob_a, 1 - eps), eps)

    odd_a = round(1.0 / prob_a, 2)
    odd_b = round(1.0 / (1.0 - prob_a), 2)
    return {team_a: odd_a, team_b: odd_b}


def print_tables(
    header: str,
    y_true: np.ndarray,  # 0/1 outcomes (Team-A win)
    p_model: np.ndarray,  # model P(Team-A)
    p_book: np.ndarray,  # sportsbook P(Team-A)  (= 1/decimal_odds_A)
    vig_book_odds: np.ndarray,  # sportsbook *juiced* decimal odds for Team-A
    profit_model_raw: np.ndarray,  # $ profit per bet if book pays *fair* odds
    profit_model_vig: np.ndarray,  # $ profit per bet if book pays *juiced* odds
):
    print(header)
    ll_model = log_loss(y_true, p_model, labels=[0, 1])
    ll_book = log_loss(y_true, p_book, labels=[0, 1])
    br_model = brier_score_loss(y_true, p_model)
    br_book = brier_score_loss(y_true, p_book)

    obs_m, pred_m = calibration_curve(y_true, p_model, n_bins=10)
    slope_model = np.polyfit(pred_m, obs_m, 1)[0]
    obs_b, pred_b = calibration_curve(y_true, p_book, n_bins=10)
    slope_book = np.polyfit(pred_b, obs_b, 1)[0]

    profit_book_raw = np.where(y_true, 1 / p_book - 1, -1)
    profit_book_vig = np.where(y_true, vig_book_odds - 1, -1)

    roi_book_raw = 100 * profit_book_raw.mean()
    roi_book_vig = 100 * profit_book_vig.mean()
    roi_model_raw = 100 * profit_model_raw.mean()
    roi_model_vig = 100 * profit_model_vig.mean()

    ev_book = profit_book_vig.mean()
    ev_model = profit_model_vig.mean()

    df = pd.DataFrame(
        {
            "Metric": [
                "Log-loss",
                "Brier",
                "Calibration slope",
                "ROI (8 % vig)",
                "ROI (0 % vig)",
                "EV / bet",
            ],
            "Sportsbook": [
                ll_book,
                br_book,
                slope_book,
                roi_book_vig,
                roi_book_raw,
                ev_book,
            ],
            "NN Model": [
                ll_model,
                br_model,
                slope_model,
                roi_model_vig,
                roi_model_raw,
                ev_model,
            ],
        }
    )
    df["Δ"] = df["NN Model"] - df["Sportsbook"]
    print(df.to_markdown(index=False))


def compute_payouts_for_match(
    matchup_url: str,
    thunderbird_odds: dict[str, dict[str, float]],
    model_odds: dict[str, float],
    winner: str,
    vig: float = 0.0,  # e.g. 0.08 for 8% vig
) -> dict[str, dict[str, float]]:
    raw = thunderbird_odds[matchup_url]
    implied = {team: 1.0 / odds for team, odds in raw.items()}

    if vig > 0:
        vigged_prob = {team: prob * (1 + vig) for team, prob in implied.items()}
        actual_odds = {team: 1.0 / vigged_prob[team] for team in raw}
    else:
        actual_odds = raw

    th_payouts = {team: (actual_odds[team] if team == winner else -1.0) for team in raw}
    model_payouts = {
        team: (model_odds[team] if team == winner else -1.0) for team in model_odds
    }

    return {
        "Thunderbird": th_payouts,
        "Model": model_payouts,
    }


def train(
    years: list[str],
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]
]:
    dataframes_by_year = read_in_data("data", years)
    transformed_data = transform_data(dataframes_by_year)
    pr_model = get_power_rating_model(transformed_data)
    match_predictor_model = get_match_predictor_model(pr_model, transformed_data)

    return pr_model, match_predictor_model


def test(
    pr_model,
    match_model,
    thunderbird_match_odds: dict[str, dict[str, float]],
    vig: float = 0.08,
):
    players_stats = load_scraped_teams_players_stats_from_csv()
    matchups_stats = load_scraped_teams_matchups_stats_from_csv()

    team_a_t, team_b_t, _ = create_match_input_tensors(
        pr_model, players_stats, matchups_stats
    )
    probs = match_model(team_a_t, team_b_t).squeeze(1).cpu().numpy()

    # full sample metrics
    fs_y, fs_pm, fs_pb, fs_vig = [], [], [], []

    # advantaged-bet profits
    adv_y, adv_pm, adv_pb, adv_vig = [], [], [], []
    profit_model_raw, profit_model_vig = [], []

    bankroll_raw = bankroll_vig = 0.0
    bets = 0
    ev_vals = []

    urls = matchups_stats["Matchup URL"].unique()
    for url, pA in zip(urls, probs):
        row = matchups_stats[matchups_stats["Matchup URL"] == url].iloc[0]
        team_a, team_b = row["Matchup"].split("_vs_")
        winner = players_stats.loc[
            players_stats["Matchup URL"] == url, "Won Match"
        ].iat[0]
        yA = 1 if winner == team_a else 0

        # book odds
        raw_odds = thunderbird_match_odds[url]
        pA_book = 1.0 / raw_odds[team_a]

        # juiced odds
        vig_odds = {t: od / (1 + vig) for t, od in raw_odds.items()}

        # add to full-sample lists
        fs_y.append(yA)
        fs_pm.append(float(pA))
        fs_pb.append(pA_book)
        fs_vig.append(vig_odds[team_a])

        # EV for each side (with juiced odds)
        evA = pA * (vig_odds[team_a] - 1) - (1 - pA)
        evB = (1 - pA) * (vig_odds[team_b] - 1) - pA

        # place a bet only if positive EV
        if evA > evB and evA > 0:
            pick, ev = team_a, evA
            p_pick = pA
        elif evB > evA and evB > 0:
            pick, ev = team_b, evB
            p_pick = 1 - pA
        else:
            continue

        bets += 1
        ev_vals.append(ev)
        win = pick == winner

        # profits from the sportsbook prices
        profit_raw = (raw_odds[pick] - 1) if win else -1
        profit_vig = (vig_odds[pick] - 1) if win else -1

        bankroll_raw += profit_raw
        bankroll_vig += profit_vig

        # store per-bet arrays for metrics
        adv_y.append(int(win))
        adv_pm.append(p_pick)
        adv_pb.append(1 / raw_odds[pick])
        adv_vig.append(vig_odds[pick])
        model_odds = 1 / p_pick  # p_pick is model prob on the side we bet
        profit_model_raw.append(model_odds - 1 if win else -1)
        profit_model_vig.append(model_odds - 1 if win else -1)  # same (no extra vig)

    # headline back-test numbers
    avg_ev = np.mean(ev_vals) if bets else 0
    roi_raw = 100 * bankroll_raw / bets if bets else 0
    roi_vig = 100 * bankroll_vig / bets if bets else 0
    edge = 100 * (bankroll_raw - bankroll_vig) / bets if bets else 0

    print("\n--- ADVANTAGED BETS Backtest ---")
    print(f"Bets placed:                      {bets}")
    print(f"Average model EV:                 {avg_ev:+.3f}")
    print(
        f"Site pays RAW odds (vig 0%):      ${bankroll_raw:+.2f} | ROI {roi_raw:+.1f}%"
    )
    print(
        f"Site pays JUICED odds (vig {vig*100:.0f}%):   "
        f"${bankroll_vig:+.2f} | ROI {roi_vig:+.1f}%"
    )
    print(f"Edge lost to vig:                 {edge:+.1f}% of stakes")

    # advantage slice
    print_tables(
        "Advantaged-Bets Scoring",
        np.array(adv_y, dtype=int),
        np.array(adv_pm),
        np.array(adv_pb),
        np.array(adv_vig),
        np.array(profit_model_raw),
        np.array(profit_model_vig),
    )

    # full-sample
    print_tables(
        "\n--- FULL-SAMPLE Proper-Scoring Metrics ---",
        np.array(fs_y, dtype=int),
        np.array(fs_pm),
        np.array(fs_pb),
        np.array(fs_vig),
        profit_model_raw=np.zeros_like(fs_y, dtype=float),
        profit_model_vig=np.zeros_like(fs_y, dtype=float),
    )


if __name__ == "__main__":
    set_display_options()
    pr_model, match_model = train(years=["2022", "2023"])
    thunderbird_match_odds = load_year_thunderbird_match_odds_from_csv("2024")
    test(pr_model, match_model, thunderbird_match_odds)
