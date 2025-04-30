import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import bootstrap


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
                torch.tensor(np.array(team_a_pr_feature), dtype=torch.float32)
            )
            team_b_pr_encoded = pr_vector(
                torch.tensor(np.array(team_b_pr_feature), dtype=torch.float32)
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


def get_power_rating_model(
    transformed_data: DATAFRAME_BY_YEAR_TYPE,
) -> Callable[[torch.Tensor], torch.Tensor]:
    scaler = StandardScaler()
    yearly_team_pr_models = []
    all_team_features = []
    year_data_cache = {}

    for year, year_data in transformed_data.items():
        players_stats = year_data["players_stats"]["team_players_stats"]
        matchups_data = year_data["matches"]["teams_matchups_stats"]

        # print(matchups_data.head(10))
        # break
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

    final_team_pr_model = create_final_pr_model(yearly_team_pr_models)
    return final_team_pr_model


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


def print_tables(header, y_true, p_model, p_book, vig_book_odds):
    print(header)
    # Log-loss & Brier
    log_model = log_loss(y_true, p_model, labels=[0, 1])
    log_book = log_loss(y_true, p_book, labels=[0, 1])
    brier_m = brier_score_loss(y_true, p_model)
    brier_b = brier_score_loss(y_true, p_book)

    # Calibration slope ≈ OLS slope of y_true on p̂ buckets
    obs, pred = calibration_curve(y_true, p_model, n_bins=10)
    slope_m = np.polyfit(pred, obs, 1)[0]
    obs, pred = calibration_curve(y_true, p_book, n_bins=10)
    slope_b = np.polyfit(pred, obs, 1)[0]

    stakes = 1
    profit_raw = np.where(y_true, (1 / p_book) - 1, -1)
    profit_vig = np.where(y_true, vig_book_odds - 1, -1)
    roi_raw = 100 * profit_raw.mean()
    roi_vig = 100 * profit_vig.mean()

    excess = profit_vig - profit_vig.mean()
    sharpe_vig = profit_vig.mean() / excess.std(ddof=1)

    # Kelly fraction f* = (bp – q) / b  where b=odds-1, q=1-p
    b = vig_book_odds - 1
    kelly_f = (p_model * b - (1 - p_model)) / b
    # kelly_growth = np.mean(np.log1p(kelly_f * b * np.where(y_true, 1, -1)))

    better_prob = (p_model > p_book) == y_true
    fraction_winprob_beats_book = better_prob.mean()

    t_stat, p_val_log = ttest_rel(
        -np.log(np.where(y_true, p_model, 1 - p_model)),
        -np.log(np.where(y_true, p_book, 1 - p_book)),
    )
    ci = bootstrap(
        (profit_vig,), np.mean, confidence_level=0.95, method="bca"
    ).confidence_interval

    summary = pd.DataFrame(
        {
            "Metric": [
                "Log-loss",
                "Brier",
                "Calibration slope",
                "ROI (8% vig)",
                "ROI (0% vig)",
                "EV / bet",
            ],
            "Sportsbook": [
                log_book,
                brier_b,
                slope_b,
                roi_vig - roi_raw + roi_raw,
                0,
                0,
            ],
            "NN Model": [
                log_model,
                brier_m,
                slope_m,
                roi_vig,
                roi_raw,
                profit_vig.mean(),
            ],
        }
    )
    summary["Δ"] = summary["NN Model"] - summary["Sportsbook"]
    print(summary.to_markdown(index=False))


def compute_payouts_for_match(
    matchup_url: str,
    thunderbird_odds: dict[str, dict[str, float]],
    model_odds: dict[str, float],
    winner: str,
    vig: float = 0.0,  # e.g. 0.08 for 8% vig
) -> dict[str, dict[str, float]]:
    """
    Returns $1 returns for both the bookmaker ("Thunderbird") and your model,
    correctly applying an over-round (vig) to the book’s fair odds.
    """

    raw = thunderbird_odds[matchup_url]

    # 1) Convert raw fair odds → implied probabilities
    implied = {team: 1.0 / odds for team, odds in raw.items()}

    if vig > 0:
        # 2) Add the over-round
        vigged_prob = {team: prob * (1 + vig) for team, prob in implied.items()}
        # 3) Convert back to decimal odds
        actual_odds = {team: 1.0 / vigged_prob[team] for team in raw}
    else:
        # No vig: offered odds == raw fair odds
        actual_odds = raw

    # 4) Build payouts assuming a $1 stake
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


def test_advantaged(
    pr_model: Callable[[torch.Tensor], torch.Tensor],
    match_model: Callable[[torch.Tensor], torch.Tensor],
    thunderbird_match_odds: dict[str, dict[str, float]],
    vig: float = 0.08,
):
    players_stats = load_scraped_teams_players_stats_from_csv()
    matchups_stats = load_scraped_teams_matchups_stats_from_csv()
    team_a_t, team_b_t, _ = create_match_input_tensors(
        pr_model, players_stats, matchups_stats
    )
    probs = match_model(team_a_t, team_b_t).squeeze(1).cpu().numpy()

    bankroll_raw = 0.0
    bankroll_vig = 0.0
    bets = 0
    ev_vals = []
    y_true, p_model, p_book, p_book_vig = [], [], [], []
    vig_book_odds = []

    urls = matchups_stats["Matchup URL"].unique()
    for url, p in zip(urls, probs):
        team_a, team_b = (
            matchups_stats.loc[matchups_stats["Matchup URL"] == url, "Matchup"]
            .iat[0]
            .split("_vs_")
        )
        raw_odds = thunderbird_match_odds[url]

        # probabilitiy calcs
        p_model.append(float(p))
        p_a_book = 1.0 / raw_odds[team_a]
        p_book.append(p_a_book)
        p_a_book_vig = p_a_book * (1 + vig)

        if vig:
            probs_no_vig = {t: 1 / od for t, od in raw_odds.items()}
            probs_vig = {t: q * (1 + vig) for t, q in probs_no_vig.items()}
            vig_odds = {t: 1 / q for t, q in probs_vig.items()}
        else:
            vig_odds = raw_odds

        vig_book_odds.append(vig_odds[team_a])
        p_book_vig.append(p_a_book_vig)

        winner = players_stats.loc[
            players_stats["Matchup URL"] == url, "Won Match"
        ].iat[0]
        y_true.append(1 if winner == team_a else 0)

        ev_a = p * (vig_odds[team_a] - 1) - (1 - p)
        ev_b = (1 - p) * (vig_odds[team_b] - 1) - p

        if ev_a > ev_b and ev_a > 0:
            pick, ev_pick = team_a, ev_a
        elif ev_b > ev_a and ev_b > 0:
            pick, ev_pick = team_b, ev_b
        else:
            continue

        bets += 1
        ev_vals.append(ev_pick)

        if pick == winner:
            bankroll_raw += raw_odds[pick] - 1
            bankroll_vig += vig_odds[pick] - 1
        else:
            bankroll_raw -= 1
            bankroll_vig -= 1

    avg_ev = sum(ev_vals) / bets if bets else 0.0
    roi_raw = 100 * bankroll_raw / bets if bets else 0.0
    roi_vig = 100 * bankroll_vig / bets if bets else 0.0
    edge_loss = 100 * (bankroll_raw - bankroll_vig) / bets

    print("\n--- ADVANTAGED BETTING Results ---")
    print(f"Bets placed:                      {bets}")
    print(f"Average model EV:                 {avg_ev:+.3f}")
    print(
        f"Site pays RAW odds (vig 0%):      ${bankroll_raw:+.2f} | ROI {roi_raw:+.1f}%"
    )
    print(
        f"Site pays JUICED odds (vig {vig*100:.0f}%):   ${bankroll_vig:+.2f} | ROI {roi_vig:+.1f}%"
    )
    print(f"Edge lost to vig:                 {edge_loss:+.1f}% of stakes")
    print_tables(
        "ADVANTAGED BETTING Scoring Metrics",
        np.array(y_true, dtype=int),
        np.array(p_model),
        np.array(p_book),
        np.array(vig_book_odds),
    )
    print_tables(
        "Full-Sample Scoring Metrics",
        np.array(y_true, dtype=int),
        np.array(p_model),
        np.array(p_book),
        np.array(vig_book_odds),
    )


if __name__ == "__main__":
    set_display_options()
    pr_model, match_model = train(years=["2022", "2023"])
    thunderbird_match_odds = load_year_thunderbird_match_odds_from_csv("2024")
    test_advantaged(pr_model, match_model, thunderbird_match_odds)
