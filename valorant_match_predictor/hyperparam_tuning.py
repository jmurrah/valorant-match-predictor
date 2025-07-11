from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from numpy.polynomial.polyutils import RankWarning
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from torch.utils.data import DataLoader, TensorDataset, random_split

from valorant_match_predictor.main import (
    create_match_input_tensors,
    get_power_rating_model,
    read_in_data,
    transform_data,
)
from valorant_match_predictor.neural_networks import MatchPredictorNeuralNetwork

warnings.simplefilter("ignore", RankWarning)
BIG_PENALTY = 9_999.0

YEARS = ["2022", "2023"]
dfs_by_year = read_in_data("data", YEARS)
transformed = transform_data(dfs_by_year)
pr_model, scaler_pr, scaler_h2h = get_power_rating_model(transformed)

a_all, b_all, y_cont_all, y_bin_all = [], [], [], []
eps = 1e-3
for yr in YEARS:
    data = transformed[yr]
    ta, tb, y_frac = create_match_input_tensors(
        pr_model,
        data["players_stats"]["team_players_stats"],
        data["matches"]["teams_matchups_stats"],
        scaler_pr,
        scaler_h2h,
    )

    y_cont = y_frac.clamp(eps, 1 - eps)
    y_bin = (y_frac >= 0.5).float()

    mask = ~(
        torch.isnan(ta).any(1) | torch.isnan(tb).any(1) | torch.isnan(y_cont.squeeze())
    )
    a_all.append(ta[mask])
    b_all.append(tb[mask])
    y_cont_all.append(y_cont[mask])
    y_bin_all.append(y_bin[mask])

team_a_all = torch.cat(a_all)
team_b_all = torch.cat(b_all)
y_cont_all = torch.cat(y_cont_all)
y_bin_all = torch.cat(y_bin_all)

print(f"Prepared {len(team_a_all)} samples from {YEARS}")


def safe_slope(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    try:
        slope = np.polyfit(pred, obs, 1)[0]
        return slope, abs(slope - 1.0)
    except (TypeError, np.linalg.LinAlgError):
        return 1.0, BIG_PENALTY


def pprint_trial(tag: str, t: optuna.trial.FrozenTrial) -> None:
    ll, slope_err = t.values
    print(
        f"{tag}  log-loss={ll:0.6f}  |slope−1|={slope_err:0.6f}  "
        f"brier={t.user_attrs['brier']:0.6f}  params={t.params}"
    )


def objective(trial: optuna.trial.Trial) -> tuple[float, float]:
    # hyper-parameters
    h1 = trial.suggest_int("hidden1", 64, 512, step=64)
    h2 = trial.suggest_int("hidden2", 16, h1, step=16)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    bs = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = 50

    # Dataset contains BOTH labels
    ds = TensorDataset(team_a_all, team_b_all, y_cont_all, y_bin_all)
    n_val = int(len(ds) * 0.2)
    tr_ds, va_ds = random_split(ds, [len(ds) - n_val, n_val])
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=bs)

    # model
    model = MatchPredictorNeuralNetwork(
        input_size=team_a_all.shape[1] + team_b_all.shape[1],
        hidden_sizes=[h1, h2],
        dropout=dropout,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # training – logits + BCEWithLogitsLoss, continuous targets
    model.train()
    for _ in range(epochs):
        for a, b, y_cont, _ in tr_dl:
            opt.zero_grad()
            logits = model(a, b).squeeze(1)
            F.binary_cross_entropy_with_logits(logits, y_cont.squeeze(1)).backward()
            opt.step()

    # validation
    model.eval()
    prob_preds, bin_truths, cont_truths = [], [], []
    with torch.no_grad():
        for a, b, y_cont, y_bin in va_dl:
            prob = torch.sigmoid(model(a, b)).cpu().numpy().ravel()
            prob_preds.append(prob)
            bin_truths.append(y_bin.cpu().numpy().ravel())
            cont_truths.append(y_cont.cpu().numpy().ravel())

    preds = np.clip(np.concatenate(prob_preds), 1e-6, 1 - 1e-6)
    y_bin = np.concatenate(bin_truths)  # 0/1 for log-loss / slope
    y_cont = np.concatenate(cont_truths)  # for Brier

    ll = log_loss(y_bin, preds, labels=[0, 1])
    brier = brier_score_loss(y_bin, preds)
    obs, pred = calibration_curve(y_bin, preds, n_bins=10)
    _, slope_err = safe_slope(obs, pred)

    trial.set_user_attr("brier", brier)
    return ll, slope_err


if __name__ == "__main__":
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name="match-NN-tuning",
    )
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    print("\nPareto-optimal trials")
    for t in study.best_trials:
        pprint_trial("★", t)

    print("\nTop-5 by calibration (may include dominated trials)")
    top5 = sorted(study.trials, key=lambda t: (t.values[1], t.values[0]))[:5]
    for i, t in enumerate(top5, 1):
        pprint_trial(str(i), t)
