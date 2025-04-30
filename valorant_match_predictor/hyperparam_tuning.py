"""
Optuna multi-objective tuning for MatchPredictorNeuralNetwork
   – Objective-1:  minimise log-loss
   – Objective-2:  minimise |calibration-slope − 1|
Extra metrics (Brier, raw slope) are stored and printed.
"""
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

# ──────────────────────────────────────────────────────────────
# 1. Load + transform data once (outside the Optuna objective)
# ──────────────────────────────────────────────────────────────
YEARS = ["2022", "2023"]
dfs_by_year = read_in_data("data", YEARS)
transformed = transform_data(dfs_by_year)

# get_power_rating_model now returns 3 things:  pr_model, scaler_pr, scaler_h2h
pr_model, scaler_pr, scaler_h2h = get_power_rating_model(transformed)

team_a_all, team_b_all, y_all = [], [], []
for yr in YEARS:
    data = transformed[yr]
    ta, tb, y_frac = create_match_input_tensors(
        pr_model,
        data["players_stats"]["team_players_stats"],
        data["matches"]["teams_matchups_stats"],
        scaler_pr,
        scaler_h2h,  # ← NEW
    )
    y_bin = (y_frac >= 0.5).float()
    mask = ~(torch.isnan(ta).any(1) | torch.isnan(tb).any(1) | y_bin.squeeze().isnan())
    team_a_all.append(ta[mask])
    team_b_all.append(tb[mask])
    y_all.append(y_bin[mask])

team_a_all = torch.cat(team_a_all)
team_b_all = torch.cat(team_b_all)
y_all = torch.cat(y_all)
print(f"Prepared {len(team_a_all)} samples from {YEARS}")

# ──────────────────────────────────────────────────────────────
# 2. Helpers
# ──────────────────────────────────────────────────────────────
warnings.simplefilter("ignore", RankWarning)  # silence polyfit warnings
BIG_PENALTY = 9_999.0  # huge number to penalise bad slope


def safe_slope(obs: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """return (slope, |slope-1|); protect against ill-conditioned polyfit"""
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


# ──────────────────────────────────────────────────────────────
# 3. Optuna objective
# ──────────────────────────────────────────────────────────────
def objective(trial: optuna.trial.Trial) -> tuple[float, float]:
    # hyper-parameters to sample
    h1 = trial.suggest_int("hidden1", 64, 512, step=64)
    h2 = trial.suggest_int("hidden2", 16, h1, step=16)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    bs = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = 50

    # dataloaders
    ds = TensorDataset(team_a_all, team_b_all, y_all)
    n_val = int(len(ds) * 0.2)
    tr_ds, va_ds = random_split(ds, [len(ds) - n_val, n_val])
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=bs)

    # model + optimiser
    model = MatchPredictorNeuralNetwork(
        input_size=team_a_all.shape[1] + team_b_all.shape[1],
        hidden_sizes=[h1, h2],
        dropout=dropout,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # training loop
    model.train()
    for _ in range(epochs):
        for a, b, y in tr_dl:
            opt.zero_grad()
            p = model(a, b)
            F.binary_cross_entropy(p, y).backward()
            opt.step()

    # validation
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for a, b, y in va_dl:
            preds.append(model(a, b).cpu().numpy().ravel())
            truths.append(y.cpu().numpy().ravel())
    preds = np.clip(np.concatenate(preds), 1e-6, 1 - 1e-6)
    truths = np.concatenate(truths)

    ll = log_loss(truths, preds)
    brier = brier_score_loss(truths, preds)
    obs, pred = calibration_curve(truths, preds, n_bins=10)
    slope, slope_err = safe_slope(obs, pred)

    trial.set_user_attr("brier", brier)  # store extra metric
    return ll, slope_err


# ──────────────────────────────────────────────────────────────
# 4. Run study
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        study_name="match-NN-tuning",
    )
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # results
    print("\nPareto-optimal trials")
    for t in study.best_trials:
        pprint_trial("★", t)

    print("\nTop-5 by calibration (may include dominated trials)")
    top5 = sorted(study.trials, key=lambda t: (t.values[1], t.values[0]))[:5]
    for i, t in enumerate(top5, 1):
        pprint_trial(str(i), t)
