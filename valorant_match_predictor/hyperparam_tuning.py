import optuna
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

from valorant_match_predictor.main import (
    read_in_data,
    transform_data,
    create_match_input_tensors,
    get_power_rating_model,
)
from valorant_match_predictor.neural_networks import MatchPredictorNeuralNetwork

# 1) PREPARE data for 2022+2023
YEARS = ["2022", "2023"]
dfs_by_year = read_in_data("data", YEARS)
transformed = transform_data(dfs_by_year)
pr_model = get_power_rating_model(transformed)

team_a_list, team_b_list, y_list = [], [], []
for year in YEARS:
    data = transformed[year]
    ta, tb, y_frac = create_match_input_tensors(
        pr_model,
        data["players_stats"]["team_players_stats"],
        data["matches"]["teams_matchups_stats"],
    )
    y_true = (y_frac >= 0.5).float()
    mask = ~(
        torch.isnan(ta).any(1) | torch.isnan(tb).any(1) | torch.isnan(y_true).squeeze()
    )
    team_a_list.append(ta[mask])
    team_b_list.append(tb[mask])
    y_list.append(y_true[mask])

team_a_all = torch.cat(team_a_list, dim=0)
team_b_all = torch.cat(team_b_list, dim=0)
y_all = torch.cat(y_list, dim=0)
print(f"Prepared {len(team_a_all)} samples from {YEARS}")


def objective(trial):
    # sample hyperparams
    h1 = trial.suggest_int("hidden1", 64, 512, step=64)
    h2 = trial.suggest_int("hidden2", 16, h1, step=16)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    bs = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = 50

    ds = TensorDataset(team_a_all, team_b_all, y_all)
    n_val = int(len(ds) * 0.2)
    n_tr = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=bs)

    model = MatchPredictorNeuralNetwork(
        input_size=team_a_all.shape[1] + team_b_all.shape[1],
        hidden_sizes=[h1, h2],
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    model.train()
    for _ in range(epochs):
        for a, b, y in tr_dl:
            optimizer.zero_grad()
            p = model(a, b)
            loss = F.binary_cross_entropy(p, y)
            loss.backward()
            optimizer.step()

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
    bs = brier_score_loss(truths, preds)
    obs, pred = calibration_curve(truths, preds, n_bins=10)
    slope = np.polyfit(pred, obs, 1)[0]
    slope_err = abs(slope - 1.0)

    trial.set_user_attr("slope", slope)
    trial.set_user_attr("brier", bs)
    return ll, bs, slope_err


if __name__ == "__main__":
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"])
    study.optimize(objective, n_trials=100)

    # Print full Pareto front
    print("\nPareto‐optimal trials:")
    for t in study.best_trials:
        ll_val, bs_val, slope_err = t.values
        print(f" params: {t.params}")
        print(
            f"   log-loss={ll_val:.4f}, brier={bs_val:.4f}, |slope−1|={slope_err:.4f}"
        )

    # Sort by calibration‐error, then log‐loss, then brier
    sorted_trials = sorted(
        study.best_trials, key=lambda t: (t.values[2], t.values[0], t.values[1])
    )

    # Print top 3
    print("\nTop 3 trials by calibration:")
    for t in sorted_trials[:3]:
        ll_val, bs_val, slope_err = t.values
        slope = t.user_attrs["slope"]
        print(f" params: {t.params}")
        print(
            f"   slope={slope:.4f}  |slope−1|={slope_err:.4f}, "
            f"log-loss={ll_val:.4f}, brier={bs_val:.4f}"
        )
