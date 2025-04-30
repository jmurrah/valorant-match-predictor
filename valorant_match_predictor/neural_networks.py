import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import KFold


class MatchPredictorNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = [512, 256],
        dropout: float = 0.2,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.output = nn.Linear(hidden_sizes[1], 1)
        self.dropout = nn.Dropout(dropout)
        # use logits + BCEWithLogitsLoss for numerical stability
        self.criterion = nn.BCEWithLogitsLoss()
        self.weight_decay = weight_decay

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, pr_features: torch.Tensor, other_feats: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([pr_features, other_feats], dim=1)
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        return self.output(x)  # logits

    def train_model(
        self,
        team_a_features: torch.Tensor,
        team_b_features: torch.Tensor,
        win_labels: torch.Tensor,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        patience: int = 50,
        val_frac: float = 0.2,
    ):
        """
        Trains with an internal train/val split and early stopping.
        If val_frac <= 0, trains on all data without internal validation.
        """
        # prepare dataset
        ds = TensorDataset(team_a_features, team_b_features, win_labels)
        n_val = int(len(ds) * val_frac)
        # split into train / val only if val_frac > 0 and there is at least one validation example
        if n_val > 0:
            n_train = len(ds) - n_val
            train_ds, val_ds = random_split(ds, [n_train, n_val])
            val_dl = DataLoader(val_ds, batch_size=batch_size)
        else:
            train_ds = ds
            val_dl = None

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        best_state = None
        best_val_loss = float("inf")
        epochs_no_improve = 0

        self.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for prb, ofb, yb in train_dl:
                optimizer.zero_grad()
                logits = self(prb, ofb)
                loss = self.criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_train_loss = total_loss / len(train_dl)

            # perform validation and early stopping if val_dl exists
            if val_dl is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for prb, ofb, yb in val_dl:
                        logits = self(prb, ofb)
                        val_loss += self.criterion(logits, yb).item()
                avg_val_loss = val_loss / len(val_dl)

                if epoch % 10 == 0 or epoch == epochs:
                    print(
                        f"Epoch {epoch}/{epochs} — "
                        f"Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}"
                    )

                # early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_state = self.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(
                            f"Early stopping at epoch {epoch} "
                            f"(no val loss improvement for {patience} epochs)"
                        )
                        break
                self.train()
            else:
                # no internal validation; just print training loss periodically
                if epoch % 10 == 0 or epoch == epochs:
                    print(f"Epoch {epoch}/{epochs} — Train Loss: {avg_train_loss:.4f}")

        # restore best weights if using validation
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()

    def predict(
        self, team_a_features: torch.Tensor, team_b_features: torch.Tensor
    ) -> float:
        self.eval()
        with torch.no_grad():
            logits = self(team_a_features, team_b_features)
            return torch.sigmoid(logits).item()


class PowerRatingNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        latent_dim: int = 8,
        hidden_dims: list[int] = [1024, 512],
        dropout: float = 0.1,
        weight_decay: float = 1e-5,
    ) -> None:
        super().__init__()
        # build encoder
        encoder_layers = []
        prev_dim = input_size
        for h in hidden_dims:
            encoder_layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_size))
        self.decoder = nn.Sequential(*decoder_layers)

        self.criterion = nn.MSELoss()
        self.weight_decay = weight_decay

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns latent representation z for batch x."""
        return self.encoder(x)

    def train_model(
        self,
        feature_tensor: torch.Tensor,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        print_every: int = 10,
        patience: int = 50,
        val_frac: float = 0.2,
    ) -> None:
        """
        Auto-encoder training with early stopping on a validation split.
        """
        ds = TensorDataset(feature_tensor)
        n_val = int(len(ds) * val_frac)
        if n_val > 0:
            n_train = len(ds) - n_val
            train_ds, val_ds = random_split(ds, [n_train, n_val])
            val_dl = DataLoader(val_ds, batch_size=batch_size)
        else:
            train_ds = ds
            val_dl = None
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        best_state = None
        best_val_loss = float("inf")
        epochs_no_improve = 0

        self.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for (X_batch,) in train_dl:
                optimizer.zero_grad()
                X_recon = self(X_batch)
                loss = self.criterion(X_recon, X_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            avg_train_loss = total_loss / len(train_dl)

            if val_dl is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for (X_batch,) in val_dl:
                        X_recon = self(X_batch)
                        val_loss += self.criterion(X_recon, X_batch).item()
                avg_val_loss = val_loss / len(val_dl)

                if epoch % print_every == 0 or epoch == epochs:
                    print(
                        f"Epoch {epoch}/{epochs} — "
                        f"Train Recon Loss: {avg_train_loss:.4f}  "
                        f"Val Recon Loss:   {avg_val_loss:.4f}"
                    )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    best_state = self.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(
                            f"Early stopping at epoch {epoch} "
                            f"(no val recon improvement for {patience} epochs)"
                        )
                        break
                self.train()
            else:
                if epoch % print_every == 0 or epoch == epochs:
                    print(
                        f"Epoch {epoch}/{epochs} — Train Recon Loss: {avg_train_loss:.4f}"
                    )

        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()

    def predict(self, feature_tensor: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            x = (
                feature_tensor.unsqueeze(0)
                if feature_tensor.dim() == 1
                else feature_tensor
            )
            z = self.encode(x)
        return z.squeeze(0).cpu().numpy()


def cross_validate_match_predictor(
    input_size: int,
    team_a_features: torch.Tensor,
    team_b_features: torch.Tensor,
    win_labels: torch.Tensor,
    k: int = 5,
    **train_kwargs,
) -> list[float]:
    """
    Perform k-fold CV, returning validation losses for each fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(team_a_features), 1):
        print(f"\n=== Fold {fold}/{k} ===")
        model = MatchPredictorNeuralNetwork(input_size, **train_kwargs)
        # split tensors
        ta_tr = team_a_features[train_idx]
        tb_tr = team_b_features[train_idx]
        y_tr = win_labels[train_idx]
        ta_va = team_a_features[val_idx]
        tb_va = team_b_features[val_idx]
        y_va = win_labels[val_idx]

        # train on TRAIN split, but we use val_frac=0 here to rely on the separate val set
        model.train_model(
            ta_tr,
            tb_tr,
            y_tr,
            # disable internal split since we're providing our own val set:
            val_frac=0.0,
            **train_kwargs,
        )

        # compute val loss
        model.eval()
        with torch.no_grad():
            logits = model(ta_va, tb_va)
            loss = model.criterion(logits, y_va).item()
        print(f"Fold {fold} Val Loss: {loss:.4f}")
        val_losses.append(loss)

    print(f"\nAverage CV Loss: {sum(val_losses)/len(val_losses):.4f}")
    return val_losses
