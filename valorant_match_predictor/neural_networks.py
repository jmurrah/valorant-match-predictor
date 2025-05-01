from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


class MatchPredictorNeuralNetwork(nn.Module):
    """
    Concatenate (Team-A ‖ Team-B) -> logit.
    We train on continuous labels with BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = (32, 16),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.out = nn.Linear(hidden_sizes[1], 1)
        self.drop = nn.Dropout(dropout)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, a_feats: torch.Tensor, b_feats: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a_feats, b_feats], dim=1)
        x = self.drop(F.relu(self.l1(x)))
        x = self.drop(F.relu(self.l2(x)))
        return self.out(x)  # raw log-odds (logit)

    def train_model(
        self,
        a_feats: torch.Tensor,
        b_feats: torch.Tensor,
        labels: torch.Tensor,  # continuous
        *,
        epochs: int = 500,
        learning_rate: float = 0.01,
        batch_size: int = 16,
        patience: int = 50,
        weight_decay: float = 2.0e-6,
    ) -> None:
        loader = DataLoader(
            TensorDataset(a_feats, b_feats, labels), batch_size=batch_size, shuffle=True
        )
        opt = optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        best, no_imp = float("inf"), 0
        self.train()
        for ep in range(1, epochs + 1):
            run = 0.0
            for a_b, b_b, y_b in loader:
                opt.zero_grad()
                loss = self.criterion(self(a_b, b_b).squeeze(1), y_b.squeeze(1))
                loss.backward()
                opt.step()
                run += loss.item()

            avg = run / len(loader)
            if avg < best:
                best, no_imp = avg, 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    print(f"Early stopping @ {ep}")
                    break
            if ep % 10 == 0:
                print(f"Epoch {ep:3d}  loss={avg:.4f}")

        self.eval()

    def predict_proba(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self(a, b))


class PowerRatingNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        latent_dim: int = 8,
        hidden_dims: list[int] = [32, 16],
        dropout: float = 0.3,
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def train_model(
        self,
        feature_tensor: torch.Tensor,
        epochs: int = 500,
        learning_rate: float = 0.01,
        batch_size: int = 16,
        patience: int = 50,
    ) -> None:
        dataloader = DataLoader(
            TensorDataset(feature_tensor), batch_size=batch_size, shuffle=True
        )
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        best_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        self.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for (X_batch,) in dataloader:
                optimizer.zero_grad()
                X_recon = self(X_batch)
                loss = self.criterion(X_recon, X_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
                best_state = self.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(no recon improvement for {patience} epochs)"
                    )
                    break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} — Recon Loss: {avg_loss:.4f}")

        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()

    def predict(self, feature_tensor: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            z = self.encode(feature_tensor)
        return z.item()


class ScaledPRModel:
    def __init__(
        self, models: list[PowerRatingNeuralNetwork], scaler: StandardScaler
    ) -> None:
        self.models = models
        self.scaler = scaler
        for m in self.models:
            m.eval()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (n_samples, n_raw_features) **un-scaled**
        returns: (ensemble_mean, ...) same shape as each model output
        """
        # numpy -> transform -> torch
        x_scaled = self.scaler.transform(x.numpy())
        x_scaled = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            preds = torch.stack([m.encode(x_scaled) for m in self.models], 0)
            return preds.mean(0)
