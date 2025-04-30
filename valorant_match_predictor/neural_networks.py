import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class MatchPredictorNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_sizes: list[int] = [64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_sizes[0])
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.output = nn.Linear(hidden_sizes[1], 1)
        self.dropout = nn.Dropout(dropout)
        self.criterion = nn.BCELoss()

    def forward(
        self, pr_features: torch.Tensor, other_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        pr_features: (batch_size, pr_feat_dim)
        other_feats:{batch_size, other_feat_dim}
        """
        x = torch.cat([pr_features, other_feats], dim=1)
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        return torch.sigmoid(self.output(x))

    def train_model(
        self,
        team_a_features: torch.Tensor,
        team_b_features: torch.Tensor,
        win_labels: torch.Tensor,
        epochs: int = 1000,
        learning_rate: float = 0.001,
        batch_size: int = 16,
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        ds = TensorDataset(team_a_features, team_b_features, win_labels)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        self.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for prb, ofb, yb in dl:
                optimizer.zero_grad()
                preds = self(prb, ofb)
                loss = self.criterion(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                avg = total_loss / len(dl)
                print(f"Epoch {epoch}/{epochs}, Loss: {avg:.4f}")
        self.eval()

    def predict(
        self, team_a_features: torch.Tensor, team_b_features: torch.Tensor
    ) -> float:
        self.eval()
        with torch.no_grad():
            prediction = self(team_a_features, team_b_features)
            return prediction.item()


class PowerRatingNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        latent_dim: int = 8,
        hidden_dims: list[int] = [64, 32],
        dropout: float = 0.1,
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

        # reconstruction loss
        self.criterion = nn.MSELoss()

        # Xavier‐initialize all weights
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
        return self.encoder(x)

    def train_model(
        self,
        feature_tensor: torch.Tensor,
        epochs: int = 1000,
        learning_rate: float = 0.001,
        batch_size: int = 16,
        print_every: int = 10,
    ) -> None:
        # print(feature_tensor)
        assert not torch.isnan(feature_tensor).any(), "Input contains NaN!"
        assert not torch.isinf(feature_tensor).any(), "Input contains Inf!"

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        dataloader = DataLoader(
            TensorDataset(feature_tensor), batch_size=batch_size, shuffle=True
        )

        self.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for (X_batch,) in dataloader:
                optimizer.zero_grad()
                X_recon = self(X_batch)
                loss = self.criterion(X_recon, X_batch)
                loss.backward()

                # clip gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            if epoch % print_every == 0:
                avg = total_loss / len(dataloader)
                print(f"Epoch {epoch}/{epochs} — Recon Loss: {avg:.4f}")

    def predict(self, feature_tensor: torch.Tensor) -> float:
        self.eval()
        with torch.no_grad():
            z = self.encode(feature_tensor)
        return z.item()
