import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class abc(nn.Module):
    def __init__(self, input_size=10) -> None:
        super().__init__()
        self.team_layer1 = nn.Linear(input_size, 64)
        self.team_layer2 = nn.Linear(64, 32)

        self.combined_layer1 = nn.Linear(64, 32)
        self.combined_layer2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1)

        self.dropout = nn.Dropout(0.2)
        self.criterion = nn.BCELoss()

    def process_team(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.team_layer1(x))
        x = self.dropout(x)
        x = F.relu(self.team_layer2(x))
        return x

    def forward(self, team_a: torch.Tensor, team_b: torch.Tensor) -> torch.Tensor:
        team_a_features = self.process_team(team_a)
        team_b_features = self.process_team(team_b)
        combined_features = torch.cat((team_a_features, team_b_features), dim=1)

        x = F.relu(self.combined_layer1(combined_features))
        x = self.dropout(x)
        x = F.relu(self.combined_layer2(x))

        win_prob = torch.sigmoid(self.output_layer(x))
        return win_prob

    def train_model(
        self,
        team_a_tensor: torch.Tensor,
        team_b_tensor: torch.Tensor,
        win_probabilities: torch.Tensor,
        epochs: int = 1000,
        learning_rate: float = 0.001,
        batch_size: int = 16,
    ) -> None:
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        dataset = TensorDataset(team_a_tensor, team_b_tensor, win_probabilities)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        for i in range(epochs):
            total_loss = 0
            for team_a_batch, team_b_batch, target_batch in dataloader:
                self.optimizer.zero_grad()

                outputs = self(team_a_batch, team_b_batch)
                loss = self.criterion(outputs, target_batch)
                loss.backward()

                self.optimizer.step()
                total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch {i+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

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
        latent_dim: int = 1,
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
        epochs: int = 200,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
        print_every: int = 20,
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
        """
        Compute the unsupervised power rating for a new team.
        Args:
            feature_tensor: Tensor of shape (1, input_size).
        Returns:
            Scalar latent code.
        """
        self.eval()
        with torch.no_grad():
            z = self.encode(feature_tensor)
        return z.item()


class TeamPredictorNeuralNetwork(nn.Module):
    def __init__(self):
        pass
