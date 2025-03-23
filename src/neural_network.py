import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class NeuralNetwork(nn.Module):
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
