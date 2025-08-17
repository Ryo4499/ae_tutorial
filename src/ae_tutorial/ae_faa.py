from torch import nn, Tensor


class AEEncoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int):
        super().__init__()
        self.dense1 = nn.Linear(in_features, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, out_features)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x: Tensor):
        h = self.dense1(x)
        h = self.batch_norm1(h)
        h = self.relu(h)
        h = self.dense2(h)
        h = self.batch_norm2(h)
        h = self.relu(h)
        h = self.dense3(h)
        return h


class AEDecoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int):
        super().__init__()
        self.dense1 = nn.Linear(out_features, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, in_features)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, z: Tensor):
        h = self.dense1(z)
        h = self.batch_norm1(h)
        h = self.relu(h)
        h = self.dense2(h)
        h = self.batch_norm2(h)
        h = self.relu(h)
        h = self.dense3(h)
        output = self.tanh(h)
        return output
