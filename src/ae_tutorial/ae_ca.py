from torch import nn, Tensor


class AEEncoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, hidden_size, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(hidden_size, out_features, kernel_size=(3, 3))
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(hidden_size)
        self.batch_norm2 = nn.BatchNorm2d(hidden_size)

    def forward(self, x: Tensor):
        h = self.conv1(x)
        h = self.batch_norm1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.batch_norm2(h)
        h = self.relu(h)
        h = self.conv3(h)
        return h


class AEDecoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(out_features, hidden_size, kernel_size=(3, 3))
        self.conv2 = nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=(3, 3))
        self.conv3 = nn.ConvTranspose2d(hidden_size, in_features, kernel_size=(3, 3))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.batch_norm1 = nn.BatchNorm2d(hidden_size)
        self.batch_norm2 = nn.BatchNorm2d(hidden_size)

    def forward(self, z: Tensor):
        h = self.conv1(z)
        h = self.batch_norm1(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.batch_norm2(h)
        h = self.relu(h)
        h = self.conv3(h)
        output = self.tanh(h)
        return output
