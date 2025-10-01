import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.uniform import Uniform


class MLP(torch.nn.Sequential):
    def __init__(self, input_dim: int, hidden_dim: int, num_hidden: int, dropout: float = 0.0) -> None:
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)



# 拆分 SCARF 為 encoder、columnwise layer、decoder
class ScarfEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden, dropout):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dim, num_hidden, dropout)
    def forward(self, x):
        return self.encoder(x)

class ScarfColumnwiseLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 這裡可用 nn.Linear 或自訂 attention，先用 Linear
        self.linear = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, x):
        return self.linear(x)

class ScarfDecoder(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_hidden, dropout):
        super().__init__()
        self.head = MLP(hidden_dim, out_dim, num_hidden, dropout)
    def forward(self, x):
        return self.head(x)

# 保留原本 SCARF 介面，組合 encoder/columnwise/decoder
class SCARF(nn.Module):
    def __init__(
        self,
        input_dim: int,
        features_low: int,
        features_high: int,
        dim_hidden_encoder: int,
        num_hidden_encoder: int,
        dim_hidden_head: int,
        num_hidden_head: int,
        corruption_rate: float = 0.6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = ScarfEncoder(input_dim, dim_hidden_encoder, num_hidden_encoder, dropout)
        self.columnwise = ScarfColumnwiseLayer(dim_hidden_encoder)
        self.decoder = ScarfDecoder(dim_hidden_encoder, dim_hidden_head, num_hidden_head, dropout)
        self.marginals = Uniform(torch.Tensor(features_low), torch.Tensor(features_high))
        self.corruption_rate = corruption_rate

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _ = x.size()
        corruption_mask = torch.rand_like(x, device=x.device) > self.corruption_rate
        x_random = self.marginals.sample(torch.Size((batch_size,))).to(x.device)
        x_corrupted = torch.where(corruption_mask, x_random, x)
        # encoder -> columnwise -> decoder
        z = self.encoder(x)
        z = self.columnwise(z)
        embeddings = self.decoder(z)
        z_corrupt = self.encoder(x_corrupted)
        z_corrupt = self.columnwise(z_corrupt)
        embeddings_corrupted = self.decoder(z_corrupt)
        return embeddings, embeddings_corrupted

    @torch.inference_mode()
    def get_embeddings(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z = self.columnwise(z)
        return z
