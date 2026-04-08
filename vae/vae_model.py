import torch
from torch import nn
from typing import List
Tensor = torch.Tensor


class BetaVAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 64,
        hidden_dims: List[int] = None,
        pooling_size: int = 2,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        def make_cnn():
            modules, _in = [], in_channels
            for h in hidden_dims:
                modules.append(nn.Sequential(
                    nn.Conv2d(_in, h, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=h),
                    nn.SiLU(),
                ))
                _in = h
            modules.append(nn.AdaptiveAvgPool2d((pooling_size, pooling_size)))
            return nn.Sequential(*modules)

        # Encoder separati — semantiche diverse meritano pesi diversi
        # state_start è rumore: encoder più leggero
        self.enc_target = make_cnn()
        self.enc_state  = make_cnn()
        # state_start è N(0,I) — non ha feature visive,
        # lo encodiamo lo stesso per coerenza ma con meno peso
        self.enc_start  = make_cnn()

        feat = hidden_dims[-1] * pooling_size * pooling_size
        self.feature_dim = feat

        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Proiezioni su spazio comune
        self.proj_start  = nn.Linear(feat, 256)
        self.proj_target = nn.Linear(feat, 256)
        self.proj_state  = nn.Linear(feat, 256)

        # Fusion: tutti e tre entrano direttamente + time
        # 256*3 + time_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(256 * 3 + time_embed_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
        )

        self.fc_mu  = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)
        nn.init.zeros_(self.fc_var.bias)

    def encode(
        self,
        state_start: Tensor,
        state_target: Tensor,
        state_t: Tensor,
        time: Tensor,
    ):
        h_start  = self.proj_start(torch.flatten(self.enc_start(state_start), 1))
        h_target = self.proj_target(torch.flatten(self.enc_target(state_target), 1))
        h_state  = self.proj_state(torch.flatten(self.enc_state(state_t), 1))

        t_emb = self.time_embed(time.unsqueeze(1))

        # Tutti e tre entrano direttamente — nessuna info viene scartata
        h = torch.cat([h_start, h_target, h_state, t_emb], dim=1)
        h = self.fusion(h)

        mu      = self.fc_mu(h)
        log_var = torch.clamp(self.fc_var(h), -10, 10)
        return mu, log_var

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(
        self,
        state_start: Tensor,
        state_target: Tensor,
        state_t: Tensor,
        time: Tensor,
    ):
        mu, log_var = self.encode(state_start, state_target, state_t, time)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var