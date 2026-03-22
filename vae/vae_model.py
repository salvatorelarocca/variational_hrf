import torch
from torch import nn
from typing import List

Tensor = torch.Tensor

class BetaVAE(nn.Module):
    """
    VAE per HRF che prende in input:
        v0:  campo di velocità al tempo 0    [B, C, H, W]
        v1:  campo di velocità al tempo 1    [B, C, H, W]
        vtau: campo di velocità al tempo tau [B, C, H, W]
        tau:  tempo scalare                  [B]
    Restituisce z, mu, log_var.

    Architettura:
        - CNN encoder condiviso per v0, v1, vtau
        - AdaptiveAvgPool2d(pooling_size, pooling_size) per gestire qualsiasi dimensione input
        - tau_embed: proietta tau scalare in tau_embed_dim dimensioni
          (invece di concatenare un singolo scalare che avrebbe peso trascurabile
          rispetto ai vettori dei campi)
        - fc_hidden: layer intermedio per aumentare la capacita' rappresentativa
        - fc_mu, fc_var: proiettano nello spazio latente
    """

    def __init__(
        self,
        in_channels:    int       = 3,
        latent_dim:     int       = 128,
        hidden_dims:    List[int] = None,
        pooling_size:   int       = 2,
        tau_embed_dim:  int       = 64,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128]

        # ── CNN encoder (condiviso per i tre campi) ──
        modules = []
        _in_ch = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(_in_ch, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=h_dim),
                    nn.LeakyReLU(),
                )
            )
            _in_ch = h_dim

        self.encoder = nn.Sequential(
            *modules,
            nn.AdaptiveAvgPool2d((pooling_size, pooling_size)),
        )

        #tau proiettato in tau_embed_dim dimensioni per dargli più valore
        self.tau_embed = nn.Sequential(
            nn.Linear(1, tau_embed_dim),
            nn.LeakyReLU(),
        )

        # 3 campi * hidden_dims[-1] * pooling_size^2  +  tau_embed_dim
        spatial_dim  = pooling_size * pooling_size
        fc_input_dim = hidden_dims[-1] * spatial_dim * 3 + tau_embed_dim

        # Layer intermedio dovrebbe essere migliore
        fc_hidden_dim = fc_input_dim // 2
        self.fc_hidden = nn.Sequential(
            nn.Linear(fc_input_dim, fc_hidden_dim),
            nn.LeakyReLU(),
        )

        #Proiezione nello spazio latente
        self.fc_mu  = nn.Linear(fc_hidden_dim, latent_dim)
        self.fc_var = nn.Linear(fc_hidden_dim, latent_dim)

        # inizializza i bias di fc_var a zero:
        # log_var parte da 0 => varianza iniziale = 1 = prior N(0,1)
        # stabilizza il training nelle prime iterazioni
        nn.init.zeros_(self.fc_var.bias)

    def encode(self, v0: Tensor, v1: Tensor, vtau: Tensor, tau: Tensor):
        # estrae feature spaziali da ogni campo
        h_v0   = torch.flatten(self.encoder(v0),   1)   # [B, hidden[-1]*pooling_size^2]
        h_v1   = torch.flatten(self.encoder(v1),   1)
        h_vtau = torch.flatten(self.encoder(vtau), 1)

        # embedding di tau: [B] -> [B, 1] -> [B, tau_embed_dim]
        tau_emb = self.tau_embed(tau.unsqueeze(1))       # [B, tau_embed_dim]

        # concatena tutto
        h = torch.cat([h_v0, h_v1, h_vtau, tau_emb], dim=1)

        # layer intermedio
        h = self.fc_hidden(h)

        mu      = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, v0: Tensor, v1: Tensor, vtau: Tensor, tau: Tensor):
        mu, log_var = self.encode(v0, v1, vtau, tau)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var