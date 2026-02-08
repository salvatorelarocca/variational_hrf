import torch
from torch import nn
from typing import List

Tensor = torch.Tensor

class BetaVAE(nn.Module):
    """
    Del VAE tengo solo l'encoder e il trick
    VAE per HRF che prende in input:
        x0, x1: campi di posizione o velocità [B, C, H, W]
        v0, v1: campi di velocità [B, C, H, W]
        tau: tempo scalare [B, 1]
    Restituisce uno spazio latente z.

    Idea di concatenazione è passare gli input matriciali a CNN separati, ottenere hidden di dimensione fissa 
    (indipendente da H e W) e poi concatenare tutto in un vettore da passare a fc_mu e fc_var.
    """

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List[int] = None):
        super().__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        # Encoder CNN con AdaptiveAvgPool2d per rendere indipendente dalla dimensione spaziale
        self.encoder = nn.Sequential(
            *modules,
            nn.AdaptiveAvgPool2d((1, 1)) #forza output di dimensione (B, hidden_dims[-1], 1, 1)
        )

        # fc_mu e fc_var: input è hidden_dims[-1]*numero_campi + 1 (tau)
        # consideriamo concatenati x0,x1,v0,v1 => 4 campi
        self.fc_mu = nn.Linear(hidden_dims[-1]*3 + 1, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*3 + 1, latent_dim)

    def encode(self, v0: Tensor, v1: Tensor, vtau: Tensor, tau: Tensor):
        # passa ogni matrice nella CNN encoder
        h_v0 = torch.flatten(self.encoder(v0), 1)
        h_v1 = torch.flatten(self.encoder(v1), 1)
        h_vtau = torch.flatten(self.encoder(vtau), 1)

        tau = tau.view(1, 1)
        tau = tau.expand(h_v0.size(0), -1)  # espande tau per concatenarlo agli hidden

        # concatena tutti gli hidden + tempi 
        h = torch.cat([h_v0, h_v1, h_vtau, tau], dim=1)

        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu: Tensor, log_var: Tensor):
        std = torch.exp(0.5 * log_var) # deviazione standard e^(0.5*log_var) = sqrt(e^log_var) = sqrt(var) = std
        eps = torch.randn_like(std) # campione casuale da distribuzione normale standard con stessa forma di std
        return mu + eps * std  # trick di reparametrizzazione: z = mu + std * eps

    def forward(self, v0: Tensor, v1: Tensor, vtau: Tensor, tau: Tensor):
        mu, log_var = self.encode(v0, v1, vtau, tau)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
