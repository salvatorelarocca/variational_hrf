import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily



class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0

    def forward(self, x):
        device   = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        if x.dim() == 1:
            emb = x[..., None] * emb[None, :]
        elif x.dim() == 2:
            emb = x[..., None] * emb[None, None, :]
        elif x.dim() == 3:
            emb = x[..., None] * emb[None, None, None, :]
        else:
            assert False
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SingleEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim=64):
        super().__init__()
        self.pos = SinusoidalPosEmb(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
        )

        self.proj = nn.Linear(in_dim, emb_dim)

    def forward(self, x):
        # x: (B, d) oppure (B,)
        if x.dim() == 1:
            x = x.unsqueeze(1)

        x = self.proj(x)              
        x = x.mean(dim=1)          

        x = self.pos(x)
        return self.mlp(x)


class PosteriorEncoder(nn.Module):
    def __init__(self, data_dim, latent_dim,
                 emb_dim=64, hidden=128):
        super().__init__()

        self.enc_x0 = SingleEncoder(data_dim, emb_dim)
        self.enc_x1 = SingleEncoder(data_dim, emb_dim)
        self.enc_xt = SingleEncoder(data_dim, emb_dim)
        self.enc_t  = SingleEncoder(1, emb_dim)
        

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        self.fc_mu  = nn.Linear(hidden, latent_dim)
        self.fc_var = nn.Linear(hidden, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x0, x1, xt, t):
        # print(x0.shape)
        # x0 = self.enc_x0(x0)
        # print(x0.shape)
        h = torch.cat([
            self.enc_x0(x0),
            self.enc_x1(x1),
            self.enc_xt(xt),
            self.enc_t(t.unsqueeze(1)),
        ], dim=1)

        h = self.mlp(h)
        mu = self.fc_mu(h)
        log_var = torch.clamp(self.fc_var(h), -10, 10)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var



def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    kl = 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1)
    return kl.sum(dim=1).mean()


class VNetD_VAE(nn.Module):
    def __init__(self, data_dim=2, depth=3,
                 hidden_dim=64, latent_dim=8):
        super().__init__()

        dim = 64
        self.data_dim = data_dim
        self.depth = depth

        # === encoders time ===
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            #torch.nn.Linear(depth*dim,depth*dim), DIFF con la versione normale
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        # === encoders data ===
        self.data_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
        )

        # === latent encoding module ===
        self.z_encoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        # flatten dopo encoding
        self.flatten = nn.Flatten()

        # === decoder ===
        input_dim = hidden_dim * data_dim + hidden_dim * data_dim * depth + 128 # concatenazione di time_emb, data_emb e z_emb

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, data_dim),
            nn.GELU(),
        )

    def forward(self, xt, t, z=None):
        # print(f'xt.shape: {xt.shape}, t.shape: {t.shape}')
        t_emb  = self.time_mlp(t)
        xt_emb = self.data_mlp(xt)
        # print(f't_emb.shape: {t_emb.shape}, xt_emb.shape: {xt_emb.shape}')

        t_emb  = self.flatten(t_emb)
        xt_emb = self.flatten(xt_emb)

        # print(f't_emb.shape: {t_emb.shape}, xt_emb.shape: {xt_emb.shape}')
        x = torch.cat([xt_emb, t_emb], dim=1)
        # print(f'x.shape after concat: {x.shape}')
        
        if z is not None:
            z = self.z_encoder(z)   
            # print(f'z.shape after encoding: {z.shape}')
            x = torch.cat([x, z], dim=1) # concatenazione con encoding z
            # print(f'x.shape after adding z: {x.shape}')

        return self.net(x)



def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    kl = 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1)
    return kl.sum(dim=1).mean()



@torch.no_grad()
def sample_hierarchical_vae(model, x_t, t, cur_depth, max_depth, N_list,
                             z=None, return_traj=False):
    """
    Identico a sample_hierarchical originale + argomento z opzionale.
    z viene campionato una volta fuori dal loop per coerenza della traiettoria.
    """
    x_0             = x_t[:, cur_depth, ...].clone()
    local_num_steps = N_list[cur_depth]
    times           = torch.linspace(0.0, 1.0, local_num_steps + 1,
                                     device=x_t.device)
    dts = torch.diff(times)

    if return_traj:
        traj = [x_0]

    for k in range(local_num_steps):
        t[cur_depth] = times[k]
        dt           = dts[k]

        if k == 0 and cur_depth != 0:
            x_0                    = torch.randn_like(x_t[:, cur_depth, ...])
            x_t[:, cur_depth, ...] = x_0

        if cur_depth + 1 == max_depth:
            f = model(
                x_t,
                t * torch.ones((x_t.shape[0], 1), device=x_t.device),
                z=z,
            )
            x_t[:, cur_depth, ...] += dt * f
        else:
            x_t[:, cur_depth, ...] += dt * sample_hierarchical_vae(
                model, x_t, t, cur_depth + 1, max_depth, N_list, z=z
            )[1]

        if return_traj:
            traj.append(x_t[:, cur_depth, ...].detach().clone())

    if return_traj:
        return x_0, x_t[:, cur_depth, ...], torch.stack(traj)
    else:
        return x_0, x_t[:, cur_depth, ...]



def load_ckpt_vae(base_dir, model, posterior, ckpt=None):
    ckpt_dir = os.path.join(base_dir, "ckpt")
    if ckpt is None:
        ckpt = sorted(
            os.listdir(ckpt_dir),
            key=lambda x: int(x.split("_")[-2].split(".")[0]),
        )[-1]
    print(f"loading {ckpt}")
    checkpoint = torch.load(os.path.join(ckpt_dir, ckpt), weights_only=True)
    model.load_state_dict(checkpoint["v_net_state_dict"])
    model.eval()
    if posterior is not None and "posterior_state_dict" in checkpoint:
        posterior.load_state_dict(checkpoint["posterior_state_dict"])
        posterior.eval()
    return model, posterior


def load_ckpt(rf_dir, dim, model, ckpt=None):
    """Compatibilità con hrfD.py originale."""
    ckpt_dir = os.path.join(rf_dir, "ckpt")
    if ckpt is None:
        ckpt = sorted(
            os.listdir(ckpt_dir),
            key=lambda x: int(x.split("_")[-2].split(".")[0]),
        )[-1]
    print(f"loading {ckpt}")
    checkpoint = torch.load(os.path.join(ckpt_dir, ckpt), weights_only=True)
    model.load_state_dict(checkpoint["v_net_state_dict"])
    model.eval()
    return model




@torch.no_grad()
def plot_traj(traj, distance, traj_dir, file_name, title):
    plt.figure()
    n, _, d     = traj.shape
    n_traj      = 1000
    start_color = "#4D4D4D"
    end_color   = "blue"
    flow_color  = "#748B47"

    if d == 1:
        plt.scatter(traj[0, :n_traj, 0].cpu().numpy(), [0]*n_traj,
                    s=4, alpha=1, c=start_color, label="Start", zorder=2)
        plt.scatter(traj[-1, :n_traj, 0].cpu().numpy(), [1]*n_traj,
                    s=4, alpha=1, c=end_color, label="End", zorder=3)
        plt.plot(traj[:, 0, 0].cpu().numpy(), np.linspace(0, 1, n),
                 linewidth=1, alpha=0.4, c=flow_color,
                 label=f"Flow WD={distance:.3f}")
        for i in range(1, n_traj):
            plt.plot(traj[:, i, 0].cpu().numpy(), np.linspace(0, 1, n),
                     linewidth=1, alpha=0.4, c=flow_color)
        plt.xticks(fontsize=18); plt.yticks(fontsize=18)
        plt.xlabel("Space", fontsize=18); plt.ylabel("Time", fontsize=18)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
    else:
        plt.scatter(traj[0, :n_traj, 0].cpu().numpy(),
                    traj[0, :n_traj, 1].cpu().numpy(),
                    s=4, alpha=0.6, c=start_color, label="Start")
        plt.scatter(traj[-1, :n_traj, 0].cpu().numpy(),
                    traj[-1, :n_traj, 1].cpu().numpy(),
                    s=4, alpha=1, c=end_color, label="End", zorder=3)
        plt.plot(traj[:, 0, 0].cpu().numpy(), traj[:, 0, 1].cpu().numpy(),
                 linewidth=1, alpha=0.4, c=flow_color,
                 label=f"Flow SWD={distance:.3f}")
        for i in range(1, n_traj):
            plt.plot(traj[:, i, 0].cpu().numpy(), traj[:, i, 1].cpu().numpy(),
                     linewidth=1, alpha=0.4, c=flow_color)
        plt.axis("off")

    plt.title(title, fontsize=16)
    plt.legend(fontsize=16, framealpha=0.5, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(traj_dir, file_name), dpi=300,
                bbox_inches="tight")
    plt.close()




class LowDimData:
    def __init__(self, data_type, device):
        self.batchsize = 100000
        self.device    = device
        self.data_init(data_type)
        self.pairs = torch.stack([self.x0, self.x1], dim=1)

    def data_init(self, data_type):
        if data_type == "1to2":
            self.dim   = 1
            self.mean  = torch.tensor([1])
            self.means = torch.ones((2, self.dim)) * self.mean
            self.means[1] = -self.means[1]
            self.var   = torch.tensor([0.02])
            self.covs  = self.var * torch.stack(
                [torch.eye(self.dim) for _ in range(2)])
            self.probs = torch.tensor([0.5, 0.5])
            tm = Categorical(self.probs)
            tc = MultivariateNormal(self.means, self.covs)
            self.target_model  = MixtureSameFamily(tm, tc)
            self.initial_model = MultivariateNormal(
                torch.zeros(self.dim), torch.eye(self.dim))
            self.x1 = self.target_model.sample(
                [self.batchsize]).to(self.device).detach()
            self.x0 = self.initial_model.sample(
                [self.batchsize]).to(self.device).detach()

        elif data_type == "2to2":
            self.dim   = 1
            self.mean  = torch.tensor([1])
            self.means = torch.ones((2, self.dim)) * self.mean
            self.means[1] = -self.means[1]
            self.var   = torch.tensor([0.1])
            self.covs  = self.var * torch.stack(
                [torch.eye(self.dim) for _ in range(2)])
            self.probs = torch.tensor([0.5, 0.5])
            tm  = Categorical(self.probs); tc  = MultivariateNormal(self.means, self.covs)
            im  = Categorical(self.probs); ic  = MultivariateNormal(self.means, self.covs)
            self.target_model  = MixtureSameFamily(tm, tc)
            self.initial_model = MixtureSameFamily(im, ic)
            self.x1 = self.target_model.sample(
                [self.batchsize]).to(self.device).detach()
            self.x0 = self.initial_model.sample(
                [self.batchsize]).to(self.device).detach()

        elif data_type == "1to5":
            self.dim   = 1
            self.means = torch.ones((5, self.dim)) * 5
            for i in range(5):
                self.means[i] *= i - 2
            self.var   = torch.tensor([0.5])
            self.covs  = self.var * torch.stack(
                [torch.eye(self.dim) for _ in range(5)])
            self.probs = torch.tensor([0.2]*5)
            tm = Categorical(self.probs); tc = MultivariateNormal(self.means, self.covs)
            self.target_model  = MixtureSameFamily(tm, tc)
            self.initial_model = MultivariateNormal(
                torch.zeros(self.dim), torch.eye(self.dim))
            self.x1 = self.target_model.sample(
                [self.batchsize]).to(self.device).detach()
            self.x0 = self.initial_model.sample(
                [self.batchsize]).to(self.device).detach()
            self.x1 = (self.x1 - torch.mean(self.x1)) / torch.std(self.x1)

        elif data_type == "2D1to6":
            self.dim   = 2
            D          = 10.
            self.probs = torch.tensor([1/6]*6)
            self.means = torch.tensor([
                [ D*np.sqrt(3)/2.,  D/2.], [-D*np.sqrt(3)/2.,  D/2.],
                [0.0,              -D   ], [ D*np.sqrt(3)/2., -D/2.],
                [-D*np.sqrt(3)/2., -D/2.], [0.0,               D   ],
            ]).float()
            self.var   = torch.tensor([0.5])
            self.covs  = self.var * torch.stack(
                [torch.eye(self.dim) for _ in range(6)])
            tm = Categorical(self.probs); tc = MultivariateNormal(self.means, self.covs)
            self.target_model  = MixtureSameFamily(tm, tc)
            self.initial_model = MultivariateNormal(
                torch.zeros(self.dim), torch.eye(self.dim))
            self.x1 = self.target_model.sample(
                [self.batchsize]).to(self.device).detach()
            self.x0 = self.initial_model.sample(
                [self.batchsize]).to(self.device).detach()
            self.x1 = (self.x1 - torch.mean(self.x1)) / torch.std(self.x1)

        elif data_type == "moon":
            self.dim     = 2
            n_out        = self.batchsize // 2
            n_in         = self.batchsize - n_out
            ox = np.cos(np.linspace(0, np.pi, n_out))
            oy = np.sin(np.linspace(0, np.pi, n_out))
            ix = 1 - np.cos(np.linspace(0, np.pi, n_in))
            iy = 1 - np.sin(np.linspace(0, np.pi, n_in)) - .5
            X  = np.vstack([np.append(ox, ix), np.append(oy, iy)]).T
            X += np.random.rand(self.batchsize, 1) * 0.2
            self.x1 = (torch.from_numpy(X)*3-1).float().to(self.device).detach()
            self.x1 = self.x1[torch.randperm(self.batchsize)]
            self.probs = torch.tensor([1/8]*8)
            self.means = torch.tensor([
                (1,0),(-1,0),(0,1),(0,-1),
                (1/np.sqrt(2),1/np.sqrt(2)),(1/np.sqrt(2),-1/np.sqrt(2)),
                (-1/np.sqrt(2),1/np.sqrt(2)),(-1/np.sqrt(2),-1/np.sqrt(2)),
            ]).float() * 5
            self.var  = torch.tensor([0.1])
            self.covs = self.var * torch.stack(
                [torch.eye(self.dim) for _ in range(8)])
            im = Categorical(self.probs); ic = MultivariateNormal(self.means, self.covs)
            self.initial_model = MixtureSameFamily(im, ic)
            self.x0 = self.initial_model.sample(
                [self.batchsize]).to(self.device).detach()
        else:
            raise NotImplementedError