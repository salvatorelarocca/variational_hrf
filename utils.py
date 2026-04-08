import copy
import os
import time

import torch
from torch import distributed as dist
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchvision.utils import save_image


def sample_rf(model, sample_shape, nfe, device, integration_method="euler", use_z=False, latent_dim=128):
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    if integration_method == "euler":
        xt = sample_rf_euler(model, sample_shape, nfe, device, use_z=use_z, latent_dim=latent_dim)
    elif integration_method == "dopri5":
        xt, nfe = sample_rf_dopri5(model, sample_shape, device, use_z=use_z, latent_dim=latent_dim)
    else:
        raise NotImplementedError
    torch.cuda.synchronize() if device.type == "cuda" else None
    interval = time.perf_counter() - start
    return xt, nfe, interval


def sample_rf_euler(model, sample_shape, nfe, device, use_z=False, latent_dim=128):
    batchsize = sample_shape[0]

    with torch.no_grad():
        if use_z:
            print("Sampling with z for RF baseline...")
            z = torch.randn(batchsize, latent_dim, device=device)
        else:
            z = None
        def wrapped_model(t, xt, args=None):
            return model(t, xt, z=z)

        node = NeuralODE(wrapped_model, solver="euler", sensitivity="adjoint")

        traj = node.trajectory(
            torch.randn(sample_shape, device=device),
            t_span=torch.linspace(0, 1, nfe, device=device),
        )

        return traj[-1].reshape(sample_shape)


def sample_rf_dopri5(model, sample_shape, device, use_z=False, latent_dim=128):
    with torch.no_grad():
        batchsize = sample_shape[0]
        step_counter = {"steps": 0}
        if use_z:
            print("Sampling with z for RF baseline...")
            z = torch.randn(batchsize, latent_dim, device=device)
        else:
            z = None
        def wrapped_model(t, xt, args=None):
            step_counter["steps"] += 1 #ogni volta che viene chiamato il modello, incrementa il contatore
            return model(t, xt, z=z)
        t_span = torch.linspace(0, 1, 2, device=device)
        xt = odeint(
            wrapped_model,
            torch.randn(sample_shape, device=device),
            t_span,
            rtol=1e-5,
            atol=1e-5,
            method="dopri5",
        )
        xt = xt[-1, :]
    return xt, step_counter['steps']


def sample_hrf(model, sample_shape, N, M, device, integration_method="euler", latent_dim=128, use_z=True):
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    if integration_method == "euler":
        xt = sample_hrf_euler(model, sample_shape, N, M, device, latent_dim=latent_dim, use_z=use_z)
        nfe = N * M
    elif integration_method == "dopri5":
        xt, nfe = sample_hrf_dopri5(model, sample_shape, N, device, latent_dim=latent_dim, use_z=use_z)
    else:
        raise NotImplementedError
    torch.cuda.synchronize() if device.type == "cuda" else None
    interval = time.perf_counter() - start
    return xt, nfe, interval


def sample_hrf_euler(model, sample_shape, N, M, device, latent_dim=128, use_z=True):
    with torch.no_grad():
        batchsize = sample_shape[0]
        xt = torch.randn(sample_shape, device=device)
        t_values = torch.arange(N, device=device) / N
        tau_values = torch.arange(M, device=device) / M
        print("Sampling with z for HRF model...") if use_z else print("Sampling without z for HRF model...")
        for i in range(N):
            t = t_values[i].expand(batchsize)
            vtau = torch.randn(sample_shape, device=device)
            z = torch.randn(batchsize, latent_dim, device=device) if use_z else None #se lo campiono fuori è coerente con la traiettoria, se lo campiono dentro è incoerente ma più vario
            for j in range(M):
                tau = tau_values[j].expand(batchsize)
                a = model(tau, vtau, t, xt, z=z)
                vtau += a / M
            xt += vtau / N
    return xt


def sample_hrf_dopri5(model, sample_shape, N, device, latent_dim=128, use_z=True):
    with torch.no_grad():
        batchsize = sample_shape[0]
        xt = torch.randn(sample_shape, device=device)
        t_values = torch.arange(N, device=device) / N
        step_counter = {"steps": 0}
        print("Sampling with z for HRF model...") if use_z else print("Sampling without z for HRF model...")
        for i in range(N):
            t = t_values[i].expand(batchsize)
            z = torch.randn(batchsize, latent_dim, device=device) if use_z else None
            def wrapped_model(tau, vtau, args=None):
                step_counter["steps"] += 1
                return model(tau, vtau, t, xt, z=z)
            tau_span = torch.linspace(0, 1, 2, device=device)
            vtau = odeint(
                wrapped_model,
                torch.randn_like(xt),
                tau_span,
                rtol=1e-5,
                atol=1e-5,
                method="dopri5",
            )
            vtau = vtau[-1, :]
            xt += vtau / N
    return xt, step_counter['steps']


def generate_samples(model, savedir, step, shape, device, net_="normal",
                     integration_method="euler", hrf=True, latent_dim=128, use_z=True):
    model.eval()
    model_ = copy.deepcopy(model)
    if hrf:
        samples, _, _ = sample_hrf(model_, shape, 2, 100, device,
                                 integration_method=integration_method,
                                 latent_dim=latent_dim, use_z=use_z)
    else:
        samples, _, _ = sample_rf(model_, shape, 100, device,
                                integration_method=integration_method,
                                latent_dim=latent_dim, use_z=use_z)
    save_image(samples.clip(-1, 1) / 2 + 0.5,
               os.path.join(savedir, f"{net_}_generated_FM_images_step_{step}.png"), nrow=4)
    model.train()


def ema(source, target, decay):
    source_dict = source.state_dict() # parametri(pesi, bais...) del modello sotto forma di dizionario
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )
    

def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader): #dataloader è un iterable iter(dataloader) restituisce un iterator x, y tupla di batch
            yield x


def load_model(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)


def setup(
        rank: int,
        total_num_gpus: int,
        master_addr: str = "localhost",
        master_port: str = "12355",
        backend: str = "nccl",
    ):
   

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    print(f"[Rank {rank}] MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    print(f"[Rank {rank}] Initializing process group...")
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )
    print(f"[Rank {rank}] Process group initialized!")

