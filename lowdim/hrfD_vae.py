import os

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from absl import app, flags
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import time

# from sample import FLAGS
from utils_vae import (
    LowDimData,
    PosteriorEncoder,
    VNetD_VAE,
    kl_divergence,
    load_ckpt_vae,
    plot_traj,
    sample_hierarchical_vae,
)


def train_hrf_vae(data, depth, N_list, checkpoint, iterations, base_dir,
                  seed, device,
                  latent_dim, beta, lr_vnet, lr_posterior, grad_clip):

    ckpt_dir = os.path.join(base_dir, "ckpt")
    img_dir  = os.path.join(base_dir, "fig")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir,  exist_ok=True)
    loss_curve  = []
    recon_curve = []
    kl_curve    = []

    # modelli
    v_net     = VNetD_VAE(data_dim=data.dim, depth=depth,
                          latent_dim=latent_dim).to(device)
    print(f'data.dim: {data.dim}')
    print(f'depth: {depth}')
    print(f'latent_dim: {latent_dim}')
    posterior = PosteriorEncoder(data_dim=data.dim,
                                 latent_dim=latent_dim).to(device)

    vnet_params = sum(p.data.nelement() for p in v_net.parameters())
    post_params = sum(p.data.nelement() for p in posterior.parameters())
    print(f"VNetD_VAE  params: {vnet_params} ({vnet_params/1e6:.3f} M)")
    print(f"Posterior  params: {post_params} ({post_params/1e6:.3f} M)")

    # AdamW — lr separati: posterior più lento per stabilità della rappresentazione
    # Il paper usa lr=1e-3 per entrambi; se vuoi seguire il paper esattamente
    # imposta --lr_posterior=1e-3
    optimizer = torch.optim.AdamW([
        {"params": v_net.parameters(),     "lr": lr_vnet},
        {"params": posterior.parameters(), "lr": lr_posterior},
    ])

    # matrice triangolare inferiore per costruire xt gerarchico
    A = torch.tril(torch.ones((depth, depth), device=device), diagonal=-1)

    with tqdm(initial=0, total=iterations) as pbar:
        for train_i in range(iterations):
            optimizer.zero_grad()

            # campiona batch
            indices = torch.randperm(len(data.pairs))[:checkpoint["batchsize"]]
            batch   = data.pairs[indices]
            x0 = batch[:, 0].detach().clone()   # (B, d)
            x1      = batch[:, 1].detach().clone()   # (B, d)
            
            x0 = torch.cat([
                x0[:, None, :],
                torch.randn((x0.shape[0], depth - 1) + x0.shape[1:],
                            device=device),
            ], dim=1)   # (B, depth, d)

            t  = torch.rand(
                (x1.shape[0], depth) + (1,) * (x1.dim() - 1),
                device=device,
            )

            # costruzione xt identica a hrfD originale
            xt = (1 - t) * x0 + t * (
                x1[:, None, ...] - torch.einsum("ij,bj...->bi...", A, x0)
            )

            target = x1 - torch.sum(x0, dim=1)   # (B, d)  

            # t squeezed per VNetD_VAE e per PosteriorEncoder
            t_sq = t.squeeze(-1)   # (B, depth)
            
            # print("Shapes for PosteriorEncoder:", 
            #       x0[:, depth-1, :].shape, 
            #       x1.shape,
            #       xt[:, depth-1, :].shape, 
            #       t_sq[:, depth-1].shape, 
            #       )
            
            z, mu, log_var = posterior(
                x0=x0[:, 0, :],      # (B, d) 
                x1=x1,               # (B, d)  
                xt=xt[:, 0, :],      # (B, d) 
                t=t_sq[:, 0],        # (B,) - tempo più interno per essere compatibili con l'architettura UNET+VAE
            )

            pred = v_net(xt, t, z=z)

            # --- recon + beta * KL ---
            recon_loss = torch.mean((target - pred) ** 2)
            kl_loss    = kl_divergence(mu, log_var)
            loss       = recon_loss + beta * kl_loss

            loss.backward()

            all_params = list(v_net.parameters()) + list(posterior.parameters())

            optimizer.step()

            loss_curve.append(loss.item())
            recon_curve.append(recon_loss.item())
            kl_curve.append(kl_loss.item())

            pbar.set_description(
                f"loss={loss.item():.4f}  "
                f"recon={recon_loss.item():.4f}  "
                f"kl_w={kl_loss.item()*beta:.6f}  "
                f"mu={mu.mean().item():.3f}  "
                f"std={log_var.exp().sqrt().mean().item():.3f}"
            )
            pbar.update(1)

            if (train_i + 1) % checkpoint["save_every_steps"] == 0 \
                    or train_i == (iterations - 1):

                checkpoint["v_net_state_dict"]     = v_net.state_dict()
                checkpoint["posterior_state_dict"] = posterior.state_dict()
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                checkpoint["step"]                 = train_i + 1
                torch.save(
                    checkpoint,
                    os.path.join(
                        ckpt_dir,
                        f"hrf_vae_{train_i+1}_D{depth}_seed{seed}.pt",
                    ),
                )

                v_net.eval()

                x0_eval = data.initial_model.sample(
                    [data.batchsize]).to(device).detach()
                x0_eval = torch.cat([
                    x0_eval[:, None, :],
                    torch.randn(
                        (x0_eval.shape[0], depth - 1) + x0_eval.shape[1:],
                        device=device,
                    ),
                ], dim=1)
                t0 = torch.zeros(depth, device=device)

                with torch.no_grad():
                    # z ~ N(0, I) 
                    z_sample = torch.randn(
                        data.batchsize, latent_dim, device=device
                    )
                    _, xt_gen = sample_hierarchical_vae(
                        v_net, x0_eval, t0, 0, depth, N_list, z=z_sample
                    )

                if data.dim == 1:
                    distance = wasserstein_distance(
                        data.x1[:, 0].cpu().numpy(),
                        xt_gen[:, 0].cpu().numpy(),
                    )
                else:
                    distance = ot.sliced_wasserstein_distance(
                        xt_gen, data.x1, seed=1
                    )

                log_file = os.path.join(base_dir, "metrics.txt")

                with open(log_file, "a") as f:
                    f.write(
                        f"\nstep={train_i+1} "
                        f"Data_type={FLAGS.data_type} "
                        f"WD={distance:.6f} "
                        f"NFE={np.prod(N_list)} "
                        f"N_list={N_list} "
                        f"mode={FLAGS.mode}"
                    )

                print(
                    f"step={train_i+1}  WD={distance:.6f}  "
                    f"NFE={np.prod(N_list)}  {N_list}"
                )

                # plot distribuzione
                plt.figure()
                if data.dim == 1:
                    bins = np.linspace(-2, 2, 201)
                    plt.hist(data.x1[:, 0].cpu().numpy(), bins=bins,
                             density=True, alpha=0.8, histtype="step",
                             linewidth=1, label="Target")
                    plt.hist(xt_gen.cpu().numpy(), bins=bins,
                             density=True, alpha=0.8, histtype="step",
                             linewidth=1, label=f"Gen WD={distance:.3f}")
                    plt.xlabel("Data"); plt.ylabel("Density")
                else:
                    plt.scatter(data.x0[:5000, 0].cpu().numpy(),
                                data.x0[:5000, 1].cpu().numpy(),
                                label="Source", alpha=0.25, s=3)
                    plt.scatter(data.x1[:5000, 0].cpu().numpy(),
                                data.x1[:5000, 1].cpu().numpy(),
                                label="Target", alpha=0.25, s=3)
                    plt.scatter(xt_gen[:5000, 0].cpu().numpy(),
                                xt_gen[:5000, 1].cpu().numpy(),
                                label=f"Gen SWD={distance:.3f}",
                                alpha=0.25, s=3)
                plt.legend()
                plt.title(
                    f"VAE step={train_i+1} NFE={np.prod(np.array(N_list))} "
                    f"beta={beta}"
                )
                plt.tight_layout()
                plt.savefig(os.path.join(img_dir, f"dist_hrf_vae_progress{time.time()}.png"))
                plt.close()

                # plot losses
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                axes[0].plot(loss_curve);  axes[0].set_title("Total Loss")
                axes[1].plot(recon_curve); axes[1].set_title("Recon Loss")
                axes[2].plot(kl_curve);    axes[2].set_title("KL Loss")
                for ax in axes:
                    ax.set_xlabel("Step")
                plt.tight_layout()
                plt.savefig(os.path.join(img_dir, f"losses_hrf_vae{time.time()}.png"))
                plt.close()

                v_net.train()

    return v_net, posterior




def main(argv):
    seed = FLAGS.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    device = (
        torch.device(f"cuda:{FLAGS.gpu}")
        if FLAGS.gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )

    data     = LowDimData(data_type=FLAGS.data_type, device=device)
    N_list   = [int(x) for x in FLAGS.N_list]
    depth    = len(N_list)

    # beta automatico secondo paper: 1.0 per 1D, 0.1 per 2D
    beta = FLAGS.beta if FLAGS.beta is not None else (1.0 if data.dim == 1 else 0.1)

    # latent_dim automatico secondo paper: 4 per 1D, 8 per 2D
    latent_dim = FLAGS.latent_dim if FLAGS.latent_dim is not None \
        else (4 if data.dim == 1 else 8)

    base_dir = os.path.join(FLAGS.base_dir, FLAGS.data_type, "hrf_vae")
    os.makedirs(base_dir, exist_ok=True)

    img_dir  = os.path.join(base_dir, "fig")
    dist_dir = os.path.join(img_dir,  "dist")
    traj_dir = os.path.join(img_dir,  "traj")
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    checkpoint = {
        "v_net_state_dict":     None,
        "posterior_state_dict": None,
        "optimizer_state_dict": None,
        "save_every_steps":     FLAGS.save_every,
        "step":                 0,
        "batchsize":            FLAGS.batchsize,
    }


    if FLAGS.mode == "train":
        train_hrf_vae(
            data          = data,
            depth         = depth,
            N_list        = N_list,
            checkpoint    = checkpoint,
            iterations    = FLAGS.iter,
            base_dir      = base_dir,
            seed          = seed,
            device        = device,
            latent_dim    = latent_dim,
            beta          = beta,
            lr_vnet       = FLAGS.lr_vnet,
            lr_posterior  = FLAGS.lr_posterior,
        )

    elif FLAGS.mode == "eval":
        step      = FLAGS.iter
        ckpt_name = f"hrf_vae_{step}_D{depth}_seed{seed}.pt"

        v_net     = VNetD_VAE(data_dim=data.dim, depth=depth,
                              latent_dim=latent_dim).to(device)
        posterior = PosteriorEncoder(data_dim=data.dim,
                                     latent_dim=latent_dim).to(device)
        v_net, posterior = load_ckpt_vae(base_dir, v_net, posterior,
                                         ckpt=ckpt_name)

        vnet_params = sum(p.data.nelement() for p in v_net.parameters())
        print(f"VNetD_VAE params: {vnet_params} ({vnet_params/1e6:.3f} M)")

        x0_eval = data.initial_model.sample(
            (data.batchsize,)).to(device).detach()
        x0_eval = torch.cat([
            x0_eval[:, None, :],
            torch.randn(
                (x0_eval.shape[0], depth - 1) + x0_eval.shape[1:],
                device=device,
            ),
        ], dim=1)
        t0 = torch.zeros(depth, device=device)

        with torch.inference_mode():
            # z ~ N(0, I) — q_phi omessa come da specifica paper
            z_sample = torch.randn(data.batchsize, latent_dim, device=device)
            x0_start, xt_gen, traj = sample_hierarchical_vae(
                v_net, x0_eval, t0, 0, depth, N_list,
                z=z_sample, return_traj=True,
            )

        if data.dim == 1:
            distance = wasserstein_distance(
                data.x1[:, 0].cpu().numpy(),
                xt_gen[:, 0].cpu().numpy(),
            )
        else:
            distance = ot.sliced_wasserstein_distance(
                data.x1, xt_gen, seed=1
            )

        plot_traj(
            traj, distance, traj_dir,
            file_name=f"traj_{ckpt_name}_{N_list}.png",
            title=f"VAE Trajectory {N_list} steps  WD={distance:.4f}",
        )

        log_file = os.path.join(base_dir, "metrics.txt")

        with open(log_file, "a") as f:
                f.write(
                    f"\nWD={distance:.6f} "
                    f"Data_type={FLAGS.data_type} "
                    f"NFE={np.prod(N_list)} "
                    f"N_list={N_list} "
                    f"mode={FLAGS.mode}"
                    f"beta={beta}"
                )

        plt.figure()
        if data.dim == 1:
            bins = np.linspace(-2, 2, 201)
            plt.hist(data.x1[:, 0].cpu().numpy(), bins=bins,
                     color="#ff7f0e", density=True, alpha=0.8,
                     histtype="step", linewidth=1, label="Target")
            plt.hist(xt_gen.cpu().numpy(), bins=bins,
                     color="#2ca02c", density=True, alpha=0.8,
                     histtype="step", linewidth=1,
                     label=f"Gen WD={distance:.3f}")
            plt.xlabel("Space", fontsize=16)
            plt.ylabel("Density", fontsize=16)
        else:
            plt.scatter(data.x0[:5000, 0].cpu().numpy(),
                        data.x0[:5000, 1].cpu().numpy(),
                        c="#1f77b4", label="Source", alpha=0.25, s=3)
            plt.scatter(data.x1[:5000, 0].cpu().numpy(),
                        data.x1[:5000, 1].cpu().numpy(),
                        c="#ff7f0e", label="Target", alpha=0.25, s=3)
            plt.scatter(xt_gen[:5000, 0].cpu().numpy(),
                        xt_gen[:5000, 1].cpu().numpy(),
                        c="#2ca02c",
                        label=f"Gen SWD={distance:.3f}",
                        alpha=0.25, s=3)
            plt.axis("off")

        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.xticks(fontsize=16); plt.yticks(fontsize=16)
        plt.legend(fontsize=16, framealpha=0.5, loc="upper right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(dist_dir, f"xdist_{ckpt_name}_{N_list}.png"),
            dpi=300, bbox_inches="tight",
        )
        plt.close()

        print(f"\nWD/SWD = {distance:.6f}  NFE = {np.prod(N_list)}  {N_list}")



if __name__ == "__main__":
    FLAGS = flags.FLAGS

    flags.DEFINE_enum(
        "data_type", None,
        ["1to2", "1to5", "2D1to6", "moon", "2to2"],
        "tipo di dataset sintetico",
    )
    flags.DEFINE_integer("batchsize",   1000,   "batch size (paper: 1000)")
    flags.DEFINE_integer("iter",       20000,   "iterazioni totali (paper: 20k)")
    flags.DEFINE_integer("gpu",            0,   "indice GPU (-1 per CPU)")
    flags.DEFINE_integer("seed",           0,   "random seed")
    flags.DEFINE_string( "base_dir", "lowdim",  "cartella base risultati")
    flags.DEFINE_enum(   "mode",        None,   ["train", "eval"], "modalità")
    flags.DEFINE_integer("save_every",  5000,   "salva checkpoint ogni N step")

    # N_list come lista, es. --n_list=2,5,10
    flags.DEFINE_list("N_list", ["10", "10"],
                      "N_list per sampling gerarchico, es. 10,10")

    # VAE — None = automatico secondo paper (4 per 1D, 8 per 2D)
    flags.DEFINE_integer("latent_dim", None,
                         "dim spazio latente (None=auto: 4 per 1D, 8 per 2D)")
    # beta — None = automatico secondo paper (1.0 per 1D, 0.1 per 2D)
    flags.DEFINE_float("beta", None,
                       "peso KL (None=auto: 1.0 per 1D, 0.1 per 2D)")

    # learning rate — paper usa 1e-3 per entrambi
    flags.DEFINE_float("lr_vnet",      1e-3,
                       "lr VNetD_VAE (paper: 1e-3)")
    flags.DEFINE_float("lr_posterior", 1e-3,
                       "lr PosteriorEncoder (paper: 1e-3; 1e-4 per più stabilità)")
    flags.DEFINE_float("grad_clip",    1.0,
                       "norma massima gradienti (clipping congiunto)")

    app.run(main)