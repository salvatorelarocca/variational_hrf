import os
from datetime import datetime

import csv
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from absl import app, flags
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from utilsvae import load_ckpt, plot_traj
from utilsvae import LowDimData, VNetD, PosteriorEncoder

@torch.no_grad()
def sample_hierarchical(model, x_t, t, cur_depth, max_depth, N_list, return_traj=False, z=None):
    x_0 = x_t[:,cur_depth,...].clone()
    local_num_steps = N_list[cur_depth]
    times = torch.linspace(0.0,1.0,local_num_steps+1,device=x_t.device)
    dts = torch.diff(times)
    if return_traj:
        traj = [x_0]
    for k in range(local_num_steps):
        current_time = times[k]
        dt = dts[k]
        t[cur_depth] = current_time
        if k == 0 and cur_depth != 0:
            x_0 = torch.randn_like(x_t[:,cur_depth,...],device=x_t.device)
            x_t[:,cur_depth,...] = x_0
        if cur_depth+1==max_depth:
            f = model(x_t,t*torch.ones((x_t.shape[0],1),device=x_t.device), z=z) #innesto z per VAE
            x_t[:,cur_depth,...] += dt*f
        else:
            x_t[:,cur_depth,...] += dt*sample_hierarchical(model, x_t, t, cur_depth+1, max_depth, N_list, z=z)[1] #innesto nella chiamata ricorsiva
        if return_traj:
            traj.append(x_t[:,cur_depth,...].detach().clone())
    if return_traj:
        return x_0, x_t[:,cur_depth,...], torch.stack(traj)
    else:
        return x_0, x_t[:,cur_depth,...]


def train_hrf(data, depth, N_list, checkpoint, iterations, base_dir, seed, device, progress):
    ckpt_dir = os.path.join(base_dir, f"ckpt")
    img_dir = os.path.join(base_dir, f"fig")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    loss_curve = []
    recons_loss_curve = []
    kld_loss_curve = []

    v_net = VNetD(data_dim=data.dim, depth=depth, latent_dim=FLAGS.latent_dim).to(device)
    posterior = PosteriorEncoder(data_dim=data.dim, latent_dim=FLAGS.latent_dim).to(device)
    optimizer = torch.optim.AdamW([
        {"params": v_net.parameters(), "lr": 1e-3},
        {"params": posterior.parameters(), "lr": 1e-3},  
    ])

    # Stampa del numero dei parametri dei due modelli decoder e posterior encoder
    model_size = 0
    model_vae_size = 0
    for param in v_net.parameters():
        model_size += param.data.nelement()
    print(f"Model params number: {model_size}")
    print("Model params: %.2f M" % (model_size / 1000 / 1000))

    for param in posterior.parameters():
        model_vae_size += param.data.nelement()
    print(f"Posterior Encoder params number: {model_vae_size}")
    print("Posterior Encoder params: %.2f M" % (model_vae_size / 1000 / 1000))

    A = torch.tril(torch.ones((depth, depth),device=device),diagonal=-1)
    with tqdm(initial=0,total=iterations) as pbar:
        for train_i in range(iterations):
            optimizer.zero_grad()
            indices = torch.randperm(len(data.pairs))[:checkpoint['batchsize']]
            batch = data.pairs[indices]
            x0 = batch[:, 0].detach().clone()   # N x d
            x1 = batch[:, 1].detach().clone()   # N x d

            print(f"x0.shape: {x0.shape}, x1.shape: {x1.shape}")

            x0 = torch.cat([x0[:,None,:], torch.randn((x0.shape[0],depth-1)+x0.shape[1:],device=device)], dim=1)   # Batch x Depth x dimdata

            t = torch.rand((x1.shape[0],depth)+(1,)*(x1.dim()-1), device=device)

            xt = (1-t)*x0 + t*(x1[:,None,...] - torch.einsum('ij,bj...->bi...', A, x0))
            # pred = v_net(xt, t.squeeze(list(range(2,t.dim())))) nel repo hanno due volte pred questa dovrebbe essere quella sostituita dalla seguente
            target = x1 - torch.sum(x0, dim=1) # N x d
            
            z, mu, log_var = posterior(
                x0=x0[:, 0, :],        # (B, d) 
                x1=x1,                 # (B, d)  
                xt=xt[:, 0, :],        # (B, d) 
                t=t.squeeze(-1)[:, 0], # (B,) 
            )

            pred = v_net(xt, t, z)
            recons_loss = torch.mean((target - pred) ** 2)
            kld_loss = torch.mean(
                                     -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1),
                                     dim=0
            )
            loss = recons_loss + FLAGS.beta * kld_loss
            loss.backward()

            optimizer.step()
            loss_curve.append(loss.item())
            recons_loss_curve.append(recons_loss.item())
            kld_loss_curve.append(kld_loss.item())

            pbar.set_description(
                f'rec_loss: {recons_loss.item():.4f} kld: {FLAGS.beta*kld_loss.item():.4f}',
                )

            pbar.update(1)

            if (train_i+1) % checkpoint['save_every_steps'] == 0 or train_i == (iterations-1):
                checkpoint['v_net_state_dict'] = v_net.state_dict()
                checkpoint['posterior_state_dict'] = posterior.state_dict()
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
                checkpoint['step'] = (train_i+1)
                torch.save(checkpoint, os.path.join(ckpt_dir, f"hrfvae_{train_i+1}_D{depth}_seed{seed}.pt"))

                
                v_net.eval()
                x0 = data.initial_model.sample([data.batchsize]).to(data.device).detach()
                x0 = torch.cat([x0[:,None,:], torch.randn((x0.shape[0],depth-1)+x0.shape[1:],device=device)], dim=1)
                t0 = torch.zeros(depth, device=device)
                z = torch.randn((x0.shape[0], FLAGS.latent_dim), device=device)  # Sample z from standard normal
                xt = sample_hierarchical(v_net, x0, t0, 0, depth, N_list, z=z)[1]
                if data.dim == 1:
                    distance = wasserstein_distance(data.x1[:, 0].cpu().numpy(), xt[:, 0].cpu().numpy())
                else:
                    distance = ot.sliced_wasserstein_distance(xt, data.x1, seed=1)
                print(f"{train_i+1} WD={distance} NFE={np.prod(N_list)} {N_list}")
                
                log_file = os.path.join(base_dir, "log_train_mode.csv")

                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)

                    writer.writerow([
                        FLAGS.mode,
                        train_i + 1,
                        FLAGS.data_type,
                        f"{distance:.6f}",
                        np.prod(N_list),
                        str(N_list),  # meglio json.dumps(N_list) se poi lo rileggi
                        FLAGS.beta,
                        FLAGS.latent_dim,
                        model_size,
                        model_vae_size
                    ])

                print(
                    f"step={train_i+1}  WD/SWD={distance:.6f}  "
                    f"NFE={np.prod(N_list)}  {N_list}"
                )

                plt.figure()
                if data.dim == 1:
                    bins = np.linspace(-2, 2, 201)
                    plt.hist(data.x1[:, 0].cpu().numpy(), bins=bins, density=True, alpha=0.8, histtype='step', linewidth=1, label=f'Target')
                    plt.hist(xt.cpu().numpy(), bins=bins, density=True, alpha=0.8, histtype='step', linewidth=1, label=f'Gen WD={distance:.3f}')
                else:
                    plt.scatter(data.x0[:5000, 0].cpu().numpy(), data.x0[:5000, 1].cpu().numpy(), label="Source", alpha=0.25, s=3)
                    plt.scatter(data.x1[:5000, 0].cpu().numpy(), data.x1[:5000, 1].cpu().numpy(), label="Target", alpha=0.25, s=3)
                    plt.scatter(xt[:5000,0].cpu().numpy(), xt[:5000,1].cpu().numpy(), label=f'Gen SWD={distance:.3f}', alpha=0.25, s=3)
                plt.xlabel("Data")
                plt.ylabel("Density")
                plt.legend()
                plt.title(f'Distribution NFE={np.prod(np.array(N_list))} {N_list}')
                plt.tight_layout()
                plt.savefig(os.path.join(img_dir, f"dist_hrf_progress.png"))
                plt.close()
                v_net.train()

                plt.figure()
                plt.plot(loss_curve)
                plt.title('Training Loss Curve')
                plt.savefig(os.path.join(img_dir, "total_loss.png"))
                plt.close()

                plt.figure()

                plt.plot(loss_curve, label="total")
                plt.plot(recons_loss_curve, label="recon")
                plt.plot(kld_loss_curve, label="kl")
                plt.legend()
                plt.title('Training Loss Curve')
                plt.tight_layout()
                plt.savefig(os.path.join(img_dir, "loss_total.png"))
                plt.close()

                plt.figure()
                plt.plot(recons_loss_curve)
                plt.title('Reconstruction Loss')
                plt.tight_layout()
                plt.savefig(os.path.join(img_dir, "loss_reconstruction.png"))
                plt.close()

                plt.figure()
                plt.plot(kld_loss_curve)
                plt.title('KL Divergence Loss')
                plt.tight_layout()
                plt.savefig(os.path.join(img_dir, "loss_kl.png"))
                plt.close()
                
    return v_net, posterior


def main(argv):
    seed = FLAGS.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    iterations = FLAGS.iter
    base_dir = os.path.join(FLAGS.base_dir, f'{FLAGS.data_type}_vae')
    if FLAGS.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{FLAGS.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    data = LowDimData(data_type=FLAGS.data_type, device=device)
    checkpoint = {
        'v_net_state_dict': None,
        'optimizer_state_dict': None,
        'save_every_steps': 5000,
        'step': 0,
        'batchsize': FLAGS.batchsize,
    }
    hrf_dir = os.path.join(base_dir, "hrfD_vae")
    img_dir = os.path.join(hrf_dir, "fig")
    dist_dir = os.path.join(img_dir, "dist")
    traj_dir = os.path.join(img_dir, "traj")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    N_list   = [int(x) for x in FLAGS.N_list]
    
    if FLAGS.mode == "train":
        # N_list = [100]
        # N_list = [10,10]
        # N_list = [2,5]
        # N_list = [1,2,5,10]
        # N_list = [1,2,5,5,10]
        # N_list = [1,1,1,1,1,1,2,5,5,10]
        v_net, _ = train_hrf(data, len(N_list), N_list, checkpoint, iterations, hrf_dir, seed, device, progress=True)

    elif FLAGS.mode == "eval":
        with torch.inference_mode():
            # N_list = [2,5]
            depth = len(N_list)
            step = FLAGS.eval_step

            v_net = VNetD(data_dim=data.dim, depth=depth, latent_dim=FLAGS.latent_dim).to(device)
            ckpt_name = f'hrfvae_{step}_D{depth}_seed{seed}'
            v_net = load_ckpt(hrf_dir, data.dim, v_net, ckpt=ckpt_name+'.pt')
            # non serve caricare il posterior nella face di sampling perché si campiona z dalla normale standard

            model_size = 0
            for param in v_net.parameters():
                model_size += param.data.nelement()
            print(f"Model params number: {model_size}")
            print("Model params: %.2f M" % (model_size / 1000 / 1000))

            
            x0_eval = data.initial_model.sample((data.batchsize,)).to(data.device).detach()
            x0 = torch.cat([x0_eval[:,None,:], torch.randn((x0_eval.shape[0],depth-1)+x0_eval.shape[1:],device=device)], dim=1)
            t0 = torch.zeros(depth, device=device)
            z = torch.randn((x0.shape[0], FLAGS.latent_dim), device=device)  # Sample z from standard normal    
            x0, xt, traj = sample_hierarchical(v_net, x0, t0, 0, depth, N_list, return_traj=True, z=z)
            if data.dim == 1:
                distance = wasserstein_distance(data.x1[:, 0].cpu().numpy(), xt[:,0].cpu().numpy())
            else:
                distance = ot.sliced_wasserstein_distance(data.x1, xt, seed=1)
            plot_traj(traj, distance, traj_dir, file_name=f"traj_{ckpt_name}_{N_list}.png", title=f'Trajectory with {N_list} Sampling Steps')

            log_file = os.path.join(hrf_dir, "log_eval_mode.csv")

            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)

                writer.writerow([
                    FLAGS.mode,
                    FLAGS.data_type,
                    f"{distance:.6f}",
                    np.prod(N_list),
                    str(N_list),  
                    FLAGS.beta,
                    FLAGS.latent_dim,
                    model_size,
                ])

                print(
                    f"WD/SWD={distance:.6f}  "
                    f"NFE={np.prod(N_list)}  {N_list}"
                )

            plt.figure()
            if data.dim == 1:
                bins = np.linspace(-2, 2, 201)
                plt.hist(data.x1[:, 0].cpu().numpy(), bins=bins, color="#ff7f0e", density=True, alpha=0.8, histtype='step', linewidth=1, label=f'Target')
                plt.hist(xt.cpu().numpy(), bins=bins, color="#2ca02c", density=True, alpha=0.8, histtype='step', linewidth=1, label=f'Gen WD={distance:.3f}')
            else:
                plt.scatter(data.x0[:5000, 0].cpu().numpy(), data.x0[:5000, 1].cpu().numpy(), c="#1f77b4", label="Source", alpha=0.25, s=3)
                plt.scatter(data.x1[:5000, 0].cpu().numpy(), data.x1[:5000, 1].cpu().numpy(), c="#ff7f0e", label="Target", alpha=0.25, s=3)
                plt.scatter(xt[:5000,0].cpu().numpy(), xt[:5000,1].cpu().numpy(), c="#2ca02c", label=f'Gen SWD={distance:.3f}', alpha=0.25, s=3)
            
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            if data.dim == 1:
                # plt.gca().get_xaxis().set_visible(False)
                plt.xticks(fontsize=16)
                plt.yticks(fontsize=16)
                plt.xlabel('Space', fontsize=16)
                plt.ylabel('Density', fontsize=16)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                # plt.gca().spines['bottom'].set_visible(False)
            else:
                plt.axis('off')
            
            plt.legend(fontsize=16, framealpha=0.5, loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(dist_dir, f"xdist_{ckpt_name}_{N_list}.png"), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_enum("data_type", None, ["1to2", "1to5", "2D1to6", "moon", "3to3", "scurve", "2to2", "tree"], "data type")
    flags.DEFINE_integer("batchsize", 5000, "batch size")
    flags.DEFINE_integer("iter", 50000, "training iterations")
    flags.DEFINE_integer("latent_dim", 4, "latent dimension for VAE")
    flags.DEFINE_integer("gpu", 0, "GPU number")
    flags.DEFINE_integer("seed", 0, "random seed")
    flags.DEFINE_list("N_list", ["10", "10"], "N_list per sampling gerarchico, es. 10,10")
    flags.DEFINE_float("beta", 1.0, "weight for KL divergence loss")
    flags.DEFINE_string("base_dir", "lowdim", "work dir")
    flags.DEFINE_enum("mode", None, ["train", "eval"], "running mode")
    flags.DEFINE_integer("eval_step", 50000, "checkpoint step to evaluate")
    

    app.run(main)