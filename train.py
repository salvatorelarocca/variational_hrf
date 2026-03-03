import copy
import os
import json

import torch
from absl import app, flags
from torch.utils import tensorboard
from tqdm import trange
from vae.vae_model import BetaVAE

from dataset import get_datalooper
from choose_model import get_model
from utils import ema, generate_samples, load_model
from cfm import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "./", help="output directory")
flags.DEFINE_string("imagenet_root", "./", help="root directory for imagenet")
flags.DEFINE_string("exp_name", "base", help="experiment name")
flags.DEFINE_enum("dataset", "imagenet32", ["cifar10", "mnist", "imagenet32"], help="dataset name")
flags.DEFINE_integer("gpu", 0, help="GPU number")
flags.DEFINE_bool("use_scale_shift_norm", False, help="use scale shift norm")
flags.DEFINE_enum("integration_method", "euler", ["euler", "dopri5"], help="integration method for sampling")

# Variational HRF
flags.DEFINE_bool("variational", False, help="train variational hrf or deterministic hrf")
flags.DEFINE_integer("latent_dim", 128, help="dimension of the latent space for the VAE in variational HRF")
flags.DEFINE_float("beta", 1.0, help="valore massimo del peso KL (raggiunto dopo l'annealing)")
flags.DEFINE_float("kl_warmup_frac", 0.3, help="frazione dell'orizzonte totale su cui beta cresce da 0 a beta_max")
flags.DEFINE_float("free_bits", 1.0, help="soglia KL minima per dimensione in nats (0.0 per disabilitare)")

# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")
flags.DEFINE_list("channel_mult", [1, 2, 2, 2], help="channel_mult of UNet")
flags.DEFINE_enum("model_type", "baseline", ["baseline", "unet_cat_hrf", "2unet_hrf"], help="architecture Unet to use")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 400_001, help="total training steps")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup steps")
flags.DEFINE_integer("batch_size", 128, help="batch size")
flags.DEFINE_integer("ot_bs", 128, help="optimal transport batch size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("continue_train", False, help="continue training from last checkpoint")

# Evaluation
flags.DEFINE_integer("save_step", 20000, help="frequency of saving checkpoints, 0 to disable")
flags.DEFINE_integer("tb_step", 50, help="frequency of saving to tensorboard")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def _kl_divergence(mu, log_var):
    """Calcola KL per ogni elemento del batch e ogni dimensione latente.
    Ritorna [B, latent_dim] dopo viene data a free_bits per evitare il collapse parziale.
    log_var = log(sigma^2)
    """
    return 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1)


def get_beta(step, total_end_step, beta_max, warmup_frac):
    """KL annealing: beta cresce linearmente da 0 a beta_max.

    Parametri:
        step:           step corrente assoluto
        total_end_step: step finale assoluto = cur_step + FLAGS.total_steps
        beta_max:       valore massimo di beta (FLAGS.beta)
        warmup_frac:    frazione dell'orizzonte totale dedicata all'annealing

    Usando l'orizzonte assoluto il comportamento è corretto sia per
    training da zero che per ripresa da checkpoint:
    - da zero:        step parte da 0, beta sale da 0 a beta_max
    - da checkpoint:  step parte già alto, beta è già a beta_max e rimane lì
    """
    warmup_steps = int(total_end_step * warmup_frac)
    if warmup_steps == 0:
        return beta_max
    return beta_max * min(step / warmup_steps, 1.0)


def kl_with_free_bits(kl_per_dim, free_bits):
    """Free bits applicato per dimensione latente, poi mediato su batch e dimensioni.

    Parametri:
        kl_per_dim: [B, latent_dim] — KL per ogni elemento del batch e ogni dimensione
        free_bits:  soglia minima per dimensione in nats

    Il clamp viene applicato PRIMA della media così ogni dimensione deve
    contribuire almeno free_bits, indipendentemente dalle altre.
    Questo impedisce che alcune dimensioni collassino a zero mentre
    la media rimane alta grazie ad altre dimensioni attive.
    Se free_bits=0.0 si comporta come il KL standard.
    """
    if free_bits <= 0.0:
        return kl_per_dim.mean()
    return torch.clamp(kl_per_dim, min=free_bits).mean()


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step, tb_step:",
        FLAGS.lr, FLAGS.total_steps, FLAGS.ema_decay, FLAGS.save_step, FLAGS.tb_step,
    )

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if FLAGS.gpu is not None and FLAGS.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{FLAGS.gpu}")
    else:
        device = torch.device("cpu")

    savedir = os.path.join(FLAGS.output_dir, f"results_{FLAGS.dataset}", f"{FLAGS.exp_name}")
    os.makedirs(savedir, exist_ok=True)
    ckptdir = os.path.join(savedir, "ckpt")
    os.makedirs(ckptdir, exist_ok=True)
    imgdir = os.path.join(savedir, "img_train")
    os.makedirs(imgdir, exist_ok=True)
    writer = tensorboard.SummaryWriter(savedir)

    # Salva la configurazione strutturale una volta sola prima del loop.
    # I parametri architetturali non cambiano durante il training.
    model_config = {
        "dataset":              FLAGS.dataset,
        "model_type":           FLAGS.model_type,
        "num_channel":          FLAGS.num_channel,
        "channel_mult":         [int(x) for x in FLAGS.channel_mult],
        "latent_dim":           FLAGS.latent_dim,
        "variational":          FLAGS.variational,
        "use_scale_shift_norm": FLAGS.use_scale_shift_norm,
        "beta":                 FLAGS.beta,
        "kl_warmup_frac":       FLAGS.kl_warmup_frac,
        "free_bits":            FLAGS.free_bits,
    }
    config_path = os.path.join(savedir, "config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Configurazione salvata in: {config_path}")

    datalooper, data_shape = get_datalooper(
        FLAGS.dataset,
        FLAGS.batch_size,
        FLAGS.num_workers,
        train=True,
        imagenet_root=FLAGS.imagenet_root,
    )

    unet = get_model(
        FLAGS.dataset,
        data_shape,
        FLAGS.channel_mult,
        FLAGS.num_channel,
        device,
        model_type=FLAGS.model_type,
        latent_dim=FLAGS.latent_dim,
        use_scale_shift=FLAGS.use_scale_shift_norm,
        use_latent=FLAGS.variational,
    )

    if FLAGS.variational:
        vae = BetaVAE(in_channels=data_shape[0], latent_dim=FLAGS.latent_dim).to(device)

    hrf = FLAGS.model_type in ["unet_cat_hrf", "2unet_hrf"]

    model_size = sum(p.data.nelement() for p in unet.parameters())
    print(f"Model params: {model_size} ({model_size/1e6:.2f} M)")

    ema_model = copy.deepcopy(unet)

    if FLAGS.variational:
        vae_size = sum(p.data.nelement() for p in vae.parameters())
        print(f"VAE params: {vae_size} ({vae_size/1e6:.2f} M)")
        optim = torch.optim.Adam(list(unet.parameters()) + list(vae.parameters()), lr=FLAGS.lr)
    else:
        optim = torch.optim.Adam(unet.parameters(), lr=FLAGS.lr)

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    cur_step = 0
    ckpt_list = os.listdir(ckptdir)
    if FLAGS.continue_train and len(ckpt_list) > 0:
        ckpt_file = sorted(ckpt_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        print(f"Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(os.path.join(ckptdir, ckpt_file), weights_only=True)
        load_model(unet, ckpt['model'])
        load_model(ema_model, ckpt['ema_model'])
        if FLAGS.variational:
            load_model(vae, ckpt['vae'])
        optim.load_state_dict(ckpt['optim'])
        sched.load_state_dict(ckpt['sched'])
        cur_step = ckpt['step']

    # Orizzonte assoluto finale: usato da get_beta per calcolare il warmup
    # in modo corretto sia per training da zero che per ripresa da checkpoint.
    total_end_step = cur_step + FLAGS.total_steps

    # Estrae le flag usate nel loop in variabili locali per leggibilità
    variational    = FLAGS.variational
    beta_max       = FLAGS.beta
    kl_warmup_frac = FLAGS.kl_warmup_frac
    free_bits      = FLAGS.free_bits
    grad_clip      = FLAGS.grad_clip
    save_step      = FLAGS.save_step
    tb_step        = FLAGS.tb_step

    FM = ConditionalFlowMatcher(sigma=0.0)

    with trange(cur_step, total_end_step, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, target = FM.sample_location_and_conditional_flow(x0, x1)

            if hrf:
                v0 = torch.randn_like(target)
                tau, vtau, target = FM.sample_location_and_conditional_flow(v0, target)
                if variational:
                    z, mu, log_var = vae(v0, target, vtau, tau)
                    pred = unet(tau, vtau, t, xt, z=z)
                else:
                    pred = unet(tau, vtau, t, xt)
            else:
                pred = unet(t, xt)

            if variational:
                recon_loss  = torch.mean((pred - target) ** 2)
                kl_per_dim  = _kl_divergence(mu, log_var)          # [B, latent_dim]
                kl_loss     = kl_per_dim.mean()                     # scalare per logging
                beta        = get_beta(step, total_end_step, beta_max, kl_warmup_frac)
                # free bits applicato per dimensione prima della media:
                # ogni dimensione latente deve contribuire almeno free_bits nats,
                # impedendo il collapse parziale anche quando la media è alta
                kl_weighted = kl_with_free_bits(kl_per_dim, free_bits)
                loss        = recon_loss + kl_weighted * beta
            else:
                loss = torch.mean((pred - target) ** 2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)
            optim.step()
            sched.step()
            ema(unet, ema_model, FLAGS.ema_decay)

            pbar.set_description(f"step {step}")
            if variational:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    recon=f"{recon_loss.item():.4f}",
                    kl=f"{kl_loss.item():.4f}",     # reale, non clamped
                    beta=f"{beta:.4f}",
                    mu=f"{mu.mean().item():.4f}",
                )
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if save_step > 0 and step % save_step == 0:
                generate_samples(unet, imgdir, step, (16, *data_shape), device,
                                 net_="normal", integration_method=FLAGS.integration_method,
                                 hrf=hrf, latent_dim=FLAGS.latent_dim)
                generate_samples(ema_model, imgdir, step, (16, *data_shape), device,
                                 net_="ema", integration_method=FLAGS.integration_method,
                                 hrf=hrf, latent_dim=FLAGS.latent_dim)
                ckpt_data = {
                    "model":     unet.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "sched":     sched.state_dict(),
                    "optim":     optim.state_dict(),
                    "step":      step,
                }
                if variational:
                    ckpt_data["vae"] = vae.state_dict()
                torch.save(
                    ckpt_data,
                    os.path.join(ckptdir, f"{FLAGS.exp_name}_{FLAGS.model_type}_{FLAGS.dataset}_weights_step_{step}.pt"),
                )

            if tb_step > 0 and step % tb_step == 0:
                writer.add_scalar("loss/total", loss, step)
                if variational:
                    writer.add_scalar("loss/recon",    recon_loss,      step)
                    writer.add_scalar("loss/kl",       kl_loss,         step)  # reale, non clamped
                    writer.add_scalar("loss/beta",     beta,            step)
                    writer.add_scalar("vae/mu",        mu.mean(),       step)
                    writer.add_scalar("vae/log_var",   log_var.mean(),  step)


if __name__ == "__main__":
    app.run(train)