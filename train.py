import copy
import math
import os
import json
from datetime import datetime

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
flags.DEFINE_bool("generate_samples", True, help="whether to generate samples during training")

# Variational
flags.DEFINE_bool("variational", False, help="train variational hrf or deterministic hrf")
flags.DEFINE_integer("latent_dim", 32, help="dimension of the latent space for the VAE in variational HRF")
flags.DEFINE_float("beta", 1.0, help="valore massimo del peso KL (raggiunto dopo l'annealing)")
flags.DEFINE_list("hidden_vae", [32, 64, 128], help="hidden dimensions for the VAE encoder CNN")
flags.DEFINE_float("kl_warmup_frac", 0.3, help="frazione dell'orizzonte totale su cui beta cresce da 0 a beta_max")

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
flags.DEFINE_integer("val_batches", 50, help="numero di batch per la validation loss (50 * batch_size campioni)")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def _kl_divergence(mu, log_var):
    """KL per ogni elemento del batch e ogni dimensione latente. [B, latent_dim]"""
    return 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1)


def evaluate(unet, vae, val_datalooper, FM, device, beta, variational, hrf, n_val_batches):
    """
    Calcola la validation loss su n_val_batches batch senza aggiornare i pesi.
    Usa .eval() per disabilitare dropout e BatchNorm in modalita' inferenza.
    Ritorna dict con loss, recon, kl (kl=0 se non variational).
    """
    unet.eval()
    if variational and vae is not None:
        vae.eval()

    total_loss  = 0.0
    total_recon = 0.0
    total_kl    = 0.0

    with torch.no_grad():
        for _ in range(n_val_batches):
            x1 = next(val_datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, target = FM.sample_location_and_conditional_flow(x0, x1)

            if hrf:
                v0 = torch.randn_like(target)
                tau, vtau, target = FM.sample_location_and_conditional_flow(v0, target)
                if variational:
                    '''il vae prende in input partenza, arrivo, valori intermeti e tempo della velocità'''
                    z, mu, log_var = vae(v0, vtau, tau) #v0, v1, vtau, tau z = q(z | start, target, state_t, time)
                    '''quindi unet deve innestare z solo nei resblock della velocità?'''
                    pred = unet(tau, vtau, t, xt, z=z)
                else:
                    pred = unet(tau, vtau, t, xt)
            else:
                if variational:
                    z, mu, log_var = vae(x0, xt, t) # uguale per hrf ma i parametri sono quelli spaziali
                    pred = unet(t, xt, z=z)
                else:
                    pred = unet(t, xt)

            if variational:
                recon_loss = torch.mean((pred - target) ** 2)
                kl_b_dim    = _kl_divergence(mu, log_var)
                kl_per_sample = kl_b_dim.sum(dim=1)        # [B]
                kl_loss    = kl_per_sample.mean()
                loss       = recon_loss + kl_loss * beta
                total_recon += recon_loss.item()
                total_kl    += kl_loss.item()
            else:
                loss = torch.mean((pred - target) ** 2)

            total_loss += loss.item()

    # ripristina modalita' training
    unet.train()
    if variational and vae is not None:
        vae.train()

    return {
        "loss":  total_loss  / n_val_batches,
        "recon": total_recon / n_val_batches,
        "kl":    total_kl    / n_val_batches,
    }


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_config = {
        "dataset":              FLAGS.dataset,
        "model_type":           FLAGS.model_type,
        "num_channel":          FLAGS.num_channel,
        "channel_mult":         [int(x) for x in FLAGS.channel_mult],
        "hidden_vae":           [int(x) for x in FLAGS.hidden_vae],
        "latent_dim":           FLAGS.latent_dim,
        "variational":          FLAGS.variational,
        "use_scale_shift_norm": FLAGS.use_scale_shift_norm,
        "beta":                 FLAGS.beta,
        "lr":                   FLAGS.lr,
        "batch_size":           FLAGS.batch_size,
        "total_steps":          FLAGS.total_steps,
        "ema_decay":            FLAGS.ema_decay,
    }

    config_path = os.path.join(savedir, "config.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Configurazione salvata in: {config_path}")

    datalooper, data_shape = get_datalooper(
        FLAGS.dataset, FLAGS.batch_size, FLAGS.num_workers,
        train=True, imagenet_root=FLAGS.imagenet_root,
    )

    val_datalooper, _ = get_datalooper(
        FLAGS.dataset, FLAGS.batch_size, FLAGS.num_workers,
        train=False, imagenet_root=FLAGS.imagenet_root,
    )

    unet = get_model(
        FLAGS.dataset, data_shape, FLAGS.channel_mult, FLAGS.num_channel, device,
        model_type=FLAGS.model_type, latent_dim=FLAGS.latent_dim,
        use_scale_shift=FLAGS.use_scale_shift_norm, use_latent=FLAGS.variational,
    )
    
    hrf = FLAGS.model_type in ["unet_cat_hrf", "2unet_hrf"]

    vae = None
    if FLAGS.variational:
        vae = BetaVAE(
            in_channels=data_shape[0],
            latent_dim=FLAGS.latent_dim,
            hidden_dims=FLAGS.hidden_vae,
        ).to(device)



    model_size = sum(p.data.nelement() for p in unet.parameters())
    print(f"Model params: {model_size} ({model_size/1e6:.2f} M)")
    model_config["model_params"] = model_size

    ema_model = copy.deepcopy(unet)

    if FLAGS.variational:
        vae_size = sum(p.data.nelement() for p in vae.parameters())
        print(f"VAE params: {vae_size} ({vae_size/1e6:.2f} M)")
        model_config["vae_params"] = vae_size
        optim = torch.optim.Adam(list(unet.parameters()) + list(vae.parameters()), lr=FLAGS.lr)
    else:
        optim = torch.optim.Adam(unet.parameters(), lr=FLAGS.lr)

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

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
        cur_step = ckpt['step'] + 1

    writer = tensorboard.SummaryWriter(
        savedir,
        purge_step=cur_step,
        filename_suffix=f".{timestamp}"
    )

    total_end_step = FLAGS.total_steps
    variational    = FLAGS.variational
    grad_clip      = FLAGS.grad_clip
    save_step      = FLAGS.save_step
    tb_step        = FLAGS.tb_step
    beta           = FLAGS.beta
    val_batches    = FLAGS.val_batches
    kl_warmup_frac = FLAGS.kl_warmup_frac
   
    FM = ConditionalFlowMatcher(sigma=0.0)

    with trange(cur_step, total_end_step, dynamic_ncols=True, initial=cur_step, total=total_end_step) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, target = FM.sample_location_and_conditional_flow(x0, x1)

            if hrf:
                v0 = torch.randn_like(target)
                tau, vtau, target = FM.sample_location_and_conditional_flow(v0, target)
                if variational:
                    #HRF+VAE
                    z, mu, log_var = vae(v0, vtau, tau, target) #non passo il target
                    pred = unet(tau, vtau, t, xt, z=z)
                else:
                    #HRF
                    pred = unet(tau, vtau, t, xt)
            else:
                if variational:
                    #RF+VAE
                    z, mu, log_var = vae(x0, xt, t, ta) 
                    pred = unet(t, xt, z=z)
                else:
                    #RF
                    pred = unet(t, xt)

            if variational:
                recon_loss = torch.mean((pred - target) ** 2)
                kl_b_dim   = _kl_divergence(mu, log_var)   # [B, latent_dim]
                kl_per_sample = kl_b_dim.sum(dim=1)        # [B]
                kl_loss    = kl_per_sample.mean()
                annealing_factor = min(1.0, (step + 1) / (total_end_step * kl_warmup_frac))
                loss       = recon_loss + kl_loss * beta * annealing_factor
            else:
                loss = torch.mean((pred - target) ** 2)

            loss.backward()
            if variational:
                torch.nn.utils.clip_grad_norm_(
                    list(unet.parameters()) + list(vae.parameters()),
                    grad_clip,
                )
            else:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)
            optim.step()
            sched.step()
            ema(unet, ema_model, FLAGS.ema_decay)

            pbar.set_description(f"step {step}")
            if variational:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    recon=f"{recon_loss.item():.4f}",
                    kl=f"{kl_loss.item():.4f}",
                    mu=f"{mu.mean().item():.4f}",
                    log_var=f"{log_var.mean().item():.4f}",
                )
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # --Salvataggio checkpoint + validation loss--
            if step % save_step == 0:
                # validation loss - calcolata prima di salvare il checkpoint
                val = evaluate(
                    unet, vae, val_datalooper, FM,
                    device, beta, variational, hrf, val_batches,
                )
                writer.add_scalar("val/loss",  val["loss"],  step)
                if variational:
                    writer.add_scalar("val/recon", val["recon"], step)
                    writer.add_scalar("val/kl", val["kl"], step)
                print(
                    f"\n  [step {step}] val_loss={val['loss']:.4f}"
                    + (f"  val_recon={val['recon']:.4f}  val_kl={val['kl']:.4f}" if variational else "")
                )

                if FLAGS.generate_samples:
                    generate_samples(unet, imgdir, step, (16, *data_shape), device,
                                     net_="normal", integration_method=FLAGS.integration_method,
                                     hrf=hrf, latent_dim=FLAGS.latent_dim, use_z=variational)
                    generate_samples(ema_model, imgdir, step, (16, *data_shape), device,
                                     net_="ema", integration_method=FLAGS.integration_method,
                                     hrf=hrf, latent_dim=FLAGS.latent_dim, use_z=variational)

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

            #Tensorboard scalars
            if step % tb_step == 0:
                writer.add_scalar("loss/total", loss, step)
                if variational:
                    writer.add_scalar("loss/recon",  recon_loss,                      step)
                    writer.add_scalar("loss/kl",     kl_loss,                         step)
                    writer.add_scalar("loss/beta",   beta*annealing_factor,           step)
                    writer.add_scalar("vae/mu",      mu.mean(),                       step)
                    writer.add_scalar("vae/log_var", log_var.mean(),                  step)


if __name__ == "__main__":
    app.run(train)