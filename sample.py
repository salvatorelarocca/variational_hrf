import json
import os

import numpy as np
import torch
from absl import app, flags
from torchvision.utils import save_image

from dataset import get_datalooper
from choose_model import get_model
from utils import sample_rf, sample_hrf, load_model

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "./", help="output directory (stesso usato in train.py)")
flags.DEFINE_string("imagenet_root", "./", help="root directory for imagenet")
flags.DEFINE_string("exp_name", "exp", help="nome esperimento (stesso usato in train.py)")
flags.DEFINE_integer("gpu", -1, help="GPU number, -1 per CPU")
flags.DEFINE_enum("integration_method", "euler", ["euler", "dopri5"], help="integration method")
flags.DEFINE_integer("num_samples", 16, help="numero di immagini da generare")


def find_savedir(output_dir, exp_name):
    """Trova la cartella dell'esperimento scansionando results_* senza richiedere --dataset."""
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"output_dir '{output_dir}' non esiste.")
    for entry in os.listdir(output_dir):
        if not entry.startswith("results_"):
            continue
        candidate = os.path.join(output_dir, entry, exp_name)
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    raise FileNotFoundError(
        f"Nessun config.json trovato per exp_name='{exp_name}' in '{output_dir}'.\n"
        f"Assicurati che --output_dir e --exp_name corrispondano a quelli usati in train.py."
    )


def eval(argv):
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{FLAGS.gpu}") if (
        FLAGS.gpu >= 0 and torch.cuda.is_available()
    ) else torch.device("cpu")

    # Trova automaticamente la cartella dell'esperimento — non serve --dataset
    savedir = find_savedir(FLAGS.output_dir, FLAGS.exp_name)
    ckptdir = os.path.join(savedir, "ckpt")
    imgdir  = os.path.join(savedir, "img_eval")
    os.makedirs(imgdir, exist_ok=True)

    # Legge la configurazione strutturale salvata una volta sola all'avvio del training
    with open(os.path.join(savedir, "config.json")) as f:
        cfg = json.load(f)
    print("Configurazione letta da config.json:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    _, data_shape = get_datalooper(
        cfg["dataset"],
        batch_size=1,
        num_workers=0,
        train=False,
        imagenet_root=FLAGS.imagenet_root,
    )

    hrf = cfg["model_type"] in ["unet_cat_hrf", "2unet_hrf"]

    # Ricostruisce il modello con la stessa architettura del training
    unet = get_model(
        cfg["dataset"],
        data_shape,
        cfg["channel_mult"],
        cfg["num_channel"],
        device,
        model_type=cfg["model_type"],
        latent_dim=cfg["latent_dim"],
        use_latent=cfg["variational"],
        use_scale_shift=cfg["use_scale_shift_norm"],
    )

    model_size = sum(p.data.nelement() for p in unet.parameters())
    print(f"Parametri modello: {model_size} ({model_size/1e6:.2f} M)")

    # Carica l'ultimo checkpoint disponibile
    ckpt_file = sorted(
        os.listdir(ckptdir),
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )[-1]
    print(f"Caricamento checkpoint: {ckpt_file}")
    ckpt = torch.load(os.path.join(ckptdir, ckpt_file), weights_only=True)
    load_model(unet, ckpt['ema_model'])
    unet.eval()

    sample_shape = (FLAGS.num_samples, *data_shape)

    with torch.no_grad():
        if hrf:
            print(f"Sampling HRF ({cfg['model_type']}), metodo: {FLAGS.integration_method}...")
            generated_img, nfe = sample_hrf(
                unet,
                sample_shape,
                N=2,
                M=100,
                device=device,
                integration_method=FLAGS.integration_method,
                latent_dim=cfg["latent_dim"],
            )
            file = f"hrf_{cfg['model_type']}_{FLAGS.integration_method}_nfe{nfe}.png"
        else:
            print(f"Sampling RF baseline, metodo: {FLAGS.integration_method}...")
            generated_img, nfe = sample_rf(
                unet,
                sample_shape,
                nfe=100,
                device=device,
                integration_method=FLAGS.integration_method,
            )
            file = f"rf_{FLAGS.integration_method}_nfe{nfe}.png"

    out_path = os.path.join(imgdir, file)
    save_image(generated_img.clip(-1, 1) / 2 + 0.5, out_path, nrow=4)
    print(f"Salvate {FLAGS.num_samples} immagini -> {out_path}  (NFE={nfe})")


if __name__ == "__main__":
    app.run(eval)
