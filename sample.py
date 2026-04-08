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
flags.DEFINE_integer("batch_size", 4, help="batch size per la generazione (riduci in caso di OOM)")  
flags.DEFINE_integer("checkpoint_step", -1, help="step del checkpoint da caricare (-1 per usare l'ultimo disponibile)")
flags.DEFINE_integer("N", 2, help="Numero di valutazioni lungo N (solo per HRF)")
flags.DEFINE_integer("M", 100, help="Numero di valutazioni lungo M (solo per HRF o RF)")
flags.DEFINE_boolean("single_images", False, help="Se True, salva ogni immagine singolarmente invece di una griglia")


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


def find_ckpt_file(ckptdir, checkpoint_step):
    """Seleziona il checkpoint: quello con lo step indicato oppure l'ultimo disponibile."""
    ckpt_list = os.listdir(ckptdir)
    if not ckpt_list:
        raise FileNotFoundError(f"Nessun checkpoint trovato in '{ckptdir}'.")

    ckpt_list_sorted = sorted(
        ckpt_list,
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    if checkpoint_step >= 0:
        matches = [f for f in ckpt_list_sorted
                   if int(f.split('_')[-1].split('.')[0]) == checkpoint_step]
        if not matches:
            available = [int(f.split('_')[-1].split('.')[0]) for f in ckpt_list_sorted]
            raise FileNotFoundError(
                f"Nessun checkpoint trovato per step={checkpoint_step}.\n"
                f"Step disponibili: {available}"
            )
        return matches[0]
    else:
        return ckpt_list_sorted[-1]


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


    savedir = find_savedir(FLAGS.output_dir, FLAGS.exp_name)
    ckptdir = os.path.join(savedir, "ckpt")
    imgdir  = os.path.join(savedir, "img_eval")
    os.makedirs(imgdir, exist_ok=True)

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

    ckpt_file = find_ckpt_file(ckptdir, FLAGS.checkpoint_step)
    print(f"Caricamento checkpoint: {ckpt_file}")
    ckpt = torch.load(os.path.join(ckptdir, ckpt_file), weights_only=True)
    load_model(unet, ckpt['ema_model'])
    unet.eval()

    use_z     = cfg["variational"]
    latent_dim = cfg["latent_dim"]

    num_samples = FLAGS.num_samples
    batch_size  = FLAGS.batch_size
    all_imgs    = []
    nfe         = 0
    generated   = 0

    print(f"Generazione di {num_samples} immagini con batch_size={batch_size}...")

    with torch.no_grad():
        while generated < num_samples:
            current_bs   = min(batch_size, num_samples - generated)
            sample_shape = (current_bs, *data_shape)
            if hrf:
                imgs, nfe, _ = sample_hrf(
                    unet,
                    sample_shape,
                    N=FLAGS.N,
                    M=FLAGS.M,
                    device=device,
                    integration_method=FLAGS.integration_method,
                    latent_dim=latent_dim,
                    use_z=use_z,
                )
            else:
                imgs, nfe, _ = sample_rf(
                    unet,
                    sample_shape,
                    nfe=FLAGS.M,
                    device=device,
                    integration_method=FLAGS.integration_method,
                    use_z=use_z,
                    latent_dim=latent_dim,
                )
            all_imgs.append(imgs.cpu())
            generated += current_bs
            print(f"  Generati {generated}/{num_samples}")
            torch.cuda.empty_cache()  


    # singolo batch sampling per misurare il tempo
    with torch.no_grad():
        sample_shape = (batch_size, *data_shape)
        if hrf:
            _, nfe, interval = sample_hrf(
                unet,
                sample_shape,
                N=FLAGS.N,
                M=FLAGS.M,
                device=device,
                integration_method=FLAGS.integration_method,
                latent_dim=latent_dim,
                use_z=use_z,
            )
        else:
            _, nfe, interval = sample_rf(
                unet,
                sample_shape,
                nfe=FLAGS.M,
                device=device,
                integration_method=FLAGS.integration_method,
                use_z=use_z,
                latent_dim=latent_dim,
            )

    generated_img = torch.cat(all_imgs, dim=0)

    if hrf:
        file = f"hrf_{cfg['model_type']}_{FLAGS.integration_method}_nfe{FLAGS.N}x{FLAGS.M}.png"
    else:
        file = f"rf_{FLAGS.integration_method}_nfe{FLAGS.M}.png"

    out_path = os.path.join(imgdir, file)

    generated_img = generated_img.clip(-1, 1) / 2 + 0.5 

    if FLAGS.single_images:
        subdir_name = file.rsplit('.', 1)[0]

        imgdir_param = os.path.join(imgdir, subdir_name)
        os.makedirs(imgdir_param, exist_ok=True)
        for i, img in enumerate(generated_img):
            single_path = os.path.join(imgdir_param, f"{i:05d}.png")
            save_image(img, single_path)
        print(f"Salvate {num_samples} immagini singole in '{imgdir_param}' (NFE={nfe})")
    else:
        save_image(generated_img, out_path, nrow=4)
        print(f"Salvate {num_samples} immagini in griglia -> {out_path} (NFE={nfe})")


    with open(os.path.join(imgdir, "times.txt"), "a") as f:
                f.write(f"{file.rsplit('.', 1)[0]}\n")
                f.write(f"Checkpoint: {ckpt_file}\n")
                f.write(f"time batch: {interval:.4f} seconds\n")
                f.write(f"time per sample: {interval / batch_size:.4f} seconds\n")
                f.write(f"time per nfe: {interval / nfe:.6f} seconds\n")

    print(f"Tempi salvati in '{imgdir}/times.txt'")
                

if __name__ == "__main__":
    app.run(eval)