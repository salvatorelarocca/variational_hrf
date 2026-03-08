"""
Calcola la FID al variare del numero di NFE e produce un grafico NFE vs FID.

Uso:
    # Calcola FID per NFE specifici
    python compute_fid.py --exp_name goku --output_dir ./test --nfe_list 10,50,100,200 --num_samples 10000 --gpu 0

    # Rigenera le immagini anche se già presenti
    python compute_fid.py --exp_name goku --output_dir ./test --nfe_list 10,50,100 --force_regenerate=True --gpu 0

Output:
    {savedir}/fid/
        nfe_10/         ← immagini generate con NFE=10
        nfe_50/         ← immagini generate con NFE=50
        ...
        real/           ← immagini reali (generate una volta sola)
        fid_results.json    ← valori FID per ogni NFE
        fid_vs_nfe.png      ← grafico NFE vs FID
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from absl import app, flags
from cleanfid import fid
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import trange

from choose_model import get_model
from dataset import get_datalooper
from utils import load_model, sample_hrf, sample_rf

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "./", help="output directory (stesso di train.py)")
flags.DEFINE_string("imagenet_root", "./", help="root directory per imagenet")
flags.DEFINE_string("exp_name", "exp", help="nome esperimento (stesso di train.py)")
flags.DEFINE_integer("gpu", 0, help="GPU number, -1 per CPU")
flags.DEFINE_enum("integration_method", "euler", ["euler", "dopri5"], help="metodo di integrazione")
flags.DEFINE_list("nfe_list", [10, 50, 100, 200, 500], help="lista di NFE da valutare")
flags.DEFINE_integer("num_samples", 10000, help="numero immagini per calcolo FID (minimo consigliato: 10000)")
flags.DEFINE_integer("batch_size", 128, help="batch size per la generazione")
flags.DEFINE_bool("force_regenerate", False, help="rigenera le immagini anche se già presenti")


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def find_savedir(output_dir, exp_name):
    """Trova la cartella dell'esperimento scansionando results_*/."""
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
        "Assicurati che --output_dir e --exp_name corrispondano a quelli usati in train.py."
    )


# ─────────────────────────────────────────────
# Immagini reali
# ─────────────────────────────────────────────

def save_real_images(dataset_name, real_img_dir, num_samples):
    """
    Salva le immagini reali del dataset in PNG per clean-fid.
    Viene eseguita una volta sola — se le immagini sono già presenti viene saltata.
    MNIST viene convertito a 3 canali perché InceptionV3 richiede RGB.
    """
    os.makedirs(real_img_dir, exist_ok=True)
    existing = [f for f in os.listdir(real_img_dir) if f.endswith(".png")]
    if len(existing) >= num_samples:
        print(f"  Immagini reali già presenti ({len(existing)}), salto.")
        return

    print(f"  Salvataggio {num_samples} immagini reali in {real_img_dir}...")

    if dataset_name == "mnist":
        # Resize a 32x32 e converti a 3 canali per InceptionV3
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    elif dataset_name == "imagenet32":
        raise NotImplementedError(
            "Per ImageNet32 prepara le immagini reali manualmente nella cartella fid/real/."
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} non supportato.")

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    for i, idx in enumerate(indices):
        img, _ = dataset[idx]
        save_image(img / 2 + 0.5, os.path.join(real_img_dir, f"real_{i:06d}.png"))

    print(f"  Salvate {len(indices)} immagini reali.")


# ─────────────────────────────────────────────
# Generazione immagini
# ─────────────────────────────────────────────

def generate_images_for_nfe(unet, cfg, data_shape, gen_img_dir, nfe,
                             device, batch_size, integration_method, num_samples):
    """
    Genera num_samples immagini con un dato NFE e le salva in gen_img_dir.
    Se le immagini sono già presenti e force_regenerate=False, viene saltata.

    Per HRF:
        NFE = N * M  (cicli esterni * cicli interni)
        Usiamo N=2 fisso e M = NFE // N per mantenere lo stesso NFE nominale.
        Se NFE < 2 usiamo N=1, M=NFE.

    Per RF baseline:
        NFE corrisponde direttamente al numero di passi di integrazione.
    """
    os.makedirs(gen_img_dir, exist_ok=True)
    existing = [f for f in os.listdir(gen_img_dir) if f.endswith(".png")]
    if len(existing) >= num_samples and not FLAGS.force_regenerate:
        print(f"  NFE={nfe}: immagini già presenti ({len(existing)}), salto.")
        return

    hrf = cfg["model_type"] in ["unet_cat_hrf", "2unet_hrf"]
    num_batches = (num_samples + batch_size - 1) // batch_size
    generated_count = 0

    # Calcola N e M per HRF
    if hrf:
        N = 2 if nfe >= 2 else 1
        M = max(1, nfe // N)
        actual_nfe = N * M
        print(f"  NFE={nfe} → N={N}, M={M} (NFE effettivo={actual_nfe})")
    else:
        print(f"  NFE={nfe}")

    unet.eval()
    with torch.no_grad():
        for _ in trange(num_batches, desc=f"  Generazione NFE={nfe}", leave=False):
            current_batch = min(batch_size, num_samples - generated_count)
            if current_batch <= 0:
                break
            batch_shape = (current_batch, *data_shape)

            if hrf:
                imgs, _ = sample_hrf(
                    unet, batch_shape,
                    N=N, M=M,
                    device=device,
                    integration_method=integration_method,
                    latent_dim=cfg["latent_dim"],
                )
            else:
                imgs, _ = sample_rf(
                    unet, batch_shape,
                    nfe=nfe,
                    device=device,
                    integration_method=integration_method,
                )

            imgs = imgs.clip(-1, 1) / 2 + 0.5  # rimappa in [0,1]

            # MNIST: 1 canale → 3 canali per InceptionV3
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)

            for img in imgs:
                save_image(img, os.path.join(gen_img_dir, f"gen_{generated_count:06d}.png"))
                generated_count += 1

    print(f"  NFE={nfe}: generate {generated_count} immagini.")


# ─────────────────────────────────────────────
# Grafico
# ─────────────────────────────────────────────

def plot_fid_vs_nfe(nfe_values, fid_values, save_path, exp_name, integration_method):
    """Produce il grafico NFE (ascisse) vs FID (ordinate)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(nfe_values, fid_values, marker='o', linewidth=2, markersize=6, color='steelblue')

    # Annota ogni punto con il valore FID
    for nfe, fid_val in zip(nfe_values, fid_values):
        ax.annotate(
            f"{fid_val:.1f}",
            (nfe, fid_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=9,
        )

    ax.set_xlabel("NFE (Number of Function Evaluations)", fontsize=12)
    ax.set_ylabel("FID ↓", fontsize=12)
    ax.set_title(f"NFE vs FID — {exp_name} ({integration_method})", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xscale('log')  # scala logaritmica sull'asse x per leggibilità

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Grafico salvato in: {save_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(argv):
    device = torch.device(f"cuda:{FLAGS.gpu}") if (
        FLAGS.gpu >= 0 and torch.cuda.is_available()
    ) else torch.device("cpu")
    print(f"Device: {device}")

    # Parsing nfe_list (absl restituisce liste di stringhe)
    nfe_list = sorted([int(x) for x in FLAGS.nfe_list])
    print(f"NFE da valutare: {nfe_list}")
    print(f"Immagini per NFE: {FLAGS.num_samples}")

    # Trova cartella esperimento e legge config
    savedir = find_savedir(FLAGS.output_dir, FLAGS.exp_name)
    with open(os.path.join(savedir, "config.json")) as f:
        cfg = json.load(f)
    print("\nConfigurazione:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    fid_dir      = os.path.join(savedir, "fid")
    real_img_dir = os.path.join(fid_dir, "real")
    os.makedirs(fid_dir, exist_ok=True)

    # Data shape
    _, data_shape = get_datalooper(
        cfg["dataset"], batch_size=1, num_workers=0,
        train=False, imagenet_root=FLAGS.imagenet_root,
    )

    # Salva immagini reali (una volta sola)
    print("\n[1/3] Immagini reali")
    save_real_images(cfg["dataset"], real_img_dir, FLAGS.num_samples)

    # Carica modello dal checkpoint più recente
    print("\n[2/3] Caricamento modello")
    unet = get_model(
        cfg["dataset"], data_shape, cfg["channel_mult"], cfg["num_channel"], device,
        model_type=cfg["model_type"], latent_dim=cfg["latent_dim"],
        use_latent=cfg["variational"], use_scale_shift=cfg["use_scale_shift_norm"],
    )
    ckptdir  = os.path.join(savedir, "ckpt")
    ckpt_file = sorted(
        os.listdir(ckptdir),
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )[-1]
    print(f"  Checkpoint: {ckpt_file}")
    ckpt = torch.load(os.path.join(ckptdir, ckpt_file), weights_only=True)
    load_model(unet, ckpt['ema_model'])
    unet.eval()

    # Loop su ogni NFE: genera immagini e calcola FID
    print("\n[3/3] Generazione e calcolo FID")
    results = {}  # {nfe: fid_score}

    for nfe in nfe_list:
        print(f"\n── NFE = {nfe} ──")
        gen_img_dir = os.path.join(fid_dir, f"nfe_{nfe}")

        # Genera immagini
        generate_images_for_nfe(
            unet, cfg, data_shape, gen_img_dir, nfe,
            device, FLAGS.batch_size, FLAGS.integration_method, FLAGS.num_samples,
        )

        # Calcola FID
        print(f"  Calcolo FID...")
        fid_score = fid.compute_fid(real_img_dir, gen_img_dir, device=device)
        results[nfe] = fid_score
        print(f"  FID (NFE={nfe}): {fid_score:.4f}")

    # Salva risultati JSON
    results_path = os.path.join(fid_dir, "fid_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "exp_name":           FLAGS.exp_name,
            "checkpoint":         ckpt_file,
            "num_samples":        FLAGS.num_samples,
            "integration_method": FLAGS.integration_method,
            "results":            results,
        }, f, indent=2)
    print(f"\nRisultati salvati in: {results_path}")

    # Stampa tabella riassuntiva
    print("\n" + "="*35)
    print(f"{'NFE':>8}  {'FID':>10}")
    print("-"*35)
    for nfe in nfe_list:
        print(f"{nfe:>8}  {results[nfe]:>10.4f}")
    print("="*35)

    # Grafico NFE vs FID
    plot_path = os.path.join(fid_dir, "fid_vs_nfe.png")
    plot_fid_vs_nfe(
        nfe_values=nfe_list,
        fid_values=[results[n] for n in nfe_list],
        save_path=plot_path,
        exp_name=FLAGS.exp_name,
        integration_method=FLAGS.integration_method,
    )


if __name__ == "__main__":
    app.run(main)