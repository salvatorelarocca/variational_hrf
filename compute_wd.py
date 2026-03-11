"""
Calcola W1 (1-Wasserstein) e SWD (Sliced Wasserstein Distance) al variare del
numero di NFE e produce un grafico NFE vs W1/SWD.

Uso:
    # Calcola W1 e SWD per NFE specifici
    python compute_wd.py --exp_name mnist_hrfvae --output_dir ./results --nfe_list 10,50,100,200 --num_samples 5000 --gpu 0

    # Rigenera le feature anche se già presenti
    python compute_wd.py --exp_name mnist_hrfvae --output_dir ./results --nfe_list 10,50,100 --force_regenerate=True --gpu 0

Note:
    - W1 è calcolata nello spazio delle feature InceptionV3 (2048-dim) tramite
      il problema di trasporto ottimale con POT (Python Optimal Transport).
    - SWD è calcolata nello stesso spazio proiettando su num_projections direzioni
      casuali e mediando le W1 monodimensionali risultanti.
    - Le feature vengono estratte una volta sola e salvate in .npy per riuso.

Output:
    {savedir}/wd/
        features_real.npy       ← feature reali (estratte una volta sola)
        features_nfe_10.npy     ← feature generate con NFE=10
        features_nfe_50.npy     ← ...
        wd_results.json         ← valori W1 e SWD per ogni NFE
        wd_vs_nfe.png           ← grafico NFE vs W1 e SWD
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import ot  # pip install POT
from absl import app, flags
from scipy.stats import wasserstein_distance
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from tqdm import trange, tqdm

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
flags.DEFINE_integer("num_samples", 5000, help="numero campioni per calcolo distanze")
flags.DEFINE_integer("batch_size", 128, help="batch size per la generazione e l'estrazione feature")
flags.DEFINE_bool("force_regenerate", False, help="rigenera le feature anche se già presenti")
flags.DEFINE_integer("num_projections", 1000, help="numero di proiezioni casuali per SWD")
flags.DEFINE_integer("swd_seed", 42, help="seed per le proiezioni casuali di SWD")


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
# Estrattore feature InceptionV3
# ─────────────────────────────────────────────

def build_feature_extractor(device):
    """
    Carica InceptionV3 pre-addestrato e rimuove l'ultimo strato di classificazione.
    Restituisce feature a 2048 dimensioni, le stesse usate per FID — questo
    permette di confrontare W1/SWD con FID sullo stesso spazio di feature.
    """
    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    # Rimuove il classificatore finale, mantiene il pooling a 2048-dim
    inception.fc = nn.Identity()
    inception.eval()
    return inception.to(device)


@torch.no_grad()
def extract_features(images_tensor, model, device, batch_size):
    """
    Estrae feature InceptionV3 da un tensore di immagini [N, 3, H, W] in [0, 1].
    Ridimensiona a 299x299 come richiesto da InceptionV3.
    Ritorna array numpy [N, 2048].
    """
    resize = transforms.Resize((299, 299), antialias=True)
    all_features = []
    for i in tqdm(range(0, len(images_tensor), batch_size), desc="    Estrazione feature", leave=False):
        batch = images_tensor[i:i + batch_size].to(device)
        batch = resize(batch)
        feats = model(batch)  # [B, 2048]
        all_features.append(feats.cpu().numpy())
    return np.concatenate(all_features, axis=0)


# ─────────────────────────────────────────────
# Raccolta immagini reali
# ─────────────────────────────────────────────

def get_real_images(dataset_name, num_samples):
    """
    Carica num_samples immagini reali come tensore [N, 3, H, W] in [0, 1].
    MNIST viene portato a 3 canali e 32x32 per coerenza con il training.
    """
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ])
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    elif dataset_name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    elif dataset_name == "imagenet32":
        raise NotImplementedError(
            "Per ImageNet32 implementa un loader personalizzato."
        )
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' non supportato.")

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    images = torch.stack([dataset[i][0] for i in indices])  # [N, C, H, W]
    return images


# ─────────────────────────────────────────────
# Generazione immagini sintetiche
# ─────────────────────────────────────────────

def generate_images(unet, cfg, data_shape, nfe, device, batch_size,
                    integration_method, num_samples):
    """
    Genera num_samples immagini con il modello per un dato NFE.
    Ritorna tensore [N, C, H, W] in [0, 1].

    Per HRF:
        NFE = N * M  (N cicli esterni, M cicli interni)
        N=2 fisso, M = NFE // N.
    Per RF baseline:
        NFE = numero di passi di integrazione.
    """
    hrf = cfg["model_type"] in ["unet_cat_hrf", "2unet_hrf"]

    if hrf:
        N = 2 if nfe >= 2 else 1
        M = max(1, nfe // N)
        actual_nfe = N * M
        print(f"    NFE={nfe} → N={N}, M={M} (NFE effettivo={actual_nfe})")
    else:
        print(f"    NFE={nfe}")

    all_images = []
    generated = 0
    unet.eval()

    with torch.no_grad():
        while generated < num_samples:
            current_batch = min(batch_size, num_samples - generated)
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

            imgs = imgs.clip(-1, 1) / 2 + 0.5  # rimappa in [0, 1]

            # MNIST: 1 canale → 3 canali per InceptionV3
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)

            all_images.append(imgs.cpu())
            generated += current_batch

    return torch.cat(all_images, dim=0)[:num_samples]


# ─────────────────────────────────────────────
# W1 — 1-Wasserstein nello spazio delle feature
# ─────────────────────────────────────────────

def compute_w1(feats_real, feats_gen):
    """
    Calcola la 1-Wasserstein distance tra due insiemi di feature tramite
    il problema di trasporto ottimale con costo L2 (POT).

    feats_real, feats_gen: array numpy [N, D]

    Nota: per N > 5000 il costo computazionale cresce quadraticamente —
    considera di ridurre num_samples o usare SWD come proxy più veloce.
    """
    n = len(feats_real)
    m = len(feats_gen)

    # Pesi uniformi per ogni distribuzione
    a = np.ones(n) / n
    b = np.ones(m) / m

    # Matrice dei costi L2
    M = ot.dist(feats_real, feats_gen, metric='euclidean')

    # Risolve il problema di trasporto ottimale (W1 = EMD con costo L2)
    w1 = ot.emd2(a, b, M)
    return float(w1)


# ─────────────────────────────────────────────
# SWD — Sliced Wasserstein Distance
# ─────────────────────────────────────────────

def compute_swd(feats_real, feats_gen, num_projections, seed):
    """
    Calcola la Sliced Wasserstein Distance tra due distribuzioni di feature.

    Algoritmo:
        1. Campiona num_projections direzioni casuali unitarie in R^D
        2. Proietta entrambe le distribuzioni su ciascuna direzione (prodotto scalare)
        3. Calcola la W1 monodimensionale (equivalente a sort + differenza) su ogni proiezione
        4. Ritorna la media delle W1 monodimensionali

    feats_real, feats_gen: array numpy [N, D]
    num_projections:       numero di proiezioni (più alto = stima più accurata)
    seed:                  seed per riproducibilità delle proiezioni
    """
    rng = np.random.default_rng(seed)
    D = feats_real.shape[1]

    # Genera direzioni casuali unitarie [num_projections, D]
    directions = rng.standard_normal((num_projections, D))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Proiezioni: [N, num_projections]
    proj_real = feats_real @ directions.T
    proj_gen  = feats_gen  @ directions.T

    # W1 monodimensionale = integrale del valore assoluto della differenza tra CDF
    # Per distribuzioni empiriche è equivalente a: mean(|sort(proj_real) - sort(proj_gen)|)
    proj_real_sorted = np.sort(proj_real, axis=0)
    proj_gen_sorted  = np.sort(proj_gen,  axis=0)

    # Se i due insiemi hanno dimensioni diverse, interpola il più piccolo
    if len(proj_real_sorted) != len(proj_gen_sorted):
        n_target = min(len(proj_real_sorted), len(proj_gen_sorted))
        idx_real = np.linspace(0, len(proj_real_sorted) - 1, n_target).astype(int)
        idx_gen  = np.linspace(0, len(proj_gen_sorted)  - 1, n_target).astype(int)
        proj_real_sorted = proj_real_sorted[idx_real]
        proj_gen_sorted  = proj_gen_sorted[idx_gen]

    swd = np.mean(np.abs(proj_real_sorted - proj_gen_sorted))
    return float(swd)


# ─────────────────────────────────────────────
# Grafico
# ─────────────────────────────────────────────

def plot_wd_vs_nfe(nfe_values, w1_values, swd_values, save_path, exp_name, integration_method):
    """Produce il grafico NFE vs W1 e SWD su due assi Y separati."""
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_w1  = 'steelblue'
    color_swd = 'darkorange'

    ax1.plot(nfe_values, w1_values,  marker='o', linewidth=2, markersize=6,
             color=color_w1,  label='W1')
    ax1.set_xlabel("NFE (Number of Function Evaluations)", fontsize=12)
    ax1.set_ylabel("W1 ↓", fontsize=12, color=color_w1)
    ax1.tick_params(axis='y', labelcolor=color_w1)

    # Annota W1
    for nfe, val in zip(nfe_values, w1_values):
        ax1.annotate(f"{val:.2f}", (nfe, val),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8, color=color_w1)

    ax2 = ax1.twinx()
    ax2.plot(nfe_values, swd_values, marker='s', linewidth=2, markersize=6,
             color=color_swd, linestyle='--', label='SWD')
    ax2.set_ylabel("SWD ↓", fontsize=12, color=color_swd)
    ax2.tick_params(axis='y', labelcolor=color_swd)

    # Annota SWD
    for nfe, val in zip(nfe_values, swd_values):
        ax2.annotate(f"{val:.4f}", (nfe, val),
                     textcoords="offset points", xytext=(0, -14),
                     ha='center', fontsize=8, color=color_swd)

    # Legenda unificata
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.set_xscale('log')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_title(f"NFE vs W1 & SWD — {exp_name} ({integration_method})", fontsize=13)

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

    nfe_list = sorted([int(x) for x in FLAGS.nfe_list])
    print(f"NFE da valutare: {nfe_list}")
    print(f"Campioni per NFE: {FLAGS.num_samples}")
    print(f"Proiezioni SWD:  {FLAGS.num_projections}")

    # Trova cartella esperimento e legge config
    savedir = find_savedir(FLAGS.output_dir, FLAGS.exp_name)
    with open(os.path.join(savedir, "config.json")) as f:
        cfg = json.load(f)
    print("\nConfigurazione:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    wd_dir = os.path.join(savedir, "wd")
    os.makedirs(wd_dir, exist_ok=True)

    # Data shape
    _, data_shape = get_datalooper(
        cfg["dataset"], batch_size=1, num_workers=0,
        train=False, imagenet_root=FLAGS.imagenet_root,
    )

    # Estrattore feature
    print("\n[1/4] Caricamento InceptionV3")
    feat_extractor = build_feature_extractor(device)

    # Feature reali (estratte una volta sola)
    real_feat_path = os.path.join(wd_dir, "features_real.npy")
    print("\n[2/4] Feature reali")
    if os.path.exists(real_feat_path) and not FLAGS.force_regenerate:
        print(f"  Già presenti, carico da {real_feat_path}")
        feats_real = np.load(real_feat_path)
    else:
        print(f"  Carico {FLAGS.num_samples} immagini reali...")
        real_images = get_real_images(cfg["dataset"], FLAGS.num_samples)
        print(f"  Estraggo feature...")
        feats_real = extract_features(real_images, feat_extractor, device, FLAGS.batch_size)
        np.save(real_feat_path, feats_real)
        print(f"  Feature reali salvate: {feats_real.shape}")

    # Carica modello
    print("\n[3/4] Caricamento modello")
    unet = get_model(
        cfg["dataset"], data_shape, cfg["channel_mult"], cfg["num_channel"], device,
        model_type=cfg["model_type"], latent_dim=cfg["latent_dim"],
        use_latent=cfg["variational"], use_scale_shift=cfg["use_scale_shift_norm"],
    )
    ckptdir = os.path.join(savedir, "ckpt")
    ckpt_list = [f for f in os.listdir(ckptdir) if f.endswith('.pt')]
    ckpt_file = sorted(ckpt_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    print(f"  Checkpoint: {ckpt_file}")
    ckpt = torch.load(os.path.join(ckptdir, ckpt_file), weights_only=True)
    load_model(unet, ckpt['ema_model'])
    unet.eval()

    # Loop su ogni NFE
    print("\n[4/4] Generazione, estrazione feature e calcolo distanze")
    results = {}

    for nfe in nfe_list:
        print(f"\n── NFE = {nfe} ──")
        gen_feat_path = os.path.join(wd_dir, f"features_nfe_{nfe}.npy")

        # Feature generate
        if os.path.exists(gen_feat_path) and not FLAGS.force_regenerate:
            print(f"  Feature già presenti, carico da {gen_feat_path}")
            feats_gen = np.load(gen_feat_path)
        else:
            print(f"  Genero {FLAGS.num_samples} immagini...")
            gen_images = generate_images(
                unet, cfg, data_shape, nfe,
                device, FLAGS.batch_size, FLAGS.integration_method, FLAGS.num_samples,
            )
            print(f"  Estraggo feature...")
            feats_gen = extract_features(gen_images, feat_extractor, device, FLAGS.batch_size)
            np.save(gen_feat_path, feats_gen)
            print(f"  Feature generate salvate: {feats_gen.shape}")

        # Calcola W1
        print(f"  Calcolo W1...")
        w1 = compute_w1(feats_real, feats_gen)
        print(f"  W1  (NFE={nfe}): {w1:.6f}")

        # Calcola SWD
        print(f"  Calcolo SWD ({FLAGS.num_projections} proiezioni)...")
        swd = compute_swd(feats_real, feats_gen, FLAGS.num_projections, FLAGS.swd_seed)
        print(f"  SWD (NFE={nfe}): {swd:.6f}")

        results[nfe] = {"w1": w1, "swd": swd}

    # Salva risultati JSON
    results_path = os.path.join(wd_dir, "wd_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "exp_name":           FLAGS.exp_name,
            "checkpoint":         ckpt_file,
            "num_samples":        FLAGS.num_samples,
            "num_projections":    FLAGS.num_projections,
            "swd_seed":           FLAGS.swd_seed,
            "integration_method": FLAGS.integration_method,
            "results":            results,
        }, f, indent=2)
    print(f"\nRisultati salvati in: {results_path}")

    # Tabella riassuntiva
    print("\n" + "="*45)
    print(f"{'NFE':>8}  {'W1':>14}  {'SWD':>14}")
    print("-"*45)
    for nfe in nfe_list:
        print(f"{nfe:>8}  {results[nfe]['w1']:>14.6f}  {results[nfe]['swd']:>14.6f}")
    print("="*45)

    # Grafico
    plot_path = os.path.join(wd_dir, "wd_vs_nfe.png")
    plot_wd_vs_nfe(
        nfe_values=nfe_list,
        w1_values=[results[n]["w1"]  for n in nfe_list],
        swd_values=[results[n]["swd"] for n in nfe_list],
        save_path=plot_path,
        exp_name=FLAGS.exp_name,
        integration_method=FLAGS.integration_method,
    )


if __name__ == "__main__":
    app.run(main)