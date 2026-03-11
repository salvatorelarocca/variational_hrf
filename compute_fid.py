"""
Calcola la FID al variare di N e M e produce un grafico NFE vs FID.

Per modelli HRF (unet_cat_hrf, 2unet_hrf):
    NFE = N * M
    Passa --N_list e --M_list con stessa cardinalità.
    Ogni coppia (N[i], M[i]) è un punto del grafico.

Per modelli baseline RF:
    NFE = N (passi di integrazione diretti)
    Passa solo --N_list. --M_list viene ignorato.

Esempi:
    # HRF
    python compute_fid.py --exp_name goku --output_dir ./test \
        --N_list 1,2,2,5 --M_list 10,25,50,20 \
        --num_samples 10000 --gpu 0

    # RF baseline (M_list non necessario)
    python compute_fid.py --exp_name goku --output_dir ./test \
        --N_list 10,50,100,200 \
        --num_samples 10000 --gpu 0

Output:
    {savedir}/fid/
        real/               ← immagini reali (generate una volta sola)
        N1_M10/             ← immagini generate con N=1, M=10  (NFE=10)
        N2_M25/             ← immagini generate con N=2, M=25  (NFE=50)
        ...
        fid_results.json    ← valori FID per ogni coppia (N, M)
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
flags.DEFINE_integer("num_samples", 10000, help="numero immagini per calcolo FID (minimo consigliato: 10000)")
flags.DEFINE_integer("batch_size", 128, help="batch size per la generazione")
flags.DEFINE_bool("force_regenerate", False, help="rigenera le immagini anche se già presenti")
flags.DEFINE_integer("num_workers", 0, help="worker per il DataLoader di clean-fid, 0 per disabilitare il multiprocessing")

# N_list: cicli esterni HRF oppure NFE diretti per RF baseline
flags.DEFINE_list("N_list", [1, 2, 5, 10], help=(
    "Lista di valori N. "
    "Per HRF: cicli esterni (NFE = N*M). "
    "Per RF baseline: NFE diretto."
))
# M_list: cicli interni HRF, opzionale — ignorato per RF baseline
flags.DEFINE_list("M_list", [], help=(
    "Lista di valori M (cicli interni HRF). "
    "Deve avere stessa cardinalità di N_list. "
    "Non necessario per RF baseline — viene ignorato."
))


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


def parse_nm_lists(N_list_raw, M_list_raw, hrf):
    """
    Valida e restituisce la lista di coppie (N, M) da valutare.

    Per HRF:
        - N_list e M_list devono avere stessa cardinalità
        - Se M_list è vuota usa M=10 come default per tutti gli N
    Per RF baseline:
        - M_list viene ignorato completamente
        - Restituisce coppie (N, None)
    """
    N_list = [int(x) for x in N_list_raw]

    if not hrf:
        # RF baseline: M non ha senso
        if M_list_raw:
            print("  Nota: --M_list ignorato per modello RF baseline.")
        return [(n, None) for n in N_list]

    # HRF: M_list necessaria
    if not M_list_raw:
        print("  Nota: --M_list non specificata per HRF, uso M=10 come default.")
        M_list = [10] * len(N_list)
    else:
        M_list = [int(x) for x in M_list_raw]
        if len(M_list) != len(N_list):
            raise ValueError(
                f"--N_list e --M_list devono avere stessa cardinalità. "
                f"Ricevuti: N_list={len(N_list)}, M_list={len(M_list)}"
            )

    return list(zip(N_list, M_list))


# ─────────────────────────────────────────────
# Immagini reali
# ─────────────────────────────────────────────

def save_real_images(dataset_name, real_img_dir, num_samples):
    """
    Salva le immagini reali del dataset in PNG per clean-fid.
    Eseguita una volta sola — se le immagini sono già presenti viene saltata.
    MNIST viene convertito a 3 canali perché InceptionV3 richiede RGB.
    """
    os.makedirs(real_img_dir, exist_ok=True)
    existing = [f for f in os.listdir(real_img_dir) if f.endswith(".png")]
    if len(existing) >= num_samples:
        print(f"  Immagini reali già presenti ({len(existing)}), salto.")
        return

    print(f"  Salvataggio {num_samples} immagini reali...")

    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.Resize(32),      # resize per InceptionV3
            transforms.Grayscale(3),    # 1 canale → 3 canali
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
            "Per ImageNet32 prepara le immagini reali manualmente in fid/real/."
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

def generate_images(unet, cfg, data_shape, gen_img_dir, N, M,
                    device, batch_size, integration_method, num_samples, hrf):
    """
    Genera num_samples immagini con i parametri N, M specificati.
    Se le immagini sono già presenti e force_regenerate=False viene saltata.

    HRF:      usa sample_hrf con N cicli esterni e M cicli interni
    Baseline: usa sample_rf con NFE = N passi di integrazione
    """
    os.makedirs(gen_img_dir, exist_ok=True)
    existing = [f for f in os.listdir(gen_img_dir) if f.endswith(".png")]
    label = f"N={N}, M={M}" if hrf else f"NFE={N}"

    if len(existing) >= num_samples and not FLAGS.force_regenerate:
        print(f"  {label}: immagini già presenti ({len(existing)}), salto.")
        return

    num_batches = (num_samples + batch_size - 1) // batch_size
    generated_count = 0

    unet.eval()
    with torch.no_grad():
        for _ in trange(num_batches, desc=f"  {label}", leave=False):
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
                # RF baseline: N è il numero di passi di integrazione (NFE)
                imgs, _ = sample_rf(
                    unet, batch_shape,
                    nfe=N,
                    device=device,
                    integration_method=integration_method,
                )

            imgs = imgs.clip(-1, 1) / 2 + 0.5

            # MNIST: 1 canale → 3 canali per InceptionV3
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1)

            for img in imgs:
                save_image(img, os.path.join(gen_img_dir, f"gen_{generated_count:06d}.png"))
                generated_count += 1

    print(f"  {label}: generate {generated_count} immagini.")


# ─────────────────────────────────────────────
# Grafico
# ─────────────────────────────────────────────

def plot_fid_vs_nfe(nfe_values, fid_values, labels, save_path, exp_name, integration_method):
    """
    Grafico NFE (ascisse, scala log) vs FID (ordinate).
    Ogni punto è annotato con la label (N=x, M=y) o NFE=x per il baseline.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(nfe_values, fid_values, marker='o', linewidth=2,
            markersize=7, color='steelblue')

    for nfe, fid_val, label in zip(nfe_values, fid_values, labels):
        ax.annotate(
            f"{label}\nFID={fid_val:.1f}",
            (nfe, fid_val),
            textcoords="offset points",
            xytext=(0, 12),
            ha='center',
            fontsize=8,
        )

    ax.set_xlabel("NFE (Number of Function Evaluations)", fontsize=12)
    ax.set_ylabel("FID ↓", fontsize=12)
    ax.set_title(f"NFE vs FID — {exp_name} ({integration_method})", fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xscale('log')

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

    # Trova cartella esperimento e legge config
    savedir = find_savedir(FLAGS.output_dir, FLAGS.exp_name)
    with open(os.path.join(savedir, "config.json")) as f:
        cfg = json.load(f)
    print("\nConfigurazione:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    hrf = cfg["model_type"] in ["unet_cat_hrf", "2unet_hrf"]

    # Valida e costruisce lista coppie (N, M)
    nm_pairs = parse_nm_lists(FLAGS.N_list, FLAGS.M_list, hrf)
    print(f"\nCoppie (N, M) da valutare: {nm_pairs}")
    print(f"Immagini per punto: {FLAGS.num_samples}")

    fid_dir      = os.path.join(savedir, "fid")
    real_img_dir = os.path.join(fid_dir, "real")
    os.makedirs(fid_dir, exist_ok=True)

    # Data shape
    _, data_shape = get_datalooper(
        cfg["dataset"], batch_size=1, num_workers=0,
        train=False, imagenet_root=FLAGS.imagenet_root,
    )

    # ── 1. Immagini reali (una volta sola) ──
    print("\n[1/3] Immagini reali")
    save_real_images(cfg["dataset"], real_img_dir, FLAGS.num_samples)

    # ── 2. Carica modello ──
    print("\n[2/3] Caricamento modello")
    unet = get_model(
        cfg["dataset"], data_shape, cfg["channel_mult"], cfg["num_channel"], device,
        model_type=cfg["model_type"], latent_dim=cfg["latent_dim"],
        use_latent=cfg["variational"], use_scale_shift=cfg["use_scale_shift_norm"],
    )
    ckptdir   = os.path.join(savedir, "ckpt")
    ckpt_file = sorted(
        os.listdir(ckptdir),
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )[-1]
    print(f"  Checkpoint: {ckpt_file}")
    ckpt = torch.load(os.path.join(ckptdir, ckpt_file), weights_only=True)
    load_model(unet, ckpt['ema_model'])
    unet.eval()

    # ── 3. Genera immagini e calcola FID per ogni coppia (N, M) ──
    print("\n[3/3] Generazione e calcolo FID")
    results  = []   # lista di dict {N, M, nfe, fid}
    nfe_list = []   # per il grafico
    fid_list = []
    label_list = []

    for N, M in nm_pairs:
        nfe   = N * M if hrf else N
        label = f"N={N},M={M}" if hrf else f"NFE={N}"
        print(f"\n── {label} (NFE={nfe}) ──")

        # Cartella dedicata per questa coppia
        folder_name = f"N{N}_M{M}" if hrf else f"NFE{N}"
        gen_img_dir = os.path.join(fid_dir, folder_name)

        generate_images(
            unet, cfg, data_shape, gen_img_dir, N, M,
            device, FLAGS.batch_size, FLAGS.integration_method,
            FLAGS.num_samples, hrf,
        )

        print(f"  Calcolo FID...")
        fid_score = fid.compute_fid(real_img_dir, gen_img_dir, device=device, num_workers=FLAGS.num_workers)
        print(f"  FID = {fid_score:.4f}")

        results.append({"N": N, "M": M, "nfe": nfe, "fid": fid_score})
        nfe_list.append(nfe)
        fid_list.append(fid_score)
        label_list.append(label)

    # Salva risultati JSON
    results_path = os.path.join(fid_dir, "fid_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "exp_name":           FLAGS.exp_name,
            "checkpoint":         ckpt_file,
            "num_samples":        FLAGS.num_samples,
            "integration_method": FLAGS.integration_method,
            "hrf":                hrf,
            "results":            results,
        }, f, indent=2)
    print(f"\nRisultati salvati in: {results_path}")

    # Tabella riassuntiva
    print("\n" + "="*45)
    if hrf:
        print(f"{'N':>5}  {'M':>5}  {'NFE':>6}  {'FID':>10}")
    else:
        print(f"{'NFE':>6}  {'FID':>10}")
    print("-"*45)
    for r in results:
        if hrf:
            print(f"{r['N']:>5}  {r['M']:>5}  {r['nfe']:>6}  {r['fid']:>10.4f}")
        else:
            print(f"{r['nfe']:>6}  {r['fid']:>10.4f}")
    print("="*45)

    # Grafico
    plot_path = os.path.join(fid_dir, "fid_vs_nfe.png")
    plot_fid_vs_nfe(
        nfe_values=nfe_list,
        fid_values=fid_list,
        labels=label_list,
        save_path=plot_path,
        exp_name=FLAGS.exp_name,
        integration_method=FLAGS.integration_method,
    )


if __name__ == "__main__":
    app.run(main)