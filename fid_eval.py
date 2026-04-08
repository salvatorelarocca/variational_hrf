import os
import json
from datetime import datetime
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, MNIST
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from absl import app, flags

FLAGS = flags.FLAGS


flags.DEFINE_string("generated_dir", None, "Cartella con le immagini generate (PNG/JPG).")
flags.DEFINE_string("real_dir", None, "Cartella con le immagini reali su disco.")
flags.DEFINE_string("dataset", None, "Dataset built-in da usare come riferimento reale (cifar10, cifar100, mnist, stl10).")
flags.DEFINE_string("data_root", "./data", "Root directory per il download dei dataset built-in.")
flags.DEFINE_integer("gpu", 0, "Indice GPU (-1 per CPU).")
flags.DEFINE_integer("batch_size", 64, "Batch size per il caricamento delle immagini.")
flags.DEFINE_integer("num_workers", 4, "Numero di worker per il DataLoader.")
flags.DEFINE_integer("img_size", None, "Ridimensiona tutte le immagini a img_size x img_size. Se non specificato usa la dimensione originale.")
flags.DEFINE_integer("feature_dim", 2048, "Dimensione delle feature di Inception (64, 192, 768, 2048).")

flags.mark_flag_as_required("generated_dir")

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

BUILTIN_DATASETS = {
    "cifar10":  (CIFAR10,  32),
    "mnist":    (MNIST,    32),
}


class FolderDataset(Dataset):
    """Carica tutte le immagini da una cartella (non ricorsivo)."""

    def __init__(self, folder: str, transform=None):
        self.paths = sorted(
            p for p in Path(folder).iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"Nessuna immagine trovata in '{folder}'")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


'''prepara le immagini per FID: uint8 in [0, 255], shape (B, 3, H, W)'''
def make_uint8_transform(img_size=None):
    """
    FrechetInceptionDistance di torchmetrics vuole tensori uint8
    con valori in [0, 255] e shape (B, 3, H, W).
    """
    ops = []
    if img_size is not None:
        ops.append(T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR))
    ops += [
        T.ToTensor(),
        T.Lambda(lambda x: (x * 255).to(torch.uint8)),
    ]
    return T.Compose(ops)

'''funzione per il caricamento del dataset reale, se si usa un dataset built-in invece di una cartella, nel nostro caso MNIST o CIFAR10'''
def get_real_loader_from_dataset(dataset_name, data_root, n_samples, batch_size, num_workers, img_size=None):
    """Restituisce un DataLoader per il test-set del dataset indicato."""
    name = dataset_name.lower()
    if name not in BUILTIN_DATASETS:
        raise ValueError(
            f"Dataset '{name}' non supportato. "
            f"Scegli tra: {list(BUILTIN_DATASETS.keys())}"
        )

    cls, default_size = BUILTIN_DATASETS[name]
    size = img_size if img_size is not None else default_size
    transform = make_uint8_transform(size)

    class RGBWrapper(Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return min(n_samples, len(self.base))

        def __getitem__(self, idx):
            img, _ = self.base[idx]
            if img.shape[0] == 1:   # MNIST: gray -> RGB
                img = img.repeat(3, 1, 1)
            return img

    kwargs = dict(root=data_root, transform=transform, download=True)
    if name == "stl10":
        kwargs["split"] = "test"
    else:
        kwargs["train"] = False

    base_ds = cls(**kwargs)
    wrapped = RGBWrapper(base_ds)

    return DataLoader(wrapped, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, drop_last=False)


def get_real_loader_from_folder(folder, n_samples, batch_size, num_workers, img_size=None):
    transform = make_uint8_transform(img_size)
    ds = FolderDataset(folder, transform=transform)
    if n_samples < len(ds):
        from torch.utils.data import Subset
        import random
        indices = random.sample(range(len(ds)), n_samples)
        ds = Subset(ds, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, drop_last=False)



def save_results(generated_dir, fid_value):
    """Salva (in append) il risultato FID"""
    result_path = Path(generated_dir) / "fid_results.json"

    entry = {
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
        "fid":           round(fid_value, 6),
        "n_samples":     len(FolderDataset(generated_dir)),
        "real_source":   FLAGS.real_dir if FLAGS.real_dir else FLAGS.dataset,
        "img_size":      FLAGS.img_size,
        "feature_dim":   FLAGS.feature_dim,
        "generated_dir": str(Path(generated_dir).resolve()),
    }

    if result_path.exists():
        with open(result_path, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(entry)

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Risultato salvato in: {result_path}")


'''funzione calcolo fid'''
def compute_fid(argv):
    # Validazione flags mutuamente esclusivi
    if FLAGS.real_dir is None and FLAGS.dataset is None:
        raise ValueError("Specifica --real_dir oppure --dataset.")
    if FLAGS.real_dir is not None and FLAGS.dataset is not None:
        raise ValueError("--real_dir e --dataset sono mutuamente esclusivi.")
    if FLAGS.feature_dim not in (64, 192, 768, 2048):
        raise ValueError(f"--feature_dim deve essere uno tra 64, 192, 768, 2048.")

    device = (
        torch.device(f"cuda:{FLAGS.gpu}")
        if FLAGS.gpu >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # --- Loader immagini generate -------------------------------------------
    gen_transform = make_uint8_transform(FLAGS.img_size)
    gen_ds = FolderDataset(FLAGS.generated_dir, transform=gen_transform)
    n_samples = len(gen_ds)
    print(f"Immagini generate trovate: {n_samples}")

    gen_loader = DataLoader(
        gen_ds,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        drop_last=False,
    )

    # --- Loader immagini reali -----------------------------------------------
    if FLAGS.real_dir is not None:
        print(f"Immagini reali da cartella: {FLAGS.real_dir}")
        real_loader = get_real_loader_from_folder(
            FLAGS.real_dir, n_samples, FLAGS.batch_size, FLAGS.num_workers, FLAGS.img_size
        )
    else:
        print(f"Immagini reali da dataset built-in: {FLAGS.dataset}")
        real_loader = get_real_loader_from_dataset(
            FLAGS.dataset, FLAGS.data_root, n_samples,
            FLAGS.batch_size, FLAGS.num_workers, FLAGS.img_size
        )

    # --- FrechetInceptionDistance --------------------------------------------
    fid_metric = FrechetInceptionDistance(
        feature=FLAGS.feature_dim,
        normalize=False,
    ).to(device)

    print("Caricamento immagini reali nella metrica FID...")
    for batch in real_loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        fid_metric.update(batch.to(device), real=True)

    print("Caricamento immagini generate nella metrica FID...")
    for batch in gen_loader:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        fid_metric.update(batch.to(device), real=False)

    fid_value = fid_metric.compute().item()

    save_results(FLAGS.generated_dir, fid_value)

    print(f"\n{'='*40}")
    print(f"  FID = {fid_value:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    app.run(compute_fid)