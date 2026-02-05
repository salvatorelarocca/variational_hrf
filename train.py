import copy
import os

import torch
from absl import app, flags # per la gestione delle flags da linea di comando pacchetto absl-py
from torch.utils import tensorboard # per la scrittura su tensorboard
from tqdm import trange


from dataset import get_datalooper
from model import get_model
from utils import ema, generate_samples, load_model
from cfm import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher, #non usato 
)


FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", "./", help="output directory")
flags.DEFINE_string("imagenet_root", "./", help="root directory for imagenet")
flags.DEFINE_string("exp_name", "base", help="experiment name")
flags.DEFINE_enum("dataset", "cifar10", ["cifar10", "mnist", "imagenet32"], help="dataset name")
flags.DEFINE_string("model", "for_cifar10mini", help="Choose the model...")
flags.DEFINE_bool("hrf", False, help="train hrf or baseline") # False per baseline, True per hrf
flags.DEFINE_integer("gpu", 0, help="GPU number")

# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")
flags.DEFINE_list("channel_mult", [1, 2, 2, 2], help="channel_mult of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping") 
'''
non guarda le epoche ma i passi, quindi con un batch size 128 su cifar10 
che ha circa 50k immagini abbiamo circa 390 passi per epoca ad arrivare
a 400k passi facciamo circa 1025 epoche
'''
flags.DEFINE_integer(
    "total_steps", 400_001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
'''
numero di step durante i quali il lr aumenta linearmente
dalla condizione iniziale al valore target
'''
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup") 
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
'''
probabilmente utilizzato per calcolare il costo di optimal transport
la flag permette di settare la dimensione del batch su cui calcolare il costo
'''
flags.DEFINE_integer("ot_bs", 128, help="optimal transport batch size")

flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
'''
dacay per l'ema (exponential moving average) del modello
'''
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
'''
permette di riprendere da un checkpoint precedente e di non partire da zero
'''
flags.DEFINE_bool("continue_train", False, help="continue training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)
flags.DEFINE_integer(
    "tb_step",
    50,
    help="frequency of saving loss to tensorboard",
)


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step, mix method, before middle block:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
        FLAGS.tb_step,
    )

    '''Inizializzazioni per la riproducuibilità dei risultati'''
    seed = 0
    torch.manual_seed(seed) # seed per la CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # seed per la singola GPU
        torch.cuda.manual_seed_all(seed) # seed per tutte le GPU
    '''alcune operazioni di cud nn non sono deterministiche per default
    non solo benchmark disabilita la ricerca delle migliori configurazioni'''
    torch.backends.cudnn.deterministic = True # rende le operazioni di cudnn deterministiche
    torch.backends.cudnn.benchmark = False # disabilita la ricerca automatica delle migliori configurazioni di cudnn

    if FLAGS.gpu is not None and FLAGS.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{FLAGS.gpu}")
    else:
        device = torch.device("cpu")

    '''Creazione delle cartelle per il salvataggio dei risultati'''
    savedir = os.path.join(FLAGS.output_dir, f"results_{FLAGS.model}", f"{FLAGS.exp_name}")
    os.makedirs(savedir, exist_ok=True)
    ckptdir = os.path.join(savedir, "ckpt")
    os.makedirs(ckptdir, exist_ok=True)
    imgdir = os.path.join(savedir, "img_train")
    os.makedirs(imgdir, exist_ok=True)
    writer = tensorboard.SummaryWriter(savedir)

    datalooper, data_shape = get_datalooper(
        FLAGS.dataset, 
        FLAGS.batch_size, 
        FLAGS.num_workers, 
        train=True, 
        imagenet_root=FLAGS.imagenet_root,
    ) # riceve il dataloader infinito e la shape dei dati 

    unet = get_model(
        FLAGS.model,
        data_shape,
        FLAGS.channel_mult,
        FLAGS.num_channel,
        device,
        hrf=FLAGS.hrf,
    ) # crea il modello UNet
    
    '''Calcolo del numero di parametri del modello per definire la complessità del modello'''
    model_size = 0
    for param in unet.parameters():
        model_size += param.data.nelement()
    print(f"Model params number: {model_size}")
    print("Model params: %.2f M" % (model_size / 1000 / 1000))

    ema_model = copy.deepcopy(unet) # copia del modello per ema
    optim = torch.optim.Adam(unet.parameters(), lr=FLAGS.lr) # ottimizzatore Adam
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr) # permette di variare il learning rate durante l'addestramento

    # continue training
    '''Caricamento del checkpoint se esiste e se la flag è settata:
    Ordina i file dei checkpoint per numero di step.
    Il nome dei checkpoint ha la seguente struttura: "exp_dataset_weights_step_20000.pt".
    x.split('_')[-1] prende "20000.pt" .split('.')[0] prende "20000" e int() lo converte in numero.
    sorted(...)[-1] prende l’ultimo checkpoint, cioè l’ultimo step salvato.
    '''
    cur_step = 0
    ckpt_list = os.listdir(ckptdir)
    if FLAGS.continue_train and len(ckpt_list) > 0:
        ckpt = sorted(ckpt_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        print(f"loading {ckpt}")
        ckpt = torch.load(os.path.join(ckptdir, ckpt), weights_only=True) # carica i pesi
        load_model(unet, ckpt['model']) # del modello unet
        load_model(ema_model, ckpt['ema_model']) # del modello ema
        optim.load_state_dict(ckpt['optim']) # stato dell'ottimizzatore
        sched.load_state_dict(ckpt['sched']) # stato del scheduler
        cur_step = ckpt['step'] # ripristina lo step corrente

    FM = ConditionalFlowMatcher(sigma=0.0)
    # FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0, ot_bs=FLAGS.ot_bs)

    with trange(cur_step, cur_step + FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1 = next(datalooper).to(device)
            x0 = torch.randn_like(x1)
            t, xt, target = FM.sample_location_and_conditional_flow(x0, x1)
            if FLAGS.hrf:
                v0 = torch.randn_like(target)
                tau, vtau, target = FM.sample_location_and_conditional_flow(v0, target)
                pred = unet(tau, vtau, t, xt)
            else:
                pred = unet(t, xt)
            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(unet, ema_model, FLAGS.ema_decay)  # new
            pbar.set_description(f'loss: {loss.item():.4f}')
            pbar.update(1)
            
            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                generate_samples(unet, imgdir, step, (16, *data_shape), device, net_="normal", hrf=FLAGS.hrf)
                generate_samples(ema_model, imgdir, step, (16, *data_shape), device, net_="ema", hrf=FLAGS.hrf)
                torch.save(
                    {
                        "model": unet.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    os.path.join(ckptdir, f"{FLAGS.exp_name}_{FLAGS.model}_weights_step_{step}.pt"),
                )
            if FLAGS.tb_step > 0 and step % FLAGS.tb_step == 0:
                writer.add_scalar("training_loss", loss, step)


if __name__ == "__main__":
    app.run(train)
