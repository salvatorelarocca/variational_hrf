import copy
import os

import torch
from torch import distributed as dist
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchvision.utils import save_image

'''
Parametri:
`model` è il modello utilizzato per il campo vettoriale, 
`sample_shape` è la forma del campione da generare,
`nfe` è il numero di funzioni valutate,
`device` è il dispositivo su cui eseguire il campionamento,
`integration_method` è il metodo di integrazione da utilizzare ("euler" o "dopri5").
Funzionalità:
`sample_rf` sample rectified flow seleziona il metodo di integrazione
Returns:
Un tensore contenente i campioni generati e il numero di funzioni valutate
'''
def sample_rf(model, sample_shape, nfe, device, integration_method="euler"):
        if integration_method == "euler":
            xt = sample_rf_euler(model, sample_shape, nfe, device)
        elif integration_method == "dopri5":
            xt, nfe = sample_rf_dopri5(model, sample_shape, device)
        else:
            raise NotImplementedError
        return xt, nfe


'''
Parametri:
`model` è il modello utilizzato per il campo vettoriale, 
`sample_shape` è la forma del campione da generare,
`nfe` è il numero di funzioni valutate,
`device` è il dispositivo su cui eseguire il campionamento,

Funzionalità:
`sample_rf_euler` utilizza NeuralODE con il solver di Eulero
Returns:
Un tensore contenente i campioni generati shape B x C x H x W
'''
def sample_rf_euler(model, sample_shape, nfe, device):
    node_ = NeuralODE(model, solver="euler", sensitivity="adjoint") # istanza del Integratore numerico
    with torch.no_grad(): #non crea il grafo computazionale il modello è usato come funzione pura
        traj = node_.trajectory(
            torch.randn(sample_shape, device=device), # campione di partenza x_0 shape B,C,H,W
            t_span=torch.linspace(0, 1, nfe, device=device), # da 0-1 con passo 1/(nfe-1) t
        ) # calcolo dell'intera traiettoria su nfe punti da 0 a 1 [x_0, x_1, ..., x_nfe-1] shape nfe,B,C,H,W
        traj = traj[-1, :].view(sample_shape) # prendo solo l'ultimo punto della traiettoria x_1, 
    return traj #return x_1

'''
Parametri:
`model` è il modello utilizzato per il campo vettoriale, 
`sample_shape` è la forma del campione da generare,
`device` è il dispositivo su cui eseguire il campionamento,

Funzionalità:
`sample_rf_dopri5` 
Returns:
`x_1` e `nfe`
'''
def sample_rf_dopri5(model, sample_shape, device):
    with torch.no_grad():
        step_counter = {"steps": 0}
        def wrapped_model(t, xt):
            step_counter["steps"] += 1 
            return model(t, xt) 
        t_span = torch.linspace(0, 1, 2, device=device)
        xt = odeint(
            wrapped_model, 
            torch.randn(sample_shape, device=device), 
            t_span, 
            rtol=1e-5, 
            atol=1e-5, 
            method="dopri5", 
        )
        xt = xt[-1, :] #stato finale come prima
        
    return xt, step_counter['steps'] #return x_1, nfe


def sample_hrf(model, sample_shape, N, M, device, integration_method="euler", vae=None):
        if integration_method == "euler":
            xt = sample_hrf_euler(model, vae, sample_shape, N, M, device)
            nfe = N * M
        elif integration_method == "dopri5":
            xt, nfe = sample_hrf_dopri5(model, sample_shape, N, device)
        else:
            raise NotImplementedError
        return xt, nfe

'''
Parametri:
idem sopra
Funzionalità: restituisce i campioni generati utilizzando il metodo di Eulero con il modello HRF
'''
def sample_hrf_euler(model, vae, sample_shape, N, M, device):
    with torch.no_grad():
        batchsize = sample_shape[0] # utilizzato in expand di seguito
        xt = torch.randn(sample_shape, device=device) # x_0
        
        t_values = torch.arange(N, device=device) / N # tempo esterno
        tau_values = torch.arange(M, device=device) / M # tempo interno

        for i in range(N): # for esterno tempo t
            t = t_values[i].expand(batchsize) # t_i
            vtau = torch.randn(sample_shape, device=device) # v_0
            for j in range(M): # for interno tempo tau
                tau = tau_values[j].expand(batchsize) 
                # z, _, _ = vae(vtau, _, tau, t) # come target che metto???
                a = model(tau, vtau, t, xt) # calcolo interazione interna
                vtau += a / M # a * (1/M passo di integrazione)
            xt += vtau / N # a * (1/N passo di integrazione)

    return xt

'''Equivalente a sopra ma con dopri5'''
def sample_hrf_dopri5(model, sample_shape, N, device):
    with torch.no_grad():
        batchsize = sample_shape[0]
        xt = torch.randn(sample_shape, device=device)
        t_values = torch.arange(N, device=device) / N
        step_counter = {"steps": 0}
        for i in range(N):
            t = t_values[i].expand(batchsize)
            def wrapped_model(tau, vtau):
                step_counter["steps"] += 1 
                return model(tau, vtau, t, xt)
            tau_span = torch.linspace(0, 1, 2, device=device)
            vtau = odeint(
                wrapped_model, 
                torch.randn_like(xt), 
                tau_span, 
                rtol=1e-5, 
                atol=1e-5, 
                method="dopri5", 
            )
            vtau = vtau[-1, :]
            xt += vtau / N

    return xt, step_counter['steps']

'''

'''
def generate_samples(model, savedir, step, shape, device, net_="normal", hrf=True):
    """Save generated images for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval() # setta il modello in eval mode

    model_ = copy.deepcopy(model) # crea una copia del modello per il campionamento

    if hrf:
        samples, _ = sample_hrf(model_, shape, 1, 100, device) # hrf + euler, t=1 e tau=100 di default a nfe 100
    else:
        samples, _ = sample_rf(model_, shape, 100, device) # rf + euler di default a nfe 100
    
    # salvataggio delle immagini generate
    # samples.clip(-1,1) satura i valori tra -1 e 1 (gli outlier generati dal modello vengono portati ai limiti -1 oppure 1)
    # poi si rimappa nell'intervallo [0,1] con /2 +0.5
    save_image(samples.clip(-1, 1) / 2 + 0.5, os.path.join(savedir, f"{net_}_generated_FM_images_step_{step}.png"), nrow=4)

    model.train()

'''
Questa funzione implementa EMA(Exponential Moving Average).
Copia i pesi del modello `source` nel modello `target` utilizzando una media esponenziale mobile con decadimento `decay`.
Rende l'integrazione numerica più stabile durante il campionamento. Tutto si traduce in triettorie più stabili 
campioni più nitidi.
Quasi tutti i modelli addestrano una rete e la smussano/stabilizzano numericamente per l'integrazione
usando EMA che viene utilizzata per il campionamento.
'''
def ema(source, target, decay):
    source_dict = source.state_dict() # parametri(pesi, bais...) del modello sotto forma di dizionario
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )
    '''
    equivalente di:
    with torch.no_grad():
        for key in source_dict.keys():
            target_dict[key] = target_dict[key] * decay + source_dict[key] * (1-decay)
    '''

'''
Trasforma un dataloader finito in uno stream infinito di batch.
L'iteratore di Dataloader ad esaurimento batch solleva StopIteration.
Questa funzione utilizza un ciclo while True per creare un generatore infinito che ripete indefinitamente
l'iterazione sui batch del dataloader.
Non abbiamo bisogno del concetto di epoca in questo caso perché vogliamo solo un flusso continuo di dati per l'addestramento.
Tecnicamente:
yield ritorna un generatore quando richiamo richiamo la prima volta next() sul generatore mi viene dato il primo batch grazie al primo loop del ciclo, 
lo stato viene mantenuto e al prossimo next() mi viene dato il successivo. 
'''
def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader): #dataloader è un iterable iter(dataloader) restituisce un iterator x, y tupla di batch
            yield x

'''Carica i pesi salvati in state_dict nel modello.
Gestisce il caso in cui i pesi siano stati salvati utilizzando DataParallel,
che aggiungono un prefisso 'module.' ai nomi dei parametri.'''
def load_model(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)

'''Inizializza l'ambiente distribuito.'''
def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    print(f"[Rank {rank}] MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    print(f"[Rank {rank}] Initializing process group...")
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )
    print(f"[Rank {rank}] Process group initialized!")

