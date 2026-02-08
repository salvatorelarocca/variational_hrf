import torch
from model import get_model  # o dove è definito get_model
from hooks import unet_shape_hook
from models.unet_2_unet import ResBlock_v, ResBlock, Downsample, Upsample, AttentionBlock

# Configurazione minima
device = "cpu"
in_channels = 3
image_size = 32  # esempio per CIFAR10
batch_size = 1
hrf = False

# Crea l'UNet
unet = get_model(
    "for_cifar10mini",
    data_shape=(in_channels, image_size, image_size),
    channel_mult=[1, 2, 2, 2],
    num_channel=32,
    device=device,
    hrf=hrf,
)

# REGISTRA GLI HOOK DI DEBUG
DEBUG_TYPES = (ResBlock_v, 
               ResBlock, 
               Downsample, 
               Upsample, 
               AttentionBlock
               )

for name, module in unet.named_modules():
    if isinstance(module, DEBUG_TYPES):
        module.register_forward_hook(unet_shape_hook(name))

# Batch di input fittizio
x = torch.randn(batch_size, in_channels, image_size, image_size, device=device)
t = torch.randint(0, 100, (batch_size,), device=device)  # timesteps fittizi
v = torch.randn_like(x)  # se usi HRF

# Esegui un forward di prova
with torch.no_grad():
    if hrf:  # se il modello HRF
        pred = unet(t, v, t, x)
    else:
        pred = unet(t, x)
