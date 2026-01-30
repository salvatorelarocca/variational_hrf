'''
Definisce e ritorna il modello UNet in base al dataset e ai parametri specificati.
'''
from models.unet import UNetModelWrapper as UNetModelWrapperBaseline
from models.unet_cat_xt_v import UNetModelWrapper as UNetModelWrapperCat
from models.unet_2_unet import UNetModelWrapper as UNetModelWrapper2UNet

'''
Seleziona:
-UNetModelWrapperCat per MNIST con HRF, 
-UNetModelWrapper2UNet per altri dataset con HRF,
-UNetModelWrapperBaseline per modelli senza HRF,
memorizzandolo in UnetModelWrapper.

Successivamente, configura il modello in base al dataset:

'''

def get_model(dataset, data_shape, channel_mult, num_channel, device, hrf=True):
    #selezione dei modelli
    if hrf:
        if dataset == "mnist":
            UNetModelWrapper = UNetModelWrapperCat #se hrt è true utilizza il modello cat per mnist 
        else:
            UNetModelWrapper = UNetModelWrapper2UNet #oppure 2unet per gli altri dataset
    else:
        UNetModelWrapper = UNetModelWrapperBaseline #se hrt è false utilizza il modello baseline progettato da openai
    
    # Configurazione del modello in base al dataset
    if dataset == "mnist":
        unet = UNetModelWrapper(
            dim=data_shape,
            num_res_blocks=1,
            num_channels=num_channel,
        ).to(
            device
        )
    elif dataset == "cifar10":
        channel_mult = [int(i) for i in channel_mult]
        unet = UNetModelWrapper(
            dim=data_shape,
            num_res_blocks=2,
            num_channels=num_channel,
            channel_mult=channel_mult,
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
        ).to(
            device
        )
    elif dataset == "imagenet32":
        channel_mult = [int(i) for i in channel_mult]
        unet = UNetModelWrapper(
            dim=data_shape,
            num_res_blocks=2,
            num_channels=num_channel,
            channel_mult=channel_mult,
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16,8",
            dropout=0.1,
        ).to(
            device
        )
    else:
        raise NotImplementedError
    
    return unet