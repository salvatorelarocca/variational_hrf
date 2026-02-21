from models.unet import UNetModelWrapper as UNetModelWrapperBaseline
from models.unet_cat_xt_v import UNetModelWrapper as UNetModelWrapperCat
from models.unet_2_unet import UNetModelWrapper as UNetModelWrapper2UNet


def get_model(dataset, data_shape, channel_mult, num_channel, device, hrf=True, latent_dim=128, use_scale_shift=False, use_latent=False):
    if hrf:
        if dataset == "mnist":
            UNetModelWrapper = UNetModelWrapperCat #se hrt è true utilizza il modello cat per mnist
            _model_name = "UNetModelWrapperCat"
        else:
            UNetModelWrapper = UNetModelWrapper2UNet #oppure 2unet per gli altri dataset
            _model_name = "UNetModelWrapper2UNet"
    else:
        UNetModelWrapper = UNetModelWrapperBaseline #se hrt è false utilizza il modello baseline progettato da openai
        _model_name = "UNetModelWrapperBaseline"
    
    print(f"Selected model: {_model_name}")

    if dataset == "mnist":
        print("for mnist dataset")
        unet = UNetModelWrapper(
            dim=data_shape,
            num_res_blocks=1,
            num_channels=num_channel,
            latent_dim=latent_dim,
            use_latent=use_latent,
            use_scale_shift_norm=use_scale_shift
        ).to(
            device
        )
    elif dataset == "cifar10":
        print("for cifar10 dataset")
        channel_mult = [int(i) for i in channel_mult]
        unet = UNetModelWrapper(
            dim=data_shape,
            num_res_blocks=2, #qualsiasi numero non rompe la coerenza dell'archi, è un iperparametro da ottimizzare e scegliere
            num_channels=num_channel,
            channel_mult=channel_mult,
            num_heads=4, #num_channels / num_heads = num_head_channels per restare coerente con l'architettura
            num_head_channels=16,
            attention_resolutions="16", # durante il downsample quando arriva a 16x16 applica l'attenzione
            dropout=0.1,
            latent_dim=latent_dim,
            use_latent=use_latent,
            use_scale_shift_norm=use_scale_shift
        ).to(
            device
        )
    elif dataset == "imagenet32":
        print("for imagenet32 dataset")
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
            latent_dim=latent_dim,
            use_latent=use_latent,
            use_scale_shift_norm=use_scale_shift
        ).to(
            device
        )
    else:
        raise NotImplementedError
    
    return unet