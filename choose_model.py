from models.unet import UNetModelWrapper as UNetModelWrapperBaseline
from models.unet_cat_xt_v import UNetModelWrapper as UNetModelWrapperCat
from models.unet_2_unet import UNetModelWrapper as UNetModelWrapper2UNet

def get_model(
    dataset,
    data_shape,
    channel_mult,
    num_channel,
    device,
    model_type="baseline",
    latent_dim=128,
    use_scale_shift=False,
    use_latent=False
):
    """
    Seleziona il modello UNet da usare in base alla flag model_type
    Parametri:
        dataset: string, nome del dataset ('mnist', 'cifar10', 'imagenet32')
        data_shape: tuple, shape del dataset (C, H, W)
        channel_mult: lista di int, moltiplicatori dei canali
        num_channel: int, numero di canali base
        device: torch.device
        model_type: 'baseline', 'cat', '2unet'
        latent_dim: dimensione spazio latente
        use_scale_shift: bool
        use_latent: bool, se usare VAE
    """
    if model_type == "baseline":
        UNetModelWrapper = UNetModelWrapperBaseline
    elif model_type == "unet_cat_hrf":
        UNetModelWrapper = UNetModelWrapperCat
    elif model_type == "2unet_hrf":
        UNetModelWrapper = UNetModelWrapper2UNet
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if use_latent:
        print(f"Selected model: {model_type} with latent space of dimension {latent_dim}")
    else:
        print(f"Selected model: {model_type} without latent space")

    if dataset == "mnist":
        unet = UNetModelWrapper(
            dim=data_shape,
            num_res_blocks=1,
            num_channels=num_channel,
            latent_dim=latent_dim,
            use_latent=use_latent,
            use_scale_shift_norm=use_scale_shift
        ).to(device)

    elif dataset == "cifar10":
        channel_mult = [int(i) for i in channel_mult]
        unet = UNetModelWrapper(
            dim=data_shape,
            num_res_blocks=2,
            num_channels=num_channel,
            channel_mult=channel_mult,
            num_heads=4,
            num_head_channels=16,
            attention_resolutions="16",
            dropout=0.1,
            latent_dim=latent_dim,
            use_latent=use_latent,
            use_scale_shift_norm=use_scale_shift
        ).to(device)

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
            latent_dim=latent_dim,
            use_latent=use_latent,
            use_scale_shift_norm=use_scale_shift
        ).to(device)

    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")

    return unet