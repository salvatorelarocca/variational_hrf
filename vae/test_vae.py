from .vae_model import *

def test_vae():
    model = BetaVAE(in_channels=3, latent_dim=128) 
    v_0 = torch.randn(16, 3, 32, 32)   
    v_1 = torch.randn(16, 3, 32, 32)
    v_tau = torch.randn(16, 3, 32, 32)
    tau = torch.randn(1)  
    print(tau.shape)
    z, mu, log_var = model(v_0, v_1, v_tau, tau)
    assert z.shape == (16, 128)
    assert mu.shape == (16, 128)
    assert log_var.shape == (16, 128)
    print("tutto apposto!")

if __name__ == "__main__":
    test_vae()