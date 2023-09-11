from modules.unet import UNet
from modules.sampler import DDPM
from modules.diffusion import Diffusion
from modules.trainer import Trainer

dataset = "mnist" # "mnist" or "cifar10"
unet = UNet(out_channels=1 if dataset =="mnist" else 3, hidden_channels=64, num_layers=2)
ddpm = DDPM()
diffusion = Diffusion(unet, ddpm)
trainer = Trainer(
    diffusion,
    batch_size=128,
    lr = 1e-4,
    output_dir="mnist",
    dataset=dataset
)

trainer.train(100)