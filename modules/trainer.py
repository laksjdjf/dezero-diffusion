import dezero
from dezero import DataLoader
from tqdm import tqdm
from dezero.transforms import Compose, ToFloat, Normalize
import numpy as np
import cupy as xp
import os

class Trainer:
    def __init__(self, diffusion, batch_size, lr, ucg=0.1, output_dir="outputs", dataset="mnist"):
        self.batch_size = batch_size
        self.diffusion = diffusion
        self.ucg = ucg
        if dataset == "mnist":
            self.train_set = dezero.datasets.MNIST(train=True, transform=Compose([ToFloat(), Normalize(127.5, 127.5)]),)
        elif dataset == "cifar10":
            self.train_set = dezero.datasets.CIFAR10(train=True, transform=Compose([ToFloat(), Normalize(127.5, 127.5)]),)
        else:
            raise ValueError(f"{dataset} is not supported.")
        
        self.train_loader = DataLoader(self.train_set, batch_size)
        self.train_loader.to_gpu()
        
        self.optimizer = dezero.optimizers.Adam().setup(self.diffusion.unet)
        self.optimizer.add_hook(dezero.optimizers.WeightDecay(lr))
        
        self.output_dir = os.path.join(output_dir, "models")
        self.image_dir = os.path.join(output_dir, "images")
        self.log_dir = os.path.join(output_dir, "logs")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def train(self, epochs, save_n_epochs=5, sample_cfg_scale=3.0, limited_steps=10000000):
        progress_bar = tqdm(range(epochs*len(self.train_set)//self.batch_size), desc="Total Steps", leave=False)
        loss_ema = None
        loss_emas = []
        for epoch in range(epochs):
            steps = 0
            for x, c in self.train_loader:

                ucg_random = xp.random.uniform(0, 1, size=(x.shape[0], 1)).astype(xp.float32) > self.ucg
                context = xp.eye(self.diffusion.unet.context_dim)[c] # one hot vector化
                context *= ucg_random
                
                loss = self.diffusion.train_step(x, context)
                self.diffusion.unet.cleargrads()
                loss.backward()
                self.optimizer.update()

                if loss_ema is not None:
                    loss_ema = 0.9 * loss_ema + 0.1 * float(loss.data)
                else:
                    loss_ema = float(loss.data)
                loss_emas.append(loss_ema)
                
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": loss_ema})
                steps += 1
                if steps > limited_steps: # test用
                    break

            if ((epoch+1) % save_n_epochs) == 0:
                self.diffusion.unet.save_weights(os.path.join(self.output_dir, f"model_{epoch:02}.npz"))
                self.diffusion.unet.to_gpu()  # セーブ時にcpuに移動してしまう仕様
                np.save(os.path.join(self.log_dir, f"log_{epoch:02}.npy"), np.array(loss_emas))
                self.diffusion.generate_grid(4, x.shape[1], x.shape[2], x.shape[3], os.path.join(self.image_dir, f"image_{epoch:02}.png"), self.train_set.labels(), cfg_scale=sample_cfg_scale)
