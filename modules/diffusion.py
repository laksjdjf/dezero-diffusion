from tqdm import tqdm
import dezero
import dezero.functions as F
from PIL import Image
from modules.unet import UNet
from modules.sampler import DDPM
import cupy as xp
import matplotlib.pyplot as plt

class Diffusion:
    '''
    ノイズ予測・サンプラーを受け取って画像生成・学習ステップを定義する。
    '''

    def __init__(self, unet, sampler):
        self.unet = unet
        self.unet.to_gpu()
        self.sampler = sampler

    def generate(self, context, channels, height, width):
        '''
        画像生成を行うメソッド。
        
        context: ラベルのxp配列
        '''
        batch_size = context.shape[0]
        with dezero.test_mode():
            with dezero.no_grad():
                x = xp.random.randn(batch_size, channels, height, width) # 初期ノイズ x_0
                for t in tqdm(reversed(range(1000)), total=1000): # t = 999, ..., 0
                    noise_pred = self.unet(x, xp.array([[t]]*batch_size).astype(xp.int32), context) # ノイズ予測
                    x = self.sampler.step(x, noise_pred, t) # x_{t+1} -> x_{t}

        images = []
        for image in x:
            image = (xp.clip(image.data*127.5 + 127.5, 0, 255)).astype(xp.uint8) # 0~255に変換
            image = xp.asnumpy(image)
            image = image.transpose(1, 2, 0).squeeze()
            image = Image.fromarray(image)
            images.append(image)
        return images
    

    def generate_grid(self, num_images, channels, height, width, image_path):
        '''
        生成画像をラベルごとにグリッド状に並べて保存するメソッド。
        '''
        num_labels = self.unet.context_dim
        fig, axes = plt.subplots(num_labels, num_images, figsize=(7, 14))
        images = self.generate(xp.arange(num_labels).repeat(num_images),channels, height, width)
        for i in range(num_labels):
            for j in range(num_images):
                axes[i, j].imshow(images[i*num_images+j], cmap='gray')
                axes[i, j].axis('off')

            axes[i, 0].text(-10, 14, f'{i}', fontsize=12, verticalalignment='center')
        fig.savefig(image_path)

    def train_step(self, image, context):
        '''
        学習1ステップ分を実装、lossを返す。
        '''

        #　加えるノイズ
        noise = xp.random.randn(*image.shape)
        
        #　ランダムな時刻を選択
        t = xp.random.randint(0, 1000, size=(image.shape[0], 1)).astype(xp.int32)
        
        # ノイズを加える
        noisy_image = self.sampler.add_noise(image, noise, t)
        
        # ノイズ予測
        noise_pred = self.unet(noisy_image, t, context)
        
        # ノイズ予測と実際のノイズのMSEを計算
        loss = F.mean_squared_error(noise, noise_pred) / (image.shape[1]*image.shape[2]*image.shape[3])
        return loss


if __name__ == "__main__":
    unet = UNet()
    ddpm = DDPM()
    diffusion = Diffusion(unet, ddpm)
    image = xp.random.randn(3, 1, 28, 28)
    loss = diffusion.train_step(image, xp.array([0, 1, 2]))
    # images = diffusion.generate(xp.array([0,1,2]),1,28,28)
