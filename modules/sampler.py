import cupy as xp
from modules.utils import expand_2d


class DDPM:
    def __init__(self, beta_start=1e-4, beta_end=0.02, T=1000):
        '''
        Denoise Diffusion Probabilistic Modelの実装
        引数のデフォルトは論文通りの値
        '''
        self.beta_start = beta_start # beta_0
        self.beta_end = beta_end # beta_T
        self.T = T 
        self.beta = xp.linspace(beta_start, beta_end, T) # beta_0, ..., beta_T
        self.sqrt_beta = xp.sqrt(self.beta) 
        self.alpha = 1 - self.beta # alpha_0, ..., alpha_T
        self.alpha_bar = xp.cumprod(self.alpha) # Π_{i=0}^t alpha_i
        self.sqrt_alpha_bar = xp.sqrt(self.alpha_bar) 
        self.beta_bar = 1 - self.alpha_bar
        self.sqrt_beta_bar = xp.sqrt(self.beta_bar)
        self.one_over_sqrt_alpha = 1 / xp.sqrt(self.alpha) # ddpm.stepで使う
        self.beta_over_sqrt_beta_bar = self.beta / self.sqrt_beta_bar # ddpm.stepで使う

    def add_noise(self, x, noise, t):
        '''
        時刻tに応じたノイズを加える
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_beta_bar_t * noise
        '''
        return expand_2d(self.sqrt_alpha_bar[t]) * x + expand_2d(self.sqrt_beta_bar[t]) * noise

    def step(self, x, noise_pred, t):
        '''
        x_t -> x_{t-1}のサンプリング
        x_{t-1} = 1/sqrt_alpha_t * (x_t - beta_t/sqrt_beta_bar_t * noise_pred) + sqrt_beta_t * noise
        '''
        noise = xp.random.randn(*x.shape)
        prev_x = self.one_over_sqrt_alpha[t] * (x - self.beta_over_sqrt_beta_bar[t] * noise_pred) + self.sqrt_beta[t] * noise
        return prev_x


if __name__ == "__main__":
    ddpm = DDPM()
    x = xp.random.randn(2, 3, 28, 28)
    noise_pred = xp.random.randn(2, 3, 28, 28)
    t = 999
    ddpm.step(x, noise_pred, t)
