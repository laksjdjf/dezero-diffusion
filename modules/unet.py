from dezero.models import Model, Sequential
import dezero.functions as F
import dezero.layers as L
from dezero.core import Function

from modules.utils import expand_2d
import cupy as xp


class Cat(Function):
    '''
    dezeroにはcatが定義されていないので、chatgptに作ってもらった。
    '''
    def __init__(self, axis=0):
        self.axis = axis

    def forward(self, *inputs):
        z = xp.concatenate(inputs, axis=self.axis)
        return z

    def backward(self, gz):
        inputs = self.inputs
        gradients = []
        start_idx = 0

        for x in inputs:
            end_idx = start_idx + x.shape[self.axis]

            indices = [slice(None)] * gz.ndim
            indices[self.axis] = slice(start_idx, end_idx)

            gradients.append(gz[tuple(indices)])

            start_idx = end_idx

        return tuple(gradients)


def cat(inputs, axis=0):
    return Cat(axis=axis)(*inputs)


class ConvBlock(Model):
    '''
    複数の畳み込み層+ばっちのーむ+ReLUによるブロック。
    最後にアップサンプリングかダウンサンプリングを行うこともある（lastで指定）。
    '''
    def __init__(self, channels, num_layers, last=None):
        '''
        channels: 畳み込み層の出力チャンネル数
        num_layers: 畳み込み層の数
        last: None or "up" or "down"
        '''
        super().__init__()
        convs = []
        norms = []
        for _ in range(num_layers):
            convs.append(L.Conv2d(channels, kernel_size=3, pad=1, nobias=True))
            norms.append(L.BatchNorm())

        self.convs = Sequential(*convs)
        self.norms = Sequential(*norms)

        if last == "up":
            self.last = L.Deconv2d(channels, kernel_size=4, stride=2, pad=1)
        elif last == "down":
            self.last = L.Conv2d(channels, kernel_size=3, stride=2, pad=1)
        else:
            self.last = None

    def forward(self, x):
        for conv, norm in zip(self.convs.layers, self.norms.layers):
            x = F.relu(norm(conv(x)))

        if self.last is not None:
            x = self.last(x)
        return x


class UNet(Model):
    def __init__(self, out_channels=1, context_dim=10, hidden_channels=16, num_blocks=2, num_layers=3):
        '''
        out_channels: 出力画像のチャンネル数
        context_dim: ラベルの数
        hidden_channels: 中間のチャンネル数、ダウンサンプルごとに2倍になる。
        num_blocks: ブロックの数。
        num_layers: ブロックごとの畳み込み層の数。
        '''
        super().__init__()
        self.context_dim = 10
        self.conv_in = L.Conv2d(hidden_channels, kernel_size=3, pad=1)

        # 時刻[0,1000]を全結合層に入力する。本当はsinとか使うやつにしたい。
        time_embs = []
        for i in range(num_blocks):
            if i == 0:
                time_embs.append(L.Linear(hidden_channels))
            else:
                time_embs.append(L.Linear(hidden_channels*(2**(i-1))))
        self.time_embs = Sequential(*time_embs)

        # one hot vectorのラベルを全結合層に入力する。
        context_embs = []
        for i in range(num_blocks):
            if i == 0:
                context_embs.append(L.EmbedID(self.context_dim, hidden_channels))
            else:
                context_embs.append(L.EmbedID(self.context_dim, hidden_channels*(2**(i-1))))
        self.context_embs = Sequential(*context_embs)

        self.down_blocks = Sequential(
            *[ConvBlock(hidden_channels*(2**i), num_layers, "down") for i in range(num_blocks)]
        )

        self.mid_blocks = ConvBlock(hidden_channels*2**num_blocks, num_layers)

        self.up_blocks = Sequential(
            *[ConvBlock(hidden_channels*(2**(num_blocks-i)), num_layers, "up") for i in range(num_blocks)]
        )

        self.conv_out = L.Conv2d(out_channels, kernel_size=3, pad=1)

    def forward(self, x, t, c):
        t = t.astype(xp.float32) / 1000 # [0,1000] -> [0,1]
        h = self.conv_in(x)
        hs = [h] # skip connection
        for down_block, time_emb, context_emb in zip(self.down_blocks.layers, self.time_embs.layers, self.context_embs.layers):
            emb = time_emb(t) + context_emb(c) # 時刻埋め込み、ラベル埋め込み
            emb = expand_2d(emb)
            h = down_block(h + emb)
            hs.append(h) # skip connection

        h = self.mid_blocks(h)

        for up_block in self.up_blocks.layers:
            res = hs.pop()
            h = up_block(cat((h, res), axis=1)) # skip connectionを結合

        h = self.conv_out(h)
        return h


if __name__ == "__main__":
    x = xp.random.randn(1, 1, 28, 28).astype(xp.float32)
    t = xp.random.randint(0, 1000, size=(1, 1)).astype(xp.int32)
    c = xp.array([1])
    model = UNet(1, 4, 2, 2)
    model.to_gpu()
    y = model(x, t, c)
    print(y.shape)
