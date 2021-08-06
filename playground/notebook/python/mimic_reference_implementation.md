---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Import modules

```python
import numpy as np
import torch
from matplotlib import pyplot as plt
```

# Setup constants

```python
np.random.seed(1234)
torch.manual_seed(1234)
```

複雑な形状を持つ確率分布のサンプルを生成するために比較的単純な確率分布からサンプルを生成し，変数変換で所望の分布を得る Normalizing Flows  
を効率よくするための手法を Lattice Field Theory に適用する.


Box Muller 変換では独立な一様分布 u1, u2 から2次元標準正規分布 z=(z1, z2) を作ることができる.
具体的には 

$$
\begin{aligned}
z_1 = \sqrt{-2\log u_1} \cos(2\pi u_2) \\
z_2 = \sqrt{-2\log u_2} \sin(2\pi u_2)
\end{aligned}
$$

```python
batch_size = 2 ** 14
u1 = np.random.random(size=batch_size)
u2 = np.random.random(size=batch_size)
z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
fig, ax = plt.subplots()
ax.hist2d(
    z1,
    z2,
    bins=30,
    range=[
        [-3.0, 3.0],
        [-3.0, 3.0],
    ],
)
ax.set_aspect("equal")
```

```python
class SimpleNormal:
    def __init__(self, loc, var):
        self.distribution = torch.distributions.normal.Normal(
            loc,
            var,
        )
        self.shape = loc.shape

    def log_prob(self, x):
        logp = self.distribution.log_prob(x)
        return torch.sum(logp, dim=tuple(range(1, logp.ndim)))

    def sample_n(self, batch_size):
        x = self.distribution.sample((batch_size,))
        return x
```

```python
shape = (3, 4, 5)
normal_prior = SimpleNormal(torch.zeros(shape), torch.ones(shape))
z = normal_prior.sample_n(17)
assert z.shape == (17, *shape)
assert normal_prior.log_prob(z).shape == (17,)
```

我々は所望の確率分布に従うサンプルを変数変換によって簡単な分布のサンプリングをすることに帰着させたい.
上記の例だと正規分布のサンプルを一様分布のサンプルから生成させた.

Box-Muller の変換 $(z_1, z_2) = f(u_1, u_2)$ は微分可能で, 逆変換を持っているので $z = (z_1, z_2)$ の分布 $q$ は一様分布 $r(u)=1\ \textrm{for}\ u\in [0, 1]\times [0, 1]$ と変換のヤコビ行列を用いて次のようにかける.

$$
\begin{aligned}
q(z) &= r(u)\left|\det \frac{\partial z_k}{\partial u_l}\right|^{-1} \\
     &= |2\pi / u_1|^{-1} \\
     &= \frac{1}{2\pi} \exp(-(z_1^2 + z_2^2)/2) 
\end{aligned}
$$

１行目の式は変換が逆変換を持っていることを要請する. 逆変換のヤコビ行列の行列式は元の変換のヤコビ行列の行列式の逆数として与えられることに注意する.

Box-Muller 変換の例は行列式を簡単に求めることができたが，一般の場合は必ずしもそれができるとは限らない. (仮にできたとしても行列式の計算をナイーブに要請されるものは計算コストが大きい.)

論文の手法はスケーリング変換を導入し行列式の計算が簡単になるような変換を提案している.J


- ここでスケーリング変換を導入する. $x$ は多次元配列として $x = (x_1, x_2)$ のように成分に分解できるとする.
$s$ を微分可能ぐらいの $x_2$ 上の任意の変換とする. さらに $g(x) = x' = (x_1', x_2')$ を次で定める変換だとする:

$$
\begin{aligned}
x_1' &= e^{s(x_2)} x_1 \\
x_2' &= x_2
\end{aligned}
$$

$g$ は次の変換を逆変換として持つ:

$$
\begin{aligned}
x_1 &= e^{-s(x_2)} x_1' \\
x_2 &= x_2'
\end{aligned}
$$

さらに $g$ は次のようなヤコビ行列を持つ:

$$
\frac{\partial g}{\partial x} = 
\begin{bmatrix}
\frac{\partial x'_1}{\partial x_1} && \frac{\partial x'_1}{\partial x_2} \\
0 && I \\
\end{bmatrix}
$$

$I$ は適当な次元の単位行列である. $g$ のヤコビ行列は三角行列なので行列式は対角成分の積で表現されることに注意する. それはベクトル値関数 $\exp(s(x_2))$ の各要素の積として得られる:

$$
\det\frac{\partial g}{\partial x} = \det {\textrm{diag}}(\exp(s(x_2))_1, \exp(s(x_2))_2, \dots, \exp(s(x_2))_K) = \prod_{k=1}^K \exp(s(x_2))_k
$$

つまりヤコビ行列の全体の成分を具体的に求められなくても $\exp(s(x_2))$ だけを計算することができれば良い.

さて, $s$ はどうやって作るか？それはまさにニューラルネットワークを使うのである.


上記の変換 $g$ は coupling layer と呼ばれる. coupling layer はニューラルネット $s$ を用いて計算する.

```python
class SimpleCouplingLayer(torch.nn.Module):
    def __init__(self):
        super(SimpleCouplingLayer, self).__init__()
        self.scaling = torch.nn.Sequential(
            torch.nn.Linear(1, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1),
            torch.nn.Tanh(),
        )

    def forward(self, x1, x2):
        lastdim = -1
        s = self.scaling(x2.unsqueeze(lastdim)).squeeze(lastdim)
        fx1 = torch.exp(s) * x1
        fx2 = x2
        logJ = s  # jacobian
        return fx1, fx2, logJ

    def reverse(self, fx1, fx2):
        lastdim = -1
        s = self.scaling(fx2.unsqueeze(lastdim)).squeeze(lastdim)
        x1 = torch.exp(-s) * fx1
        x2 = fx2
        logJ = -s  # jacobian
        return x1, x2, logJ


# init weights in a way that gives interesting behavior without training
def set_weights(m):
    if hasattr(m, "weight") and m.weight is not None:
        torch.nn.init.normal_(m.weight, mean=1, std=2)
    if hasattr(m, "bias") and m.bias is not None:
        m.bias.data.fill_(-1)


torch.manual_seed(1234)
np.random.seed(1234)
coupling_layer = SimpleCouplingLayer()
assert np.allclose(
    coupling_layer.scaling[0].weight.detach().numpy(),
    np.array(
        [
            [-0.9420415],
            [-0.19620287],
            [-0.48031163],
            [-0.2667173],
            [-0.88339853],
            [0.40128946],
            [-0.8964052],
            [-0.06372154],
        ]
    ),
)
coupling_layer.scaling.apply(set_weights)
assert np.allclose(
    coupling_layer.scaling[0].weight.detach().numpy(),
    np.array(
        [
            [1.0856667],
            [-0.51008666],
            [1.7997566],
            [-3.2858078],
            [2.2274332],
            [0.09974618],
            [3.1197715],
            [1.7173864],
        ]
    ),
)
```

coupling layer の動作確認をしてみる:

```python
rng = np.random.default_rng(12345)

batch_size = 1024
# generate sample
x1 = 2 * torch.from_numpy(rng.random(size=batch_size).astype(np.float32)) - 1
x2 = 2 * torch.from_numpy(rng.random(size=batch_size).astype(np.float32)) - 1
# forward x' = g(x)
gx1, gx2, logJ = coupling_layer.forward(x1, x2)
print(gx1)
print(gx2)
# reverse g^{-1}(x')
rev_x1, rev_x2, rev_logJ = coupling_layer.reverse(gx1, gx2)
print(rev_x1)
print(rev_x2)
gx1 = gx1.detach().numpy()
gx2 = gx2.detach().numpy()

rev_x1 = rev_x1.detach().numpy()
rev_x2 = rev_x2.detach().numpy()

fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True)
for a in ax:
    a.set_xlim(-1.1, 1.1)
    a.set_ylim(-1.1, 1.1)
ax[0].scatter(x1, x2)
ax[1].scatter(gx1, gx2)
ax[2].scatter(rev_x1, rev_x2)
```

<!-- #region tags=[] -->
## **Composition**

coupling layer をいくつか合成させることで所望の変数変換を得たい. つまり $g_1, g_2, \dots$, のように複数の coupling layer $g_i$ を合成させる. $J_i$ を $g_i$ のヤコビ行列の行列式とする. この時, 合成写像によって得られるヤコビ行列の行列式は $J_i$ の積によって得られるので, 確率密度関数の変換は次の式で与えられる:

\begin{equation}
\begin{split}
    q(x) &= r(z) \left| \det \frac{\partial f(z)}{\partial z} \right|^{-1} = r(z) \prod_{i} J_i^{-1}.
\end{split}
\end{equation}

実装上では $g_i$ に対して $\log(J_i)$ を保持しておいて積の代わりに和として保存しておく.
<!-- #endregion -->

```python
def apply_flow_to_prior(r, coupling_layers, *, batch_size):
    z = r.sample_n(batch_size)
    logq = r.log_prob(z)
    x1 = z[:, 0]
    x2 = z[:, 1]
    for lay in coupling_layers:
        x1, x2, logJ = lay.forward(x1, x2)
        logq = logq - logJ
    return x1, x2, logq  # 点 x における `\log(q(x))` の値を計算
```

```python
rng = np.random.default_rng(2021)

L = 8
lattice_shape = (L, L)

phi_ex1 = rng.normal(size=lattice_shape).astype(np.float32)
phi_ex2 = rng.normal(size=lattice_shape).astype(np.float32)

cfgs = torch.from_numpy(np.stack((phi_ex1, phi_ex2), axis=0))
```
