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

```python
rng = np.random.default_rng(12345)
w1 = torch.from_numpy(rng.normal(size=(8, 1)).astype(np.float32))
w2 = torch.from_numpy(rng.normal(size=(8, 8)).astype(np.float32))
w3 = torch.from_numpy(rng.normal(size=(1, 8)).astype(np.float32))
print(w1)
coupling_layer.scaling[0].weight.data = w1
coupling_layer.scaling[2].weight.data = w2
coupling_layer.scaling[4].weight.data = w3
coupling_layer.scaling[0].bias.data = -torch.ones(8, dtype=torch.float32)
coupling_layer.scaling[2].bias.data = -torch.ones(8, dtype=torch.float32)
coupling_layer.scaling[4].bias.data = -torch.ones(1, dtype=torch.float32)
```

coupling layer の動作確認をしてみる:

```python
rng = np.random.default_rng(12345)

batch_size = 1024
# generate sample
x1 = 2 * torch.from_numpy(rng.random(size=batch_size).astype(np.float32)) - 1
x2 = 2 * torch.from_numpy(rng.random(size=batch_size).astype(np.float32)) - 1
print(x1)
# forward x' = g(x)
gx1, gx2, logJ = coupling_layer.forward(x1, x2)
print(gx1.detach().numpy())
print(gx2.detach().numpy())
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

作用を離散化したものを計算したい.

\begin{equation}
\begin{split}
S^E_{\text{cont}}[\phi] &= \int d^2\vec{x} ~ (\partial_\mu \phi(\vec{x}))^2 + m^2 \phi(\vec{x})^2 + \lambda \phi(\vec{x})^4 \\
\rightarrow S^E_{\text{latt}}(\phi) &= \sum_{\vec{n}} \phi(\vec{n}) \left[ \sum_{\mu \in \{1,2\}} 2\phi(\vec{n}) - \phi(\vec{n}+\hat{\mu}) - \phi(\vec{n}-\hat{\mu}) \right] + m^2 \phi(\vec{n})^2 + \lambda \phi(\vec{n})^4
\end{split}
\end{equation}

```python
Nd = len(cfgs.shape) - 1
dims = range(1, Nd + 1)
dims
```

```python
class ScalarPhi4Action:
    def __init__(self, M2, lam):
        self.M2 = M2
        self.lam = lam

    def __call__(self, cfgs):
        """
        cfgs.shape == (batch_size, L, L)
        """
        action_density = self.M2 * cfgs ** 2 + self.lam * cfgs ** 4
        dims = range(1, cfgs.ndim)
        for mu in dims:
            action_density += 2 * cfgs ** 2
            action_density -= cfgs * torch.roll(cfgs, -1, mu)
            action_density -= cfgs * torch.roll(cfgs, 1, mu)
        return torch.sum(action_density, dim=tuple(dims))
    
print("Actions for example configs:", ScalarPhi4Action(M2=1.0, lam=1.0)(cfgs))
print("Actions for example configs:", ScalarPhi4Action(M2=-4.0, lam=8)(cfgs).detach().numpy())
```

```python
class ScalarPhi4Action:
    def __init__(self, M2, lam):
        self.M2 = M2
        self.lam = lam
    def __call__(self, cfgs):
        # potential term
        action_density = self.M2*cfgs**2 + self.lam*cfgs**4
        # kinetic term (discrete Laplacian)
        Nd = len(cfgs.shape)-1
        dims = range(1,Nd+1)
        for mu in dims:
            action_density += 2*cfgs**2
            action_density -= cfgs*torch.roll(cfgs, -1, mu)
            action_density -= cfgs*torch.roll(cfgs, 1, mu)
        return torch.sum(action_density, dim=tuple(dims))

print("Actions for example configs:", ScalarPhi4Action(M2=1.0, lam=1.0)(cfgs))
```

```python
import numpy as np
import torch
```

```python
M2 = -4.0
lam = 8.0
lattice_shape = (8, 8)
phi4_action = ScalarPhi4Action(M2=M2, lam=lam)
```

```python
torch.manual_seed(1234)
np.random.seed(1234)
```

```python
torch.manual_seed(12345)
def grab(var):
    return var.detach().cpu().numpy()

prior = SimpleNormal(torch.zeros(lattice_shape), torch.ones(lattice_shape))
torch_z = prior.sample_n(1)
assert np.allclose(
    torch_z.detach().numpy(),
    np.array(
        [
            [
                [
                    -1.4798077e00,
                    4.8730600e-01,
                    -3.0127938e00,
                    4.4385520e-01,
                    3.5975620e-01,
                    -1.2348130e-02,
                    2.1852244e-01,
                    -1.2814684e00,
                ],
                [
                    2.4111958e00,
                    1.9991280e00,
                    7.8478569e-01,
                    -1.0194712e00,
                    -2.1057579e-01,
                    6.2683845e-01,
                    9.3176258e-01,
                    1.8675473e-01,
                ],
                [
                    9.5892775e-01,
                    -1.1370966e00,
                    1.3050791e-03,
                    1.3174164e00,
                    -1.0148309e00,
                    -5.4285949e-01,
                    4.3074253e-01,
                    -1.9256700e00,
                ],
                [
                    1.2755769e00,
                    -1.1315589e00,
                    8.6800182e-01,
                    7.0788389e-01,
                    2.0584559e-01,
                    -9.3001032e-01,
                    1.1424503e-01,
                    -4.4502813e-01,
                ],
                [
                    -8.5305727e-01,
                    -8.4074384e-01,
                    -3.9632735e-01,
                    -2.5913042e-01,
                    -6.7731214e-01,
                    7.0912451e-02,
                    -4.5837721e-01,
                    1.6847131e00,
                ],
                [
                    1.4235240e-01,
                    6.4272028e-01,
                    -7.0122153e-01,
                    1.0413089e00,
                    -2.3503485e00,
                    1.8441176e-01,
                    6.3359553e-01,
                    -1.5297261e00,
                ],
                [
                    -1.2016140e00,
                    6.7755446e-02,
                    4.5683756e-02,
                    4.6942052e-01,
                    7.8615564e-01,
                    -1.1713554e00,
                    6.3933975e-01,
                    6.0147840e-01,
                ],
                [
                    3.8572937e-01,
                    -4.6499664e-01,
                    -1.7946686e00,
                    3.8316032e-01,
                    -8.0198818e-01,
                    -9.1925912e-02,
                    -4.9732769e-01,
                    -1.4870524e00,
                ],
            ]
        ]
    ),
)

torch.manual_seed(1)
torch_z = prior.sample_n(1024)
z = grab(torch_z)
print(z[0,0,0])
print(z[1,4,5])
print(z[10,3,2])

print(f"z.shape = {z.shape}")

fig, ax = plt.subplots(4, 4, dpi=125, figsize=(4, 4))
for i in range(4):
    for j in range(4):
        ind = i * 4 + j
        ax[i, j].imshow(np.tanh(z[ind]), vmin=-1, vmax=1, cmap="viridis")
        ax[i, j].axes.xaxis.set_visible(False)
        ax[i, j].axes.yaxis.set_visible(False)
plt.show()
```

```python
fig, ax = plt.subplots(4,4, figsize=(4,4))
for x1 in range(2):
    for y1 in range(2):
        i1 = x1*2 + y1
        for x2 in range(2):
            for y2 in range(2):
                i2 = x2*2 + y2
                ax[i1,i2].hist2d(z[:,x1,y1], z[:,x2,y2], range=[[-3,3],[-3,3]], bins=20)
                ax[i1,i2].set_xticks([])
                ax[i1,i2].set_yticks([])
                if i1 == 3:
                    ax[i1,i2].set_xlabel(rf'$\phi({x2},{y2})$')
                if i2 == 0:
                    ax[i1,i2].set_ylabel(rf'$\phi({x1},{y1})$')
fig.suptitle("Correlations in Various Lattice Sites")
plt.show()
```

We can also investigate the correlation between the "effective action" defining the model distribution (here, $-\log{r}(z)$) and the true action ($S(z)$). If the prior distribution was already a good model for the true distribution, all samples should have identical action under the prior and true distributions, up to an overall shift. In other words, these should have linear correlation with slope $1$.

```python
S_eff = -grab(prior.log_prob(torch_z))
print(S_eff)
S = grab(phi4_action(torch_z))
print(S)
fit_b = np.mean(S) - np.mean(S_eff)
print(f'slope 1 linear regression S = -logr + {fit_b:.4f}')
fig, ax = plt.subplots(1,1, dpi=125, figsize=(4,4))
ax.hist2d(S_eff, S, bins=20, range=[[-800, 800], [200,1800]])
xs = np.linspace(-800, 800, num=4, endpoint=True)
ax.plot(xs, xs + fit_b, ':', color='w', label='slope 1 fit')
ax.set_xlabel(r'$S_{\mathrm{eff}} \equiv -\log~r(z)$')
ax.set_ylabel(r'$S(z)$')
ax.set_aspect('equal')
plt.legend(prop={'size': 6})
plt.show()
```

# Affine coupling layers

次のような変換でも良い. `s`, `t` は後で実装するように，ニューラルネットワークになる.
\begin{equation}
    g(\phi_1, \phi_2) = \left(e^{s(\phi_2)} \phi_1 + t(\phi_2),  \phi_2\right),
\end{equation}
with inverse given by:
\begin{equation} g^{-1}(\phi_1', \phi_2') =  \left((\phi_1' - t(\phi_2')) e^{-s(\phi_2')}, \phi_2'\right)  \end{equation}

$\phi$ を $\phi = (\phi_1, \phi_2)$ の分解の仕方は任意性があるが, バイナリーチェッカーボードパターンを作り, 1 を持つ部分が frozen となるように定義する.


## チェッカーボックス

```python
def make_checker_mask(shape, parity):
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker

assert torch.all(make_checker_mask(lattice_shape, 0) == torch.from_numpy(np.array(
        [
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0]
        ]
        )
    )
)
```

```python
class AffineCoupling(torch.nn.Module):
    def __init__(self, net,*,mask_shape, mask_parity):
        super(AffineCoupling, self).__init__()
        self.mask = make_checker_mask(mask_shape, mask_parity)
        self.flip_mask = 1- make_checker_mask(mask_shape, mask_parity)

        self.net = net
    def forward(self, x):
        pass
    def reverse(self, fx):
        pass
```
