---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Circular のお勉強

padding_mode の振る舞いを調査


# Import modules

```python
import numpy as np
import torch
import torch.nn as nn
```

# Conv2d インスタンスを生成

```python
c = nn.Conv2d(1,2,kernel_size=(3,3), stride=1, padding=(1,1), padding_mode="circular")
```

# 初期値を強制挿入

- 重み（フィルタ）は全て 1 を成分とするものにする.
- バイアスは簡単のために全部ゼロとする.

```python
c.weight.data = torch.ones(c.weight.shape)
c.bias.data = torch.zeros(c.bias.shape)
```

# 入力値の作成

- 形状が `5x5` の簡単なものについて試す.

```python
x = np.arange(25).reshape(5, 5).astype(np.float32)
t = torch.from_numpy(x).view(1,1,*x.shape)
t
```

- 実行結果

```python
c(t)
```

# `x` を pad する

```python
x
```

`mode = "wrap"` によって周期境界条件を作ることができる.

```python
x_pad = np.pad(x, (1,1), mode="wrap")
x_pad
```

```python
print(np.sum(x_pad[0:3, 0:3]))
print(np.sum(x_pad[0:3, 0+1:3+1]))
print(np.sum(x_pad[0:3, 0+2:3+2]))
print(np.sum(x_pad[0:3, 0+3:3+3]))
print(np.sum(x_pad[0:3, 0+4:3+4]))
```

これらの値が `c(t)[0,0,0,:]` と一致していることを確認せよ.

```python
c(t)[0,0,0,:]
```
