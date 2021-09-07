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

```python
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
```

```python
conv = nn.Conv2d(2, 16, (3, 3))
```

```python
conv.weight.shape
```

```python
plt.hist(conv.weight.data.numpy().flatten(), density=True)
```

```python
k = 1/(2 * 3 * 3)
```

```python
np.sqrt(k)
```

```python

```
