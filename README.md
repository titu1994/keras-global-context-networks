# Keras Global Context Attention Blocks

Keras implementation of the Global Context block from the paper [GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond](https://arxiv.org/abs/1904.11492).

Supports Conv1D, Conv2D and Conv3D directly with no modifications.

<img src="https://github.com/titu1994/keras-global-context-networks/blob/master/images/gc.PNG?raw=true" height=100% width=100%>

# Usage

Import `global_context_block` from `gc.py` and provide it a tensor as input.

```python
from gc import global_context_block

ip = Input(...)
x = ConvND(...)(ip)

# apply Global Context
x = global_context_block(x, reduction_ratio=16, transform_activation='linear')
...
```

# Parameters

There are just two parameters to manage : 
```
 - reduction_ratio: The ratio to scale the transform block.
 - transform_activation: The activation function prior to addition of the input with the context.
                         The paper uses no activation, but `sigmoid` may do better.
```

# Requirements
  - Keras 2.2.4+
  - Tensorflow (1.13+) or CNTK
