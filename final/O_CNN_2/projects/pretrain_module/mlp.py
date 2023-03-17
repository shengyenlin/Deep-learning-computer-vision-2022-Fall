import torch
from torch import nn
def MLP(inp_size=768, middle_size=768, out_size=96, layers=2, norm_type='batch_norm'):
    assert layers >= 1

    def submodule(inp_size, out_size):
        return nn.Sequential(
            nn.Linear(inp_size, out_size, bias=False),
            nn.BatchNorm1d(out_size)
            if norm_type == 'batch_norm' else nn.LayerNorm(out_size),
        )

    mlp = nn.Sequential()
    for i in range(layers - 1):
        mlp.add_module(f'mlp_{i}', submodule(inp_size, middle_size))
        inp_size = middle_size
        mlp.add_module(f'relu_{i}', nn.ReLU(inplace=True))
    mlp.add_module(f'mlp_last', submodule(middle_size, out_size))
    
    return mlp