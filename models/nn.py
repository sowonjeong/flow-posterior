import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with optional batch normalization."""

    def __init__(self, layers, nonlinearity=nn.ELU, nonlinearity_params=None,
                 out_nonlinearity=None, out_nonlinearity_params=None, normalize=False):
        super().__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1

        self.layers = nn.ModuleList()
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))
                if nonlinearity_params is not None:
                    self.layers.append(nonlinearity(*nonlinearity_params))
                else:
                    self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            if out_nonlinearity_params is not None:
                self.layers.append(out_nonlinearity(*out_nonlinearity_params))
            else:
                self.layers.append(out_nonlinearity())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
