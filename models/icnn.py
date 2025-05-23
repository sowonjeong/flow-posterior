import torch
from torch import nn
import torch.nn.functional as F

"""
The Following Code is largely inspired
by the paper Input Convex Neural Network
and the the FICNN implementation is borrowed 
from CP-Flow.

Input Convex Neural Networks (Amos et al., 2017)
https://arxiv.org/abs/1609.07152
CP-Flow Repository (Huang et al., 2021)
https://github.com/CW-Huang/CP-Flow
"""


class FICNN(nn.Module):

    def __init__(self, input_dim: int, feature_dim: int, out_dim: int, num_layers: int,
                 act=F.softplus, nonneg=F.relu, reparam=True) -> None:
        """
        Implementation of the Fully Input Convex Neural Networks (Amos et al., 2017)
        non-negative weights restricted using reparameterization or non-negative clipping
        after optimizer step

        :param input_dim: input data dimension
        :param feature_dim: intermediate feature dimension
        :param out_dim: output dimension
        :param num_layers: number of layers
        :param act: choice of activation for z path
        :param nonneg: activation function for non-negative weights
        :param reparam: handling of non-negative constraints
        """

        super(FICNN, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # feature path
        Lz = list()
        Lz.append(nn.Linear(input_dim, feature_dim))
        for k in range(num_layers - 1):
            Lzi = nn.Linear(feature_dim, feature_dim // 2, bias=True)
            # positive weights
            with torch.no_grad():
                Lzi.weight.data = nonneg(Lzi.weight)
            Lz.append(Lzi)
        Lzk = nn.Linear(feature_dim, 1, bias=False)
        with torch.no_grad():
            Lzk.weight.data = nonneg(Lzk.weight)
        Lz.append(Lzk)
        self.Lz = torch.nn.ModuleList(Lz)

        # passthrough layers
        Ly = list()
        for k in range(num_layers - 1):
            Ly.append(nn.Linear(input_dim, feature_dim // 2))
        Ly.append(nn.Linear(input_dim, 1, bias=False))
        self.Ly = torch.nn.ModuleList(Ly)

        # augmentation layers
        Ly2 = list()
        for k in range(num_layers - 1):
            Ly2.append(nn.Linear(input_dim, feature_dim // 2))
        self.Ly2 = torch.nn.ModuleList(Ly2)

        self.act = act
        self.nonneg = nonneg
        self.reparam = reparam

    def forward(self, x):

        z = self.act(self.Lz[0](x))
        for lz, ly, ly2 in zip(self.Lz[1:-1], self.Ly[:-1], self.Ly2[:]):
            down = 1 / len(z.t())
            if self.reparam is True:
                # reparameterization
                Lzi_pos = self.nonneg(lz.weight)
                z = self.act(F.linear(z, Lzi_pos, lz.bias) * down + ly(x))
            else:
                z = self.act(lz(z) * down + ly(x))
            aug = ly2(x)
            aug = self.act(aug)
            z = torch.cat([z, aug], -1)
        down = 1 / len(z.t())
        if self.reparam is True:
            Lzk_pos = self.nonneg(self.Lz[-1].weight)
            z = F.linear(z, Lzk_pos, self.Lz[-1].bias) * down + self.Ly[-1](x)
        else:
            z = self.Lz[-1](z) * down + self.Ly[-1](x)
        return z
    
    def check_convexity(self, alpha=0.5):
        theta1 = torch.randn(100, self.input_dim)
        theta2 = torch.randn(100, self.input_dim)
        f1 = self(theta1) 
        f2 = self(theta2)
        f_mix = self(alpha * theta1 + (1 - alpha) * theta2)

        lhs = f_mix
        rhs = alpha * f1 + (1 - alpha) * f2
        return lhs, rhs, lhs <= rhs


class PICNN(nn.Module):

    def __init__(self, input_x_dim: int, input_y_dim: int, feature_dim: int, feature_y_dim: int, out_dim: int,
                 num_layers: int, act=F.softplus, act_v=nn.ELU(), nonneg=F.relu, reparam=True) -> None:
        """
        Implementation of the Partially Input Convex Neural Networks (Amos et al., 2017)
        non-negative weights restricted using reparameterization or non-negative clipping
        after optimizer step

        :param input_x_dim: input data convex dimension
        :param input_y_dim: input data non-convex dimension
        :param feature_dim: intermediate feature dimension
        :param feature_y_dim: intermediate context dimension
        :param out_dim: output dimension
        :param num_layers: number of layers
        :param act: choice of activation for w path
        :param act_v: choice of activation in v path and for v in hadamard product with x
        :param nonneg: activation function for non-negative weights
        :param reparam: handling of non-negative constraints
        """

        super(PICNN, self).__init__()
        self.input_x_dim = input_x_dim
        self.input_y_dim = input_y_dim
        self.feature_dim = feature_dim
        self.feature_y_dim = feature_y_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # forward path for v(y)
        Lv = list()
        Lv.append(nn.Linear(input_y_dim, feature_y_dim, bias=True))
        for k in range(num_layers - 1):
            Lv.append(nn.Linear(feature_y_dim, feature_y_dim, bias=True))
        self.Lv = nn.ModuleList(Lv)

        # forward path for v into w
        Lvw = list()
        Lvw.append(nn.Linear(input_y_dim, feature_dim, bias=False))
        for k in range(num_layers - 1):
            Lvw.append(nn.Linear(feature_y_dim, feature_dim, bias=False))
        Lvw.append(nn.Linear(feature_y_dim, out_dim, bias=False))
        self.Lvw = nn.ModuleList(Lvw)

        # forward path for w product, positive weights
        Lw = list()
        Lw0 = nn.Linear(input_x_dim, feature_dim, bias=True)
        # positive weights
        with torch.no_grad():
            Lw0.weight.data = nonneg(Lw0.weight)
        Lw.append(Lw0)

        for k in range(num_layers - 1):
            Lwk = nn.Linear(feature_dim, feature_dim, bias=True)
            with torch.no_grad():
                Lwk.weight.data = nonneg(Lwk.weight)
            Lw.append(Lwk)

        LwK = nn.Linear(feature_dim, out_dim, bias=True)
        with torch.no_grad():
            LwK.weight.data = nonneg(LwK.weight)
        Lw.append(LwK)
        self.Lw = nn.ModuleList(Lw)

        # context path for v times w
        Lwv = list()
        Lwv.append(nn.Linear(input_y_dim, input_x_dim, bias=True))
        for k in range(num_layers):
            Lwv.append(nn.Linear(feature_y_dim, feature_dim, bias=True))
        self.Lwv = nn.ModuleList(Lwv)

        # context path for v times x
        Lxv = list()
        for k in range(num_layers):
            Lxv.append(nn.Linear(feature_y_dim, input_x_dim, bias=True))
        self.Lxv = nn.ModuleList(Lxv)

        # forward path for x product
        Lx = list()
        for k in range(num_layers - 1):
            Lx.append(nn.Linear(input_x_dim, feature_dim, bias=False))
        Lx.append(nn.Linear(input_x_dim, out_dim, bias=False))
        self.Lx = nn.ModuleList(Lx)

        self.act = act
        self.act_v = act_v
        self.nonneg = nonneg
        self.reparam = reparam

    def forward(self, in_x, in_y):

        # first layer activation
        v = in_y
        w0_prod = torch.mul(in_x, F.relu(self.Lwv[0](v)))    # relu for non-negativity
        if self.reparam is True:
            # reparameterization
            Lw0_pos = self.nonneg(self.Lw[0].weight)
            w = self.act(F.linear(w0_prod, Lw0_pos, self.Lw[0].bias) + self.Lvw[0](v))
        else:
            w = self.act(self.Lw[0](w0_prod) + self.Lvw[0](v))

        # zip the models
        NN_zip = zip(self.Lv[:-1], self.Lvw[1:-1], self.Lw[1:-1],
                     self.Lwv[1:-1], self.Lxv[:-1], self.Lx[:-1])

        # intermediate layers activations
        for lv, lvw, lw, lwv, lxv, lx in NN_zip:
            down = 1 / len(w.t())
            v = self.act_v(lv(v))
            wk_prod = torch.mul(w, F.relu(lwv(v)))
            xk_prod = torch.mul(in_x, lxv(v))
            if self.reparam is True:
                Lwk_pos = self.nonneg(lw.weight)
                w = self.act(F.linear(wk_prod, Lwk_pos, lw.bias) * down + lx(xk_prod) + lvw(v))
            else:
                w = self.act(lw(wk_prod) * down + lx(xk_prod) + lvw(v))

        # last layer activation
        down = 1 / len(w.t())
        vK = self.act_v(self.Lv[-1](v))
        wK_prod = torch.mul(w, F.relu(self.Lwv[-1](vK)))
        xK_prod = torch.mul(in_x, self.Lxv[-1](vK))
        if self.reparam is True:
            LwK_pos = self.nonneg(self.Lw[-1].weight)
            w = F.linear(wK_prod, LwK_pos, self.Lw[-1].bias) * down + self.Lx[-1](xK_prod) + self.Lvw[-1](vK)
        else:
            w = self.Lw[-1](wK_prod) * down + self.Lx[-1](xK_prod) + self.Lvw[-1](vK)
        return w
    def check_convexity(self,alpha=0.5):
        theta1 = torch.randn(100, self.input_x_dim)
        theta2 = torch.randn(100, self.input_x_dim)
        y_fixed = torch.randn(100, self.input_y_dim)
        f1 = self(theta1, y_fixed)
        f2 = self(theta2, y_fixed)
        f_mix = self(alpha * theta1 + (1 - alpha) * theta2, y_fixed)

        lhs = f_mix
        rhs = alpha * f1 + (1 - alpha) * f2
        return lhs, rhs, lhs <= rhs


