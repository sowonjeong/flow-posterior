import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch import nn, autograd
import matplotlib.pyplot as plt
import time 
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath('..'))
from models.icnn import *
from utils.source_generator import *
    
class Flow(nn.Module):
    def __init__(self, x_dim: int = 2, theta_dim: int = 2, h: int = 64, n_layers = 4, g_convex = True):
        super().__init__()
        self.x_dim = x_dim
        self.theta_dim = theta_dim
        self.g_convex = g_convex

        # Network for F(x, t)
        self.net_F = nn.Sequential(
            self.make_MLP_block(x_dim + 1, h),
            self.make_MLP_block(h, h * 2),
            self.make_MLP_block(h * 2, x_dim, final_layer=True),
        )

        # Network for G(F(x, t), theta, t)
        if g_convex:
            self.net_G = PICNN(input_x_dim = theta_dim,
                        input_y_dim = x_dim + 1,
                        feature_dim = h,
                        feature_y_dim = h,
                        num_layers = n_layers, 
                        out_dim = 1)
        else:
            self.net_G = nn.Sequential(
                self.make_MLP_block(theta_dim + x_dim + 1, h),
                self.make_MLP_block(h, h * 2),
                self.make_MLP_block(h * 2, h * 4),
                self.make_MLP_block(h * 4, theta_dim, final_layer=True),
            )

    def make_MLP_block(self, input_channels, output_channels, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.BatchNorm1d(output_channels),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ELU()
            )
        else:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
            )
    
    def grad(self, theta, F_t, t) -> torch.Tensor:
        theta.requires_grad_(True)
        out = self.net_G(theta, torch.cat((F_t, t), -1))
        grad = autograd.grad(torch.sum(out), theta, retain_graph=True, create_graph=True)[0]
        return grad

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        y = x_t[:, 0:self.x_dim]
        theta = x_t[:, self.x_dim:]
        F_t = self.net_F(torch.cat((y, t), -1))
        if self.g_convex:
            G_t = self.grad(theta, F_t, t)
        else:    
            G_t = self.net_G(torch.cat((F_t, theta, t),-1))
        return torch.cat([F_t, G_t], dim=1)
    
    # fast helper for freezing
    def toggle_requires_grad(self, part, requires_grad: bool):
        for p in getattr(self, part).parameters():
            p.requires_grad_(requires_grad)

    def ode_rhs(self, t, x_t):
        t = t.expand(x_t.shape[0], 1)  
        return self.forward(x_t, t)

    def solve_ode(self, x_t: torch.Tensor, t_start: float, t_end: float, step_size = 100, method= 'dopri5', rtol=1e-7, atol=1e-5):
        t_span = torch.linspace(t_start, t_end, step_size)
        return odeint(self.ode_rhs, x_t, t_span, method=method,rtol = rtol,atol = atol)[-1]
    
    def solve_ode_F_marginal(self, y: torch.Tensor,
                             t_start: float, t_end: float, step_size = 100, method= 'dopri5', rtol=1e-7, atol=1e-5) -> torch.Tensor:
        t_span = torch.linspace(t_start, t_end, step_size)
        F_final = odeint(
            lambda t, y_: self.net_F(torch.cat((y_, t.repeat(y.shape[0], 1)), -1)), 
            y0=y, 
            t=t_span, 
            method=method,
            rtol = rtol,
            atol = atol
        )[-1]
        return F_final
    
    def sample_posterior(self, y, source, step_size = 5, method = 'dopri5'):
        n_samples = source.shape[0]
        y0 = self.solve_ode_F_marginal(y, t_start = 1, t_end = 0, method = 'rk4')
        out = self.solve_ode(torch.cat((y0.repeat(n_samples,1), source), -1), t_start = 0, t_end = 1, step_size = step_size, method = method)
        return out[:, self.x_dim:]
   
    def train_model(self, sampler, n_steps=10000, batch_size=256, lr=1e-3,  device="cuda", source = "gaussian", path_fn=None):
        self.to(device)
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=500, verbose=True, min_lr=1e-5)
        loss_fn = nn.MSELoss()
        loss_history = []
        start = time.time()
        
        for step in tqdm(range(n_steps)):
            # Sample batch
            param, data = sampler(n=self.x_dim, batch_size=batch_size, as_torch=True, device=device)
            x1 = torch.cat([data, param], dim=1)
            x0 = source_generator(x1, type = source)
            
            if path_fn is None:
                t = torch.rand(len(x1), 1, device=device)
                x_t = (1 - t) * x0 + t * x1
                dx_t = x1 - x0
            else:
                t, x_t, dx_t =  path_fn.sample_location_and_conditional_flow(x0, x1)

            optimizer.zero_grad()
            loss = loss_fn(self(x_t, t.view(-1,1)), dx_t)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            loss_history.append(loss.item())
            
            if step % 1000 == 0:
                end = time.time()
                print(f"{step+1}: loss {loss.item():0.3f} time {(end - start):0.2f}")
                start = end

        plt.plot(loss_history, label='Total Loss')
        plt.legend()
        return loss_history