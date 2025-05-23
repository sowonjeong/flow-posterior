import torch 

def spherical_uniform(x_1, uniform = False, tau = None):
    if uniform: 
        x_0 = 2 * torch.rand_like(x_1) - 1
    else:
        x_0 = torch.randn_like(x_1)
    x_0 = x_0 / x_0.norm(dim=1, keepdim=True)
    if tau is None:
        r = torch.rand((x_1.shape[0],1)) 
        out = r * x_0
    else:
        tau = torch.Tensor([tau]).view(-1,1)
        out = tau.repeat(x_1.shape[0],1) * x_0
    return out

def source_generator(x_1, type = "gaussian"):
    if type == "gaussian":
        return torch.randn_like(x_1)
    elif type == "uniform":
        return torch.rand_like(x_1)
    elif type == "spherical_gaussian":
        return spherical_uniform(x_1, uniform=False)
    elif type == "spherical_uniform":
        return spherical_uniform(x_1, uniform = True)
    elif type == 'binary':
        return torch.randint(0, 2, size=x_1.shape).float() * 2 - 1  # {-1, +1}
    elif type == 'laplace':
        return torch.distributions.Laplace(0, 1).sample(x_1.shape)
    else:
        raise ValueError(f"Unknown source type: {type}")