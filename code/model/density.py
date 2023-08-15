#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch.nn as nn
import torch
import numpy as np


class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None, **kwargs):
        return self.density_func(sdf, beta=beta, **kwargs)


class LaplaceDensity(Density):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    def __init__(self, params_init={}, beta_min=0.0001, step_func_sigma=10):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min).cuda()
        sigma = 1 / np.pi / step_func_sigma
        # self.heaviside = lambda x: ( 0.5 + (x/sigma).atan() / np.pi )
        self.heaviside = lambda x: x.sign()/2 + 0.5

    def density_func(self, sdf, beta=None, n_d_r=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        out = alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

        if n_d_r is not None:
            out = torch.where(sdf > -0.1, out * self.heaviside(n_d_r.reshape(out.shape)), out)  # NOTE hard code

        return out

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

    def abs_inv(self, abs_sdf, beta=None):
        if beta is None:
            beta = self.get_beta()
        out = -(abs_sdf * beta / 0.5).log() * beta
        return out


class AbsDensity(Density):  # like NeRF++
    def density_func(self, sdf, beta=None):
        return torch.abs(sdf)


class SimpleDensity(Density):  # like NeRF
    def __init__(self, params_init={}, noise_std=1.0):
        super().__init__(params_init=params_init)
        self.noise_std = noise_std

    def density_func(self, sdf, beta=None):
        if self.training and self.noise_std > 0.0:
            noise = torch.randn(sdf.shape).cuda() * self.noise_std
            sdf = sdf + noise
        return torch.relu(sdf)
