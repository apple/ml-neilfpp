#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import numpy as np
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as Func

from utils import geometry

EPS = 1e-7

def fibonacci_sphere_sampling(normals, sample_num, random_rotate=True):

    N = normals.shape[0]
    delta = np.pi * (3.0 - np.sqrt(5.0))

    # fibonacci sphere sample around z axis 
    idx = torch.arange(sample_num).cuda().float().unsqueeze(0).repeat([N, 1])   # [N, S]
    z = 1 - 2 * idx / (2 * sample_num - 1)                                      # [N, S]
    rad = torch.sqrt(1 - z ** 2)                                                # [N, S]
    theta = delta * idx                                                         # [N, S]        
    if random_rotate:
        z_jitter_rad = 1 / (2 * sample_num - 1)
        z = (torch.rand(N, 1).cuda() * 2 - 1) * z_jitter_rad + z
        z = 1 - (1 - z.abs()).abs()
        theta = torch.rand(N, 1).cuda() * 2 * np.pi + theta                     # [N, S]
    y = torch.cos(theta) * rad                                                  # [N, S]
    x = torch.sin(theta) * rad                                                  # [N, S]
    z_samples = torch.stack([x, y, z], axis=-1).permute([0, 2, 1])              # [N, 3, S]

    # rotate to normal
    z_vector = torch.zeros_like(normals)                                        # [N, 3]
    z_vector[:, 2] = 1                                                          # [N, 3]
    rotation_matrix = geometry.rotation_between_vectors(z_vector, normals)      # [N, 3, 3]
    incident_dirs = rotation_matrix @ z_samples                                 # [N, 3, S]
    incident_dirs = Func.normalize(incident_dirs, dim=1).permute([0, 2, 1])     # [N, S, 3]
    incident_areas = torch.ones_like(incident_dirs)[..., 0:1] * 2 * np.pi       # [N, S, 1]

    return incident_dirs, incident_areas


class UniformSampler(nn.Module):
    def __init__(self, scene_bounding_sphere, near, N_samples, far=-1):
        super().__init__()
        self.near = near
        self.far = 2.0 * scene_bounding_sphere if far == -1 else far  # default far is 2*R
        self.N_samples = N_samples
        self.scene_bounding_sphere = scene_bounding_sphere

    def get_z_vals(self, ray_dirs, cam_loc, density_func, im_func, **kwargs):
        take_sphere_intersection = kwargs.get('take_sphere_intersection', False)

        if not take_sphere_intersection:
            near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0], 1).cuda()
        else:
            sphere_intersections = geometry.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)
            near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()
            far = sphere_intersections[:,1:]

        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()
        z_vals = near * (1. - t_vals) + far * (t_vals)

        if self.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).cuda()

            z_vals = lower + (upper - lower) * t_rand

        return z_vals


class ErrorBoundSampler(nn.Module):
    def __init__(self, scene_bounding_sphere, near, N_samples, N_samples_eval, N_samples_extra,
                 eps, beta_iters, max_total_iters,
                 inverse_sphere_bg=False, N_samples_inverse_sphere=0, add_tiny=1e-5):
        super().__init__()
        self.near = near
        self.far = 2.0 * scene_bounding_sphere
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.uniform_sampler = UniformSampler(scene_bounding_sphere, near, N_samples_eval)

        self.N_samples_extra = N_samples_extra

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny

        self.inverse_sphere_bg = inverse_sphere_bg
        if inverse_sphere_bg:  # TODO check when used
            self.inverse_sphere_sampler = UniformSampler(1.0, 0.0, N_samples_inverse_sphere, far=1.0)

    def get_z_vals(self, ray_dirs, cam_loc, density_func, im_func, **kwargs):
        normal_consistency = kwargs.get('normal_consistency', False)
        take_sphere_intersection = kwargs.get('take_sphere_intersection', False)

        beta0 = density_func.get_beta().detach()

        # Start with uniform sampling
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, density_func, im_func, take_sphere_intersection=take_sphere_intersection or self.inverse_sphere_bg)
        samples, samples_idx, sdf = z_vals, None, None

        # Get maximum beta from the upper bound (Lemma 2)
        beta = self._get_max_beta(z_vals)

        total_iters, not_converge = 0, True

        # Algorithm 1
        while True:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)

            # Calculating the SDF only for the new sampled points
            sdf, n_d_r = self._calc_sdf_for_new_samples(points_flat, ray_dirs, z_vals, samples, samples_idx, sdf, im_func, normal_consistency)

            # Calculating the bound d* (Theorem 1)
            dists, d_star = self._calc_bound(z_vals, sdf)

            # Updating beta using line search
            beta = self._update_beta(beta, z_vals, dists, sdf, n_d_r, density_func, beta0, d_star)

            # Upsample more points
            transmittance, weights = self._get_weights(z_vals, dists, sdf, n_d_r, density_func, beta)

            #  Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0
            not_finish = not_converge and total_iters < self.max_total_iters

            if not_finish:
                # Sample more points proportional to the current error bound
                bins, cdf, N = self._sample_from_err_bound(z_vals, dists, transmittance, beta, d_star)
            else:
                # Sample the final sample set to be used in the volume rendering integral
                bins, cdf, N = self._sample_from_weights(z_vals, weights)

            # Invert CDF
            samples = self._invert_cdf(N, bins, cdf, uniform=not_finish or not self.training)

            # Adding samples if we not converged
            if not_finish:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)
            
            if not not_finish: break

        z_samples = samples
        z_vals_extra = self._get_extra_samples(ray_dirs, cam_loc, z_vals, take_sphere_intersection)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        # add some of the near surface points
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],)).cuda()
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))

        if self.inverse_sphere_bg:
            z_vals_inverse_sphere = self.inverse_sphere_sampler.get_z_vals(ray_dirs, cam_loc, density_func, im_func, take_sphere_intersection=False)
            z_vals_inverse_sphere = z_vals_inverse_sphere * (1./self.scene_bounding_sphere)
            z_vals = (z_vals, z_vals_inverse_sphere)

        return z_vals, z_samples_eik

    def _get_error_bound(self, beta, density_func, sdf, z_vals, dists, d_star, n_d_r=None):

        density = density_func(sdf.reshape(z_vals.shape), beta=beta, n_d_r=n_d_r)
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), dists * density[:, :-1]], dim=-1)
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists ** 2.) / (4 * beta ** 2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(-integral_estimation[:, :-1])

        return bound_opacity.max(-1)[0]
    
    def _get_max_beta(self, z_vals):

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (dists ** 2.).sum(-1)
        beta = torch.sqrt(bound)
        return beta
    
    def _calc_sdf_for_new_samples(self, points_flat, ray_dirs, z_vals, samples, samples_idx, sdf, im_func, normal_consistency):

        if not normal_consistency:
            with torch.no_grad():
                samples_sdf = im_func.get_sdf_vals(points_flat)
        else:
            samples_sdf, samples_grad = im_func.get_sdf_vals_grad(points_flat, grad_order=1)
            samples_ndr = - (
                ray_dirs.unsqueeze(1) * 
                Func.normalize(samples_grad.reshape(ray_dirs.shape[0], -1, 3), dim=-1)
            ).sum(-1).reshape(-1,1)

        if samples_idx is not None:
            sdf_merge = torch.cat([sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                    samples_sdf.reshape(-1, samples.shape[1])], -1)
            sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            if normal_consistency:
                ndr_merge = torch.cat([n_d_r.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                                        samples_ndr.reshape(-1, samples.shape[1])], -1)
                n_d_r = torch.gather(ndr_merge, 1, samples_idx).reshape(-1, 1)
            else:
                n_d_r = None
        else:
            sdf = samples_sdf
            n_d_r = samples_ndr if normal_consistency else None

        return sdf, n_d_r
    
    def _calc_bound(self, z_vals, sdf):

        d = sdf.reshape(z_vals.shape)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
        first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
        second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
        d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1).cuda()
        d_star[first_cond] = b[first_cond]
        d_star[second_cond] = c[second_cond]
        s = (a + b + c) / 2.0
        area_before_sqrt = s * (s - a) * (s - b) * (s - c)
        mask = ~first_cond & ~second_cond & (b + c - a > 0)
        d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
        d_star = (d[:, 1:].sign() * d[:, :-1].sign() == 1) * d_star  # Fixing the sign

        return dists, d_star

    def _update_beta(self, beta, z_vals, dists, sdf, n_d_r, density_func, beta0, d_star):

        curr_error = self._get_error_bound(beta0, density_func, sdf, z_vals, dists, d_star, n_d_r)
        beta[curr_error <= self.eps] = beta0
        beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
        for j in range(self.beta_iters):
            beta_mid = (beta_min + beta_max) / 2.
            curr_error = self._get_error_bound(beta_mid.unsqueeze(-1), density_func, sdf, z_vals, dists, d_star, n_d_r)
            beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
            beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]

        return beta_max
    
    def _get_weights(self, z_vals, dists, sdf, n_d_r, density_func, beta):

        density = density_func(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1), n_d_r=n_d_r)
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)
        alpha = 1 - torch.exp(-free_energy)
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        weights = alpha * transmittance  # probability of the ray hits something here

        return transmittance, weights
    
    def _sample_from_err_bound(self, z_vals, dists, transmittance, beta, d_star):

        bins = z_vals
        error_per_section = torch.exp(-d_star / beta.unsqueeze(-1)) * (dists ** 2.) / (4 * beta.unsqueeze(-1) ** 2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (torch.clamp(torch.exp(error_integral),max=1.e6) - 1.0) * transmittance[:,:-1]

        pdf = bound_opacity + self.add_tiny
        pdf = pdf / torch.sum(pdf, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

        return bins, cdf, self.N_samples_eval
    
    def _sample_from_weights(self, z_vals, weights):

        bins = z_vals
        pdf = weights[..., :-1]
        pdf = pdf + self.add_tiny  # prevent nans
        pdf = pdf / torch.sum(pdf, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

        return bins, cdf, self.N_samples
    
    def _invert_cdf(self, N, bins, cdf, uniform):

        if uniform:
            u = torch.linspace(0., 1., steps=N).cuda().unsqueeze(0).repeat(cdf.shape[0], 1)
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N]).cuda()
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples
    
    def _get_extra_samples(self, ray_dirs, cam_loc, z_vals, take_sphere_intersection):

        near, far = self.near * torch.ones(ray_dirs.shape[0], 1).cuda(), self.far * torch.ones(ray_dirs.shape[0],1).cuda()
        if take_sphere_intersection or self.inverse_sphere_bg: # if inverse sphere then need to add the far sphere intersection
            far = geometry.get_sphere_intersections(cam_loc, ray_dirs, r=self.scene_bounding_sphere)[:,1:]

        if self.N_samples_extra > 0:
            if self.training:
                sampling_idx = torch.randperm(z_vals.shape[1])[:self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(0, z_vals.shape[1]-1, self.N_samples_extra).long()
            z_vals_extra = torch.cat([near, far, z_vals[:,sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)

        return z_vals_extra
