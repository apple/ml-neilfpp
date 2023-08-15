#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np
from skimage import measure
import trimesh
import tinycudann as tcnn

from utils import geometry, general
from .embedder import ComposedEmbedder
from .density import LaplaceDensity, AbsDensity
from .ray_sampling import ErrorBoundSampler, fibonacci_sphere_sampling

def compute_gradient(x, y):
    d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return gradients

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            sdf_bounding_sphere,
            cut_bounding_sphere,
            dims,
            aux_dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            hess_type='analytic',
            embed_config=[{'otype':'Identity'}],
            sphere_scale=1.0,
    ):
        super().__init__()
        d_in = 3
        d_out = 1

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.cut_bounding_sphere = cut_bounding_sphere
        self.sphere_scale = sphere_scale
        self.hess_type = hess_type
        dims = [d_in] + dims + [d_out]

        self.embed_fn = ComposedEmbedder(embed_config)
        dims[0] += self.embed_fn.n_feat_dims

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif self.embed_fn.n_feat_dims > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.embed_fn.n_feat_dims > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -self.embed_fn.n_feat_dims:], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            # if l == 0:
            #     param = lin.weight_g if weight_norm else lin.weight
            #     assert type(param) is nn.Parameter
            #     setattr(param, 'custom_meta', {})
            #     param.custom_meta['optim_args'] = {'weight_decay': 0.01}

            setattr(self, "lin" + str(l), lin)

        self.aux_dims = aux_dims
        if len(aux_dims) > 0:
            d_aux = sum([v[1] for v in aux_dims])
            lin = nn.Linear(dims[-2], d_aux)
            if geometric_init:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(d_aux))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            self.aux_lin = lin

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, sdf_only=False):

        input = self.embed_fn(input)
        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            if (not sdf_only) and (l == self.num_layers - 2):
                feat = x
                aux = self.aux_lin(x) if hasattr(self, 'aux_lin') else None

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)


        if sdf_only:
            return x
        else:
            return x, feat, aux

    def get_outputs(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            sdf, feature_vectors, aux = self(x)
            gradients = compute_gradient(x, sdf)
        sdf = self.clamp_sdf(x, sdf)
        if aux is not None:
            aux_names, aux_dims = zip(*self.aux_dims)
            split_aux = aux.split(aux_dims, dim=-1)
            aux = {k:v for k,v in zip(aux_names, split_aux)}
        else:
            aux = {}

        return sdf, feature_vectors, aux, gradients

    def get_sdf_vals(self, x, clamp=True):
        sdf = self(x, sdf_only=True)
        if clamp:
            sdf = self.clamp_sdf(x, sdf)
        return sdf

    def get_sdf_vals_grad(self, x, clamp=True, grad_order=0):
        grad = hess = None
        if grad_order >= 1:
            with torch.enable_grad():
                x.requires_grad_()
                sdf = self.get_sdf_vals(x, clamp=False)
                grad = compute_gradient(x, sdf)
                if grad_order >= 2:
                    if self.hess_type == 'analytic':
                        hess = torch.stack([compute_gradient(x, grad[:,i:i+1]) for i in range(grad.shape[1])], dim=1)
                    elif self.hess_type == 'numeric':
                        jitter_range = (2/4096, 2/1024)  # NOTE hard code
                        jitter_dir = torch.eye(3, dtype=x.dtype, device=x.device).unsqueeze(0)  # 133
                        jitter_step = torch.rand((x.shape[0],1,1), dtype=x.dtype, device=x.device)  # m11
                        jitter_step = jitter_step * (jitter_range[1]-jitter_range[0]) + jitter_range[0]  # TODO pos+neg?
                        jitter_points_all = x.unsqueeze(1) + jitter_dir * jitter_step  # m33
                        jitter_points_sample = torch.randint(0, 3, (jitter_points_all.shape[0],), dtype=torch.long, device=jitter_points_all.device)
                        jitter_points = jitter_points_all[torch.arange(jitter_points_all.shape[0], dtype=torch.long, device=jitter_points_all.device), jitter_points_sample]
                        jitter_output = self.get_sdf_vals(jitter_points, clamp=False)
                        jitter_grad = compute_gradient(jitter_points, jitter_output)  # m3
                        hess = (jitter_grad - grad).unsqueeze(1) / jitter_step
                    else:
                        raise NotImplementedError
                if clamp:
                    sdf = self.clamp_sdf(x, sdf)
            return [sdf, grad, hess][:grad_order+1]
        else:
            sdf = self.get_sdf_vals(x, clamp=clamp)
            return sdf
    
    def clamp_sdf(self, x, sdf):
        # outside of obj bounding sphere
        if self.cut_bounding_sphere > 0.0:
            inbound_mask = x.norm(dim=-1, keepdim=True) < self.cut_bounding_sphere
            sdf = torch.where(inbound_mask, sdf, self.cut_bounding_sphere)
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf


class ImplicitNetworkBG(nn.Module):  # TODO fused mlp
    def __init__(
            self,
            dims,
            aux_dims,
            skip_in=(),
            weight_norm=True,
            embed_config=[{'otype':'Identity'}],
    ):
        super().__init__()
        d_in = 4
        d_out = 1

        dims = [d_in] + dims + [d_out]

        self.embed_fn = ComposedEmbedder(embed_config, input_dim=d_in)
        dims[0] += self.embed_fn.n_feat_dims

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            # if l == 0:
            #     param = lin.weight_g if weight_norm else lin.weight
            #     assert type(param) is nn.Parameter
            #     setattr(param, 'custom_meta', {})
            #     param.custom_meta['optim_args'] = {'weight_decay': 0.01}

            setattr(self, "lin" + str(l), lin)

        self.aux_dims = aux_dims
        if len(aux_dims) > 0:
            d_aux = sum([v[1] for v in aux_dims])
            lin = nn.Linear(dims[-2], d_aux)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            self.aux_lin = lin

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, sdf_only=False):
        input = self.embed_fn(input)
        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            if (not sdf_only) and (l == self.num_layers - 2):
                feat = x
                aux = self.aux_lin(x) if hasattr(self, 'aux_lin') else None

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        if sdf_only:
            return x[:,:1]
        else:
            return x, feat, aux

    def get_outputs(self, x):
        sdf, feature_vectors, aux = self(x)
        if aux is not None:
            aux_names, aux_dims = zip(*self.aux_dims)
            split_aux = aux.split(aux_dims, dim=-1)
            aux = {k:v for k,v in zip(aux_names, split_aux)}
        else:
            aux = {}

        return sdf, feature_vectors, aux

    def get_sdf_vals(self, x):
        sdf = self(x, sdf_only=True)
        return sdf


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_out,
            dims,
            weight_norm=True,
            embed_config_view=[{'otype':'Identity'}],
            fused=False,
            last_act='None',
            sigmoid_output_scale=1.0,
            force_reflection=False,
    ):
        super().__init__()

        d_in_map = {
            'refnerf': 6,
            'idr': 9,
            'nerf': 3,
        }
        d_in = d_in_map[mode]

        self.mode = mode
        self.force_reflection = force_reflection
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = ComposedEmbedder(embed_config_view)
        dims[0] += self.embedview_fn.n_feat_dims

        self.fused = fused
        if fused:
            assert all([v == dims[1] for v in dims[1:-1]])
            fused_config = {
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': "None",
                'n_neurons': dims[1],
                'n_hidden_layers': len(dims) - 2,
            }
            self.fused_net = tcnn.Network(dims[0], dims[-1], fused_config)

        else:
            self.num_layers = len(dims)

            for l in range(0, self.num_layers - 1):
                out_dim = dims[l + 1]
                lin = nn.Linear(dims[l], out_dim)

                if weight_norm:
                    lin = nn.utils.weight_norm(lin)

                setattr(self, "lin" + str(l), lin)

            self.relu = nn.ReLU()

        if last_act == 'Sigmoid':
            if sigmoid_output_scale > 1:
                self.last_active_fun = lambda x: x.sigmoid() * sigmoid_output_scale
            else:
                self.last_active_fun = nn.Sigmoid()
        elif last_act == 'Exponential':
            self.last_active_fun = lambda x: (4**x)/2  # keep similar init value
        else:
            raise NotImplementedError

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.mode == 'refnerf' or self.force_reflection:
            view_dirs = 2 * (view_dirs*normals).sum(-1, keepdim=True) * normals - view_dirs

        view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        elif self.mode == 'refnerf':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input
        if self.fused:
            x = self.fused_net(x).float()

        else:
            for l in range(0, self.num_layers - 1):
                lin = getattr(self, "lin" + str(l))
                x = lin(x)
                if l < self.num_layers - 2:
                    x = self.relu(x)

        x = self.last_active_fun(x)

        return x

class Geometry(nn.Module):
    def __init__(self, config_model, phase, num_reg_samples=None, calc_hess=None):
        super().__init__()
        self.feature_vector_size = config_model['implicit_network']['dims'][-1]
        self.aux_dims = config_model['implicit_network']['aux_dims']
        self.scene_bounding_sphere = config_model.get('scene_bounding_sphere', 1.0)
        self.cutoff_bounding_sphere = config_model.get('cutoff_bounding_sphere', 1.0)
        self.object_bounding_sphere = config_model.get('object_bounding_sphere', 1.0)

        self.bkgd_type = config_model['background']['otype']
        assert self.bkgd_type in ['none', 'uniform', 'nerfpp']

        self.implicit_network = ImplicitNetwork(
            (self.scene_bounding_sphere if self.bkgd_type == 'none' else 0.0),
            self.cutoff_bounding_sphere,
            **config_model.get('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **config_model.get('rendering_network'))

        self.density = LaplaceDensity(**config_model.get('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, inverse_sphere_bg=(self.bkgd_type=='nerfpp'), **config_model.get('ray_sampler'))

        config_bkgd = config_model['background']
        if self.bkgd_type == 'uniform':
            self.bg_color = torch.tensor(config_bkgd.get("bg_color", [1.0, 1.0, 1.0])).float().cuda()
        elif self.bkgd_type == 'nerfpp':
            self.bg_feature_vector_size = config_bkgd['implicit_network']['dims'][-1]
            self.bg_implicit_network = ImplicitNetworkBG(**config_bkgd.get('implicit_network'))
            self.bg_rendering_network = RenderingNetwork(self.bg_feature_vector_size, **config_bkgd.get('rendering_network'))
            self.bg_density = AbsDensity(**config_bkgd.get('density', {}))

        self.phase = phase
        self.num_reg_samples = num_reg_samples
        self.calc_hess = calc_hess

    def forward(self, input):
        # Parse model input
        intrinsics = input["intrinsics"].reshape([-1, 4, 4])                            # [N, 4, 4]
        pose = input["pose"].reshape([-1, 4, 4])                                        # [N, 4, 4] NOTE: idr pose is inverse of mvsnet extrinsic
        uv = input["uv"].reshape([-1, 2])                                               # [N, 2]

        ray_dirs, cam_loc = geometry.get_camera_params(uv, pose, intrinsics)                           # [N, 3]

        num_pixels, _ = ray_dirs.shape

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, 
                                    self.density, self.implicit_network, take_sphere_intersection=True)
        z_max = None
        if self.bkgd_type == 'nerfpp':
            z_vals, z_vals_bg = z_vals
            z_max = z_vals[:,-1]
            z_vals = z_vals[:,:-1]
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = -ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        sdf, feature_vectors, imp_aux, gradients = self.implicit_network.get_outputs(points_flat)

        unit_pred_normals_flat = Func.normalize(imp_aux['pred_grad'], dim=-1)
        used_normals = unit_pred_normals_flat if self.rendering_network.mode == 'refnerf' else gradients
        rgb_flat = self.rendering_network(points_flat, used_normals, dirs_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights, bg_transmittance = self.volume_rendering(z_vals, sdf, z_max=z_max)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        # uniform background assumption
        if self.bkgd_type == 'uniform':
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)
        elif self.bkgd_type == 'nerfpp':
            # Background rendering
            N_bg_samples = z_vals_bg.shape[1]
            z_vals_bg = torch.flip(z_vals_bg, dims=[-1, ])  # 1--->0

            bg_dirs = ray_dirs.unsqueeze(1).repeat(1,N_bg_samples,1)
            bg_locs = cam_loc.unsqueeze(1).repeat(1,N_bg_samples,1)

            bg_points = self.depth2pts_outside(bg_locs, bg_dirs, z_vals_bg)  # [..., N_samples, 4]
            bg_points_flat = bg_points.reshape(-1, 4)
            bg_dirs_flat = bg_dirs.reshape(-1, 3)

            bg_sdf, bg_feature_vectors, bg_aux = self.bg_implicit_network.get_outputs(bg_points_flat)
            bg_rgb_flat = self.bg_rendering_network(None, None, -bg_dirs_flat, bg_feature_vectors)
            bg_rgb = bg_rgb_flat.reshape(-1, N_bg_samples, 3)

            bg_weights = self.bg_volume_rendering(z_vals_bg, bg_sdf)

            bg_rgb_values = torch.sum(bg_weights.unsqueeze(-1) * bg_rgb, 1)

            rgb_values = rgb_values + bg_transmittance.unsqueeze(-1) * bg_rgb_values

        normals = gradients #Func.normalize(gradients, dim=-1)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        normal_map = Func.normalize(normal_map, dim=-1)

        point_map = torch.sum(weights.unsqueeze(-1) * points, 1)

        if self.bkgd_type == 'none':
            render_masks = point_map.abs().max(dim=-1)[0] < self.object_bounding_sphere
        else:
            render_masks = bg_transmittance < 0.05  # NOTE hard code

        aux_output = {}
        if self.training and self.phase in ['geo', 'joint']:
            assert self.num_reg_samples is not None
            assert self.calc_hess is not None
            # Sample points for the eikonal loss
            object_rand_points = torch.empty(self.num_reg_samples, 3).uniform_(-self.object_bounding_sphere, self.object_bounding_sphere).cuda()
            scene_rand_points = torch.empty(num_pixels, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()
            rand_points = torch.cat([scene_rand_points, object_rand_points], dim=0)
            if self.calc_hess:
                rand_sdfs, rand_grad, rand_hess = self.implicit_network.get_sdf_vals_grad(rand_points, clamp=False, grad_order=2)
                aux_output['hess_theta'] = rand_hess
            else:
                rand_sdfs, rand_grad = self.implicit_network.get_sdf_vals_grad(rand_points, clamp=False, grad_order=1)

            # add some of the near surface points
            near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            near_sdfs, near_grad = self.implicit_network.get_sdf_vals_grad(near_points, clamp=False, grad_order=1)
            
            aux_output['grad_theta'] = torch.cat([rand_grad, near_grad], dim=0)
            aux_output['rand_sdfs'] = rand_sdfs

            if 'pcd_positions' in input:
                pcd_positions = input['pcd_positions'].reshape(-1, 3)                   # [P, 3]
                aux_output['pcd_sdfs'], aux_output['pcd_grad'] = \
                    self.implicit_network.get_sdf_vals_grad(pcd_positions, clamp=False, grad_order=1)

        return point_map, normal_map, -ray_dirs, rgb_values, render_masks, num_pixels, aux_output

    def radiance_variance(self, points, normals):
        sample_num = 256
        light_dir, _ = fibonacci_sphere_sampling(normals, sample_num, random_rotate=True)
        sdf, feature_vectors, imp_aux, gradients = self.implicit_network.get_outputs(points)
        unit_pred_normals_flat = Func.normalize(imp_aux['pred_grad'], dim=-1)
        used_normals = unit_pred_normals_flat if self.rendering_network.mode == 'refnerf' else gradients
        points_flat, used_normals, feature_vectors = \
            [arr.unsqueeze(1).repeat(1,sample_num,1).reshape(-1,arr.shape[-1])
            for arr in [points, used_normals, feature_vectors]]
        dirs_flat = light_dir.reshape(-1,3)
        rgb_flat = self.rendering_network(points_flat, used_normals, dirs_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, sample_num, rgb_flat.shape[-1])
        rgb_var = rgb.mean(2, keepdim=True).var(1).sqrt()
        return rgb_var
    
    def trace(self, points, normals, ray_dirs, sample=None, validate_normal=False):
        raise NotImplementedError # TODO
    
    def trace_and_render(self, points, normals, ray_dirs, sample=None, validate_normal=False):
        N, S, _ = ray_dirs.shape

        # sample
        trace_total_num = N * S
        trace_sample_num = points.shape[0] // sample if sample is not None else 1e9
        if trace_total_num > trace_sample_num:
            trace_sample = torch.multinomial(torch.full((trace_total_num,), 1/trace_total_num), trace_sample_num, replacement=False)
        else:
            trace_sample = slice(None, None)
        temp = torch.zeros((trace_total_num,), dtype=torch.bool, device=points.device)
        temp[trace_sample] = 1
        trace_sample = temp

        ray_dirs = ray_dirs.reshape(trace_total_num, 3)[trace_sample]
        points = points.unsqueeze(1).repeat(1,S,1).reshape(trace_total_num, 3)[trace_sample]
        normals = normals.unsqueeze(1).repeat(1,S,1).reshape(trace_total_num, 3)[trace_sample]

        # add small displacement from points along ray_dirs
        # ===== NOTE hard code params =====
        disp_max_density = torch.tensor(0.1, dtype=points.dtype, device=points.device)
        disp_max = 0.2
        min_ndv = np.cos(80/180*np.pi)
        actual_clearance_scale = 0.1
        weights_window = 0.05
        peak_weights_thresh = 0.8
        # =================================
        disp_min_sdf = self.density.abs_inv(disp_max_density).detach()
        n_d_v = (ray_dirs * normals).sum(-1,keepdim=True)
        displacement = (disp_min_sdf / n_d_v.clamp(1e-4)).clamp(max=disp_max)
        points = points + ray_dirs * displacement

        # start_sdf = self.implicit_network.get_sdf_vals(points, clamp=False)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, points, 
                                    self.density, self.implicit_network, take_sphere_intersection=True)
        z_max = None
        if self.bkgd_type == 'nerfpp':
            z_vals, z_vals_bg = z_vals
            z_max = z_vals[:,-1]
            z_vals = z_vals[:,:-1]
        N_samples = z_vals.shape[1]

        points_vol = points.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points_vol.reshape(-1, 3)

        dirs = -ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        sdf, feature_vectors, imp_aux, gradients = self.implicit_network.get_outputs(points_flat)

        n_d_r = None

        weights, bg_transmittance = self.volume_rendering(z_vals, sdf, n_d_r, z_max=z_max)

        trace_pos = torch.sum(weights.unsqueeze(-1) * points_vol, 1)
        trace_z = torch.sum(weights * z_vals, 1, keepdim=True)

        # filter valid rays
        ndv_mask = n_d_v[:,0] > min_ndv
        # start_mask = start_sdf[:,0] > disp_min_sdf * actual_clearance_scale
        valid_ray_mask = ndv_mask #& start_mask
        # get hits
        if self.bkgd_type == 'none':
            trace_mask = trace_pos.abs().max(dim=-1)[0] < self.object_bounding_sphere
        else:
            trace_mask = bg_transmittance < 0.05  # NOTE hard code
        # check weight sum around traced position
        peak_weights = (weights * ((trace_z - z_vals).abs() < weights_window)).sum(1)
        peak_weights_mask = peak_weights > peak_weights_thresh
        trace_mask = trace_mask & peak_weights_mask
        
        trace_pos = trace_pos[trace_mask & valid_ray_mask]

        # combine two mask
        trace_mask_combine = trace_sample.clone()
        trace_mask_combine[trace_sample] = trace_mask & valid_ray_mask
        trace_mask_combine = trace_mask_combine.reshape(N, S)
        # miss mask
        miss_mask_combine = trace_sample.clone()
        miss_mask_combine[trace_sample] = (~trace_mask) & valid_ray_mask
        miss_mask_combine = miss_mask_combine.reshape(N, S)

        used_normals = Func.normalize(imp_aux['pred_grad'], dim=-1) if self.rendering_network.mode == 'refnerf' else gradients

        # apply mask
        def apply_mask(arr):
            flat = False
            if arr.shape[0] == N * N_samples:
                flat = True
                arr = arr.reshape(N, N_samples, *arr.shape[1:])
            arr = arr[trace_mask & valid_ray_mask]
            if flat:
                arr = arr.reshape(-1, *arr.shape[2:])
            return arr
        ray_dirs, points_flat, used_normals, dirs_flat, feature_vectors, weights = \
            [apply_mask(arr) for arr in [ray_dirs, points_flat, used_normals, dirs_flat, feature_vectors, weights]]

        rgb_flat = self.rendering_network(points_flat, used_normals, dirs_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        trace_render_rgb = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        misc = {}
        # misc['trace_pos'] = trace_pos
        # misc['points_vol'] = points_vol[trace_mask & valid_ray_mask]

        return trace_mask_combine, miss_mask_combine, trace_render_rgb, misc
    
    def plot_point_apr(self, point, width):
        raise NotImplementedError  # TODO

    def volume_rendering(self, z_vals, sdf, n_d_r=None, z_max=None):
        density_flat = self.density(sdf, n_d_r=n_d_r)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        if z_max is None:
            dist_pad = torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)
        else:
            dist_pad = z_max.unsqueeze(-1) - z_vals[:, -1:]
        dists = torch.cat([dists, dist_pad], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        fg_transmittance = transmittance[:, :-1]
        weights = alpha * fg_transmittance  # probability of the ray hits something here
        bg_transmittance = transmittance[:, -1]  # factor to be multiplied with the bg volume rendering

        return weights, bg_transmittance

    def bg_volume_rendering(self, z_vals_bg, bg_sdf):
        bg_density_flat = self.bg_density(bg_sdf)
        bg_density = bg_density_flat.reshape(-1, z_vals_bg.shape[1]) # (batch_size * num_pixels) x N_samples

        bg_dists = z_vals_bg[:, :-1] - z_vals_bg[:, 1:]
        bg_dists = torch.cat([bg_dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(bg_dists.shape[0], 1)], -1)

        # LOG SPACE
        bg_free_energy = bg_dists * bg_density
        bg_shifted_free_energy = torch.cat([torch.zeros(bg_dists.shape[0], 1).cuda(), bg_free_energy[:, :-1]], dim=-1)  # shift one step
        bg_alpha = 1 - torch.exp(-bg_free_energy)  # probability of it is not empty here
        bg_transmittance = torch.exp(-torch.cumsum(bg_shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        bg_weights = bg_alpha * bg_transmittance # probability of the ray hits something here

        return bg_weights

    def depth2pts_outside(self, ray_o, ray_d, depth):
        '''
        ray_o, ray_d: [..., 3]
        depth: [...]; inverse of distance to sphere origin
        '''

        o_dot_d = torch.sum(ray_d * ray_o, dim=-1)
        under_sqrt = o_dot_d ** 2 - ((ray_o ** 2).sum(-1) - self.scene_bounding_sphere ** 2)
        d_sphere = torch.sqrt(under_sqrt) - o_dot_d
        p_sphere = ray_o + d_sphere.unsqueeze(-1) * ray_d
        p_mid = ray_o - o_dot_d.unsqueeze(-1) * ray_d
        p_mid_norm = torch.norm(p_mid, dim=-1)

        rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
        rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
        phi = torch.asin(p_mid_norm / self.scene_bounding_sphere)
        theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
        rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

        # now rotate p_sphere
        # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                       torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                       rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
        p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

        return pts

    def get_surface_trace(self, path, epoch, scale_mat=None, resolution=100, grid_boundary=[-1.0, 1.0], return_mesh=False, level=0):

        def get_grid_uniform(resolution, grid_boundary=[-2.0, 2.0]):
            x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
            y = x
            z = x

            xx, yy, zz = np.meshgrid(x, y, z)
            grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

            return {"grid_points": grid_points.cuda(),
                    "shortest_axis_length": 2.0,
                    "xyz": [x, y, z],
                    "shortest_axis_index": 0}

        grid = get_grid_uniform(resolution, grid_boundary)
        points = grid['grid_points']

        z = []
        for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
            z.append(self.implicit_network.get_sdf_vals(pnts, clamp=False)[:, 0].detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        if (not (np.min(z) > level or np.max(z) < level)):

            z = z.astype(np.float32)

            verts, faces, normals, values = measure.marching_cubes(
                volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=level,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                        grid['xyz'][0][2] - grid['xyz'][0][1],
                        grid['xyz'][0][2] - grid['xyz'][0][1]))

            verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            I, J, K = faces.transpose()

            # traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            #                     i=I, j=J, k=K, name='implicit_surface',
            #                     color='#ffffff', opacity=1.0, flatshading=False,
            #                     lighting=dict(diffuse=1, ambient=0, specular=0),
            #                     lightposition=dict(x=0, y=0, z=-1), showlegend=True)]
            traces = None

            meshexport = trimesh.Trimesh(verts, faces, normals)

            if scale_mat is not None:
                meshexport.apply_transform(scale_mat)

            meshexport.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')

            if return_mesh:
                return meshexport
            return traces
        return None

class GeoLoss(nn.Module):
    def __init__(self, rgb_loss, 
                 eikonal_weight, hessian_weight, minsurf_weight, 
                 orientation_weight, pred_normal_weight, feature_weight,
                 pcd_weight, remove_black=False, rf_loss_scale_mod=1.0):
        super().__init__()
        self.rgb_loss = general.get_class(rgb_loss)(reduction='mean')
        self.eikonal_weight = eikonal_weight
        self.hessian_weight = hessian_weight
        self.minsurf_weight = minsurf_weight
        self.orientation_weight = orientation_weight
        self.pred_normal_weight = pred_normal_weight
        self.feature_weight = feature_weight
        self.pcd_weight = pcd_weight
        print ("eikonal_weight", eikonal_weight)
        print ("hessian_weight", hessian_weight)
        print ("minsurf_weight", minsurf_weight)
        print ("orientation_weight", orientation_weight)
        print ("pred_normal_weight", pred_normal_weight)
        print ("feature_weight", feature_weight)
        print ("pcd_weight", pcd_weight)
        self.remove_black = remove_black
        self.rf_loss_scale_mod = rf_loss_scale_mod

    def get_rgb_loss(self,rgb_values, rgb_gt):
        if self.remove_black:
            valid = (rgb_gt > 0).long().sum(-1) > 0
            rgb_values = rgb_values[valid]
            rgb_gt = rgb_gt[valid]
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_hessian_loss(self, hessians):
        return hessians.abs().mean()

    def get_minsurf_loss(self, sdfs, sigma=10):
        return (sigma / (sigma * sigma + sdfs**2) / np.pi).mean()

    def get_orientation_loss(self, weights, ndv, mask=slice(None,None)):
        return (weights * ndv.clamp(min=0)**2).sum(1)[mask].mean()

    def get_pred_normal_loss(self, weights, normals_dot, mask=slice(None,None)):
        return (weights * (1 - normals_dot)).sum(1)[mask].mean()

    def get_feature_loss(self, feat, feat_gt):
        feat = feat.reshape(-1, 8, 4)
        feat_gt = feat_gt.reshape(-1, 8, 4)
        cos_sim = Func.cosine_similarity(feat, feat_gt, dim=-1)
        return (1 - cos_sim).abs().mean()

    def get_pcd_loss(self, pcd_sdfs, pcd_grad, pcd_normals):
        pcd_loss = pcd_sdfs.abs().mean()
        if pcd_normals is not None:
            normal_loss = (1 - Func.cosine_similarity(pcd_grad, pcd_normals, dim=-1)).abs().mean()
            pcd_loss = pcd_loss + normal_loss
        return pcd_loss

    def forward(self, model_outputs, ground_truth, progress_iter):
        ZERO = torch.tensor(0.0).cuda().float()

        rgb_gt = ground_truth['rgb'].cuda().reshape(-1, 3)

        # RGB loss
        rgb_loss = self.get_rgb_loss(model_outputs['geo_rgb'], rgb_gt)
        rgb_loss = rgb_loss * self.rf_loss_scale_mod

        geo_outputs = model_outputs['geo_aux_output']
        # Regularization 1: Eikonal loss
        if 'grad_theta' in geo_outputs:
            eikonal_loss = self.get_eikonal_loss(geo_outputs['grad_theta'])
        else:
            eikonal_loss = ZERO
        
        # Regularization 2: Hessian loss
        if 'hess_theta' in geo_outputs:
            hessian_loss = self.get_hessian_loss(geo_outputs['hess_theta'])
        else:
            hessian_loss = ZERO

        # Regularization 3: Minimum surface loss
        if 'rand_sdfs' in geo_outputs:
            minsurf_loss = self.get_minsurf_loss(geo_outputs['rand_sdfs'])
        else:
            minsurf_loss = ZERO

        # Regularization 4: back orientation penalty
        if 'weights' in geo_outputs and 'ndv' in geo_outputs and model_outputs['render_masks'].sum().item() > 0:
            orientation_loss = self.get_orientation_loss(geo_outputs['weights'],
                                                         geo_outputs['ndv'],
                                                         model_outputs['render_masks'])
        else:
            orientation_loss = ZERO

        # Regularization 5: predicted normal
        if 'weights' in geo_outputs and 'normals_dot' in geo_outputs and model_outputs['render_masks'].sum().item() > 0:
            pred_normal_loss = self.get_pred_normal_loss(geo_outputs['weights'],
                                                         geo_outputs['normals_dot'],
                                                         model_outputs['render_masks'])
        else:
            pred_normal_loss = ZERO

        # Feature loss
        if 'vis_feature' in geo_outputs:
            feature_gt = ground_truth['feat'].cuda().reshape(-1, 32)
            feature_loss = self.get_feature_loss(geo_outputs['vis_feature'], feature_gt)
        else:
            feature_loss = ZERO

        # sdf loss
        if 'pcd_sdfs' in geo_outputs and 'pcd_grad' in geo_outputs:
            pcd_normals = ground_truth['pcd_normals'].cuda().reshape(-1,3) if 'pcd_normals' in ground_truth else None
            pcd_loss = self.get_pcd_loss(geo_outputs['pcd_sdfs'], geo_outputs['pcd_grad'], pcd_normals)
        else:
            pcd_loss = ZERO

        loss = rgb_loss + \
               self.orientation_weight * orientation_loss + \
               self.pred_normal_weight * pred_normal_loss + \
               self.feature_weight * feature_loss + \
               self.pcd_weight * pcd_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.hessian_weight * hessian_loss + \
               self.minsurf_weight * minsurf_loss

        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'hessian_loss': hessian_loss,
            'minsurf_loss': minsurf_loss,
        }

        return loss
