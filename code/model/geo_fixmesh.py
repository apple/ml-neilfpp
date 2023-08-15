#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn
import torch.nn.functional as Func

from utils.geometry import equirectangular_proj, sph2cart, get_camera_params
from .nn_arch import NeILFMLP

class Geometry(nn.Module):
    def __init__(self, config_model, phase, ray_tracer):
        super().__init__()

        # surface light field
        slf_config = config_model['slf_network']
        self.slf_nn = NeILFMLP(**slf_config)

        self.ray_tracer = ray_tracer
        self.phase = phase
    
    def input_preprocessing(self, input):
        # parse model input
        intrinsics = input["intrinsics"].reshape([-1, 4, 4])                            # [N, 4, 4]
        pose = input["pose"].reshape([-1, 4, 4])                                        # [N, 4, 4] NOTE: idr pose is inverse of mvsnet extrinsic
        uv = input["uv"].reshape([-1, 2])                                               # [N, 2]
        points = input["positions"].reshape([-1, 3])                                    # [N]
        normals = Func.normalize(input["normals"].reshape([-1, 3]), dim=1)              # [N, 3]
        total_samples = uv.shape[0]

        # pixel index to image coord
        uv = uv + 0.5                                                                   # [N, 2]

        # get viewing directions
        ray_dirs, _ = get_camera_params(uv, pose, intrinsics)                  # [N, 3]
        view_dirs = -ray_dirs                                                           # [N, 3]

        # get mask
        render_masks = (points != 0).sum(-1) > 0                                        # [N]

        return points, normals, view_dirs, render_masks, total_samples
    
    def forward(self, input):
        # parse input
        points, normals, view_dirs, render_masks, total_samples \
            = self.input_preprocessing(input)

        # slf rendering TODO whether sample more for slf?
        slf_input = torch.cat([points, view_dirs], dim=-1)  # view_dirs is out-going from surface
        slf_rgb = self.slf_nn(slf_input)

        return points, normals, view_dirs, slf_rgb, render_masks, total_samples, {}
    
    def trace(self, points, normals, ray_dirs, sample=None, validate_normal=True):
        # ray trace
        trace_origin = points.unsqueeze(1).repeat(1,ray_dirs.shape[1],1).reshape(-1,3)
        trace_dir = ray_dirs.reshape(-1,3)
        trace_result_np = self.ray_tracer.trace_torch(trace_origin, trace_dir)
        trace_pos, trace_normal = trace_result_np.reshape(ray_dirs.shape[0], ray_dirs.shape[1], 6).split((3,3),-1)
        # get hits and sample
        trace_mask = trace_pos[...,0] < 9000
        trace_hit_num = trace_mask.nonzero(as_tuple=False).shape[0]
        trace_sample_num = points.shape[0] // sample if sample is not None else 1e9
        if trace_hit_num > trace_sample_num:
            trace_sample = torch.multinomial(torch.full((trace_hit_num,), 1/trace_hit_num), trace_sample_num, replacement=False)
        else:
            trace_sample = slice(None, None)
        temp = torch.zeros_like(trace_mask[trace_mask])
        temp[trace_sample] = 1
        trace_sample = temp
        # correct normal dir
        if validate_normal:
            trace_normal_neg = ((trace_normal[trace_mask][trace_sample] * ray_dirs[trace_mask][trace_sample]).sum(-1,True) < 0) *2-1
            trace_normal = trace_normal[trace_mask][trace_sample] * trace_normal_neg
        # combine mask
        miss_mask = ~trace_mask
        trace_mask_ = trace_mask.clone()
        trace_mask_[trace_mask] = trace_sample
        
        return trace_pos, trace_normal, trace_mask, miss_mask
    
    def trace_and_render(self, points, normals, ray_dirs, sample=None, validate_normal=True):
        trace_pos, trace_normal, trace_mask, miss_mask = self.trace(points, normals, ray_dirs, sample, validate_normal)
        # alt second pass: use slf
        trace_slf_input = torch.cat([trace_pos[trace_mask], -ray_dirs[trace_mask]], dim=-1)
        trace_render_rgb = self.slf_nn(trace_slf_input)
        return trace_mask, miss_mask, trace_render_rgb, {}
    
    def plot_point_apr(self, point, width):

        # incident direction of all pixels in the env map
        eval_sph, valid_mask = equirectangular_proj(width, meridian=0)                      # [H, W, 2]
        eval_sph = eval_sph.to(point.device)
        valid_mask = valid_mask.to(point.device)
        eval_cart = sph2cart(eval_sph, dim=-1)                                              # [H, W, 3]
        eval_cart_flat = -1 * eval_cart.view([-1, 3])                                       # [N, 3]

        point = point.unsqueeze(0).repeat([eval_cart_flat.shape[0], 1])                     # [N, 3]
        nn_inputs = torch.cat([point, eval_cart_flat], axis=1)                              # [N, 6]
        slf_map = self.slf_nn(nn_inputs).view(-1, width, 3)                        # [N, 3]

        slf_map *= valid_mask
        return slf_map

    def get_surface_trace(self, *args, **kwargs):
        pass

class GeoLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.rf_loss_scale_mod = kwargs.get('rf_loss_scale_mod', 1.0)

    def forward(self, model_outputs, ground_truth, progress_iter):
        
        masks = model_outputs['render_masks'].float()
        mask_sum = masks.sum().clamp(min=1e-7)

        # slf loss
        rgb_gt = ground_truth['rgb'].cuda().reshape(-1, 3)
        slf_rgb = model_outputs['geo_rgb']
        slf_loss = ((slf_rgb - rgb_gt).abs() * masks.unsqueeze(1)).sum() / mask_sum / 3
        slf_loss = slf_loss * self.rf_loss_scale_mod

        return slf_loss