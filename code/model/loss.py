#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch
from torch import nn
from utils import general

class NeILFLoss(nn.Module):
    def __init__(self, rgb_loss_type, lambertian_weighting, smoothness_weighting, trace_weighting, var_weighting, geo_loss, phase, **kwargs):
        super().__init__()
        self.rgb_loss_func = general.get_class(rgb_loss_type)(reduction='none')
        self.reg_weight = lambertian_weighting
        self.smooth_weight = smoothness_weighting
        self.trace_weight = trace_weighting
        self.var_weighting = var_weighting
        self.mono_neilf = kwargs.get('mono_neilf', False)
        self.remove_black = kwargs.get('remove_black', False)
        self.trace_grad_scale = kwargs.get('trace_grad_scale', 0)

        self.geo_loss = geo_loss
        self.phase = phase

    def forward(self, model_outputs, ground_truth, progress_iter):
        ZERO = torch.tensor(0.0).cuda().float()
        mat_loss = geo_loss = ZERO
        
        if self.phase in ['mat', 'joint']:
            rgb_gt = ground_truth['rgb'].cuda().reshape(-1, 3)
            masks = model_outputs['render_masks']

            # get rid of invalid pixels (because of undistortion)
            if self.remove_black:
                valid = (rgb_gt > 0).long().sum(-1) > 0
                masks = masks & valid

            masks = masks.float()
            mask_sum = masks.sum().clamp(min=1e-7)

            # rendered rgb 
            rgb_values = model_outputs['rgb_values']
            rgb_loss = (self.rgb_loss_func(rgb_values, rgb_gt) * masks.unsqueeze(1)).sum() / mask_sum / 3

            # smoothness smoothness
            rgb_grad = ground_truth['rgb_grad'].cuda().reshape(-1)
            brdf_grads = model_outputs['brdf_grads']                # [N, 2, 3]
            smooth_loss = (brdf_grads.norm(dim=-1).sum(dim=-1) * (-rgb_grad).exp() * masks).mean()

            # lambertian assumption
            roughness = model_outputs['roughness']
            metallic = model_outputs['metallic']
            # reg_loss = ((roughness - 1).abs() * masks.unsqueeze(1)).sum() / mask_sum + \
            #     ((metallic - 0).abs() * masks.unsqueeze(1)).sum() / mask_sum
            reg_loss = ((metallic - 0).abs() * masks.unsqueeze(1)).sum() / mask_sum

            # trace
            trace_nn_rgb = model_outputs['trace_nn_rgb']
            trace_render_rgb = model_outputs['trace_render_rgb']
            trace_render_rgb = general.scale_grad(trace_render_rgb, self.trace_grad_scale)
            if self.mono_neilf:
                gray_weight = torch.tensor([[0.2989, 0.5870, 0.1140]], dtype=trace_render_rgb.dtype, device=trace_render_rgb.device)
                trace_render_rgb = (trace_render_rgb * gray_weight).sum(dim=-1, keepdim=True)
                trace_nn_rgb = trace_nn_rgb[..., :1]
            trace_loss = (trace_nn_rgb - trace_render_rgb).abs().sum() / (trace_nn_rgb.numel() + 1e-9)

            # variance guidance
            var_loss = ZERO
            if 'rgb_var' in model_outputs and progress_iter <= 1000:
                rgb_variance = model_outputs['rgb_var']
                var_min, var_max = 0.01, 0.15
                tgt_min, tgt_max = 0.25, 0.75
                roughness_target = (rgb_variance - var_min) / (var_max - var_min)
                roughness_target = (1-roughness_target.clamp(0,1)) * (tgt_max - tgt_min) + tgt_min
                var_loss = ((roughness - roughness_target).abs() * masks.unsqueeze(1)).sum() / mask_sum

            mat_loss =  rgb_loss + \
                        self.smooth_weight * smooth_loss + \
                        self.reg_weight * reg_loss + \
                        self.trace_weight * trace_loss + \
                        self.var_weighting * var_loss

        # slf loss
        if self.phase in ['geo', 'joint']:
            geo_loss = self.geo_loss(model_outputs, ground_truth, progress_iter)

        return mat_loss + geo_loss
