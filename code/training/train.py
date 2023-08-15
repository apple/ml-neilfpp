#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import gc
import argparse
import sys
sys.path.append('../code')
import numpy as np
import random
import tqdm
from importlib import import_module
import json

# time
from datetime import datetime

# torch
import torch
import tinycudann as tcnn

# neilf
from dataset.dataset import NeILFDataset
from model.neilf_brdf import NeILFModel
from model.loss import NeILFLoss
import utils.io as io
import utils.general as general

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()

class NeILFTrainer():

    def _create_output_folders(self):

        # create brdf/lighting output folders
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        self.timestamp_folder = os.path.join(self.output_folder, self.phase, self.timestamp)
        self.plots_folder = os.path.join(self.timestamp_folder, 'plots')
        os.makedirs(self.plots_folder, exist_ok=True)

        # create model checkpoint folders
        self.checkpoint_folder = os.path.join(self.timestamp_folder, 'checkpoints')
        os.makedirs(os.path.join(self.checkpoint_folder, 'ModelParameters'), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_folder, 'OptimizerParameters'), exist_ok=True)
        os.makedirs(os.path.join(self.checkpoint_folder, 'SchedulerParameters'), exist_ok=True)

    def _create_optimizer(self):

        self.lr = self.config['train']['lr']
        self.lr_decay = self.config['train']['lr_decay']
        self.lr_decay_iters = self.config['train']['lr_decay_iters']
        self.lr_warmup_start_factor = self.config['train']['lr_warmup_start_factor']
        self.lr_warmup_iters = self.config['train']['lr_warmup_iters']
        self.use_ldr_image = self.config['model']['use_ldr_image']
        # NeILF and BRDF MLPs
        param_groups = []
        if self.phase in ['geo', 'joint']:
            param_groups.append({'name': 'geometry', 'params': self.model.geometry.parameters()})
        if self.phase in ['mat', 'joint']:
            param_groups.append({'name': 'neilf_pbr', 'params': self.model.neilf_pbr.parameters()})
        # # learnable HDR-LDR gamma correction
        # if self.use_ldr_image:
        #     self.lr_scaler = self.config['train']['lr_scaler']
        #     param_groups.append(
        #         {'name': 'gamma', 'params': self.model.gamma, 'lr': self.lr_scaler})
        # Find special params
        for pg in param_groups:
            pg['params'] = list(pg['params'])
        special_groups = [
            {'params': [p], **p.custom_meta['optim_args']}
            for p in sum([g['params'] for g in param_groups], start=[])
            if hasattr(p, 'custom_meta') and 'optim_args' in p.custom_meta
        ]
        special_params_id = [id(sg['params'][0]) for sg in special_groups]
        params_id2name = {id(p): n for n, p in self.model.named_parameters()}
        if len(special_groups) > 0:
            print('Special optim args:')
            for sg in special_groups:
                sp_args = sg['params'][0].custom_meta['optim_args']
                sp_name = params_id2name[id(sg['params'][0])]
                print(sp_name, sp_args)
            for pg in param_groups:
                pg['params'] = [p for p in pg['params'] if id(p) not in special_params_id]
            param_groups += special_groups
        # Edit param groups
        for g in param_groups:
            if 'lr_scale' in g:
                g['lr'] = self.lr * g['lr_scale']
                del g['lr_scale']
            if 'name' in g:
                del g['name']
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=0, eps=1e-12)
        def scheduler_func(epoch):
            epoch = epoch + 1
            if epoch > self.lr_warmup_iters: 
                mult = 1
                for milestone in self.lr_decay_iters:
                    if epoch > milestone:
                        mult *= self.lr_decay
                return mult
            else:
                gamma = (1/self.lr_warmup_start_factor) ** (1/self.lr_warmup_iters)
                return self.lr_warmup_start_factor * gamma ** (epoch-1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, scheduler_func)

    def _load_checkpoint(self, is_continue):
        if start_iteration := io.load_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.output_folder, self.timestamp, self.phase,
            self.last_timestamp, self.last_checkpoint, is_continue
        ).get('start_iteration'):
            self.start_iteration = start_iteration
    
    def _save_checkpoint(self, iteration, is_final=False):
        io.save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.checkpoint_folder, iteration, is_final
        )

    def __init__(self, 
                 input_folder, 
                 output_folder, 
                 config_path, 
                 phase,
                 no_load,
                 is_continue, 
                 timestamp, 
                 checkpoint,
                 config_override,
                 seed):

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.phase = phase
        self.is_continue = is_continue
        self.last_timestamp = timestamp
        self.last_checkpoint = checkpoint
        
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        print(f'current phase: {self.phase}')

        # create output folders
        self._create_output_folders()

        # load config
        config = io.load_config(config_path, self.phase)
        for k, v in config_override.items():
            k = k[2:]
            print(f'{k} -> {v}')
            path = k.split('.')
            io.modify_dict(config, path, v, convert_type=True)
        self.config = config

        # load input data and create dataset
        num_pixel_samples = config['train']['num_pixel_samples']
        self.dataset = NeILFDataset(
            input_folder, num_pixel_samples=num_pixel_samples, mode='train', **config['dataset'])

        # create training data loader
        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=1, shuffle=False, drop_last=False, 
            collate_fn=self.dataset.collate_fn)
        
        # geometry model
        geometry_module = import_module(config['model']['geometry_module'])
        kwargs = {}
        if config['model']['geometry_module'] == 'model.geo_fixmesh':
            from pyrt import PyRT
            ray_tracer = PyRT(*self.dataset.tracer_mesh, 0)
            kwargs['ray_tracer'] = ray_tracer
        if config['model']['geometry_module'] == 'model.geo_volsdf':
            kwargs['num_reg_samples'] = config['train']['num_reg_samples']
            kwargs['calc_hess'] = (config['train']['geo_loss']['hessian_weight'] > 0)
        geometry = geometry_module.Geometry(config['model']['geometry'], self.phase, **kwargs)

        # adjust neilf init output
        if self.phase == 'mat' and config['model']['neilf_network']['init_output'] < 0:
            # init 1 ~ render 0.5 ~ ldr 0.73
            init_value = self.dataset.train_rgb_median ** (2.2 if config['model']['use_ldr_image'] else 1.0) / 0.5
            config['model']['neilf_network']['init_output'] = init_value
            print(f'init neilf to {init_value} for data median {self.dataset.train_rgb_median}')

        # create model
        self.model = NeILFModel(config['model'], geometry, self.phase, rgb_var=config['train']['var_weighting']>0)
        self.model.set_mat2geo_grad_scale(config['train']['mat2geo_grad_scale'])
        if torch.cuda.is_available():
            self.model.cuda()

        # create loss functions
        rgb_loss_type = config['train']['rgb_loss']
        lambertian_weighting = config['train']['lambertian_weighting']
        smoothness_weighting = config['train']['smoothness_weighting']
        trace_weighting = config['train']['trace_weighting']
        var_weighting = config['train']['var_weighting']
        remove_black = config['train'].get('remove_black', False)
        if config['model']['geometry_module'] == 'model.geo_fixmesh':
            rf_output_scale = config['model']['geometry']['slf_network'].get('sigmoid_output_scale',1.0)
        elif config['model']['geometry_module'] == 'model.geo_volsdf':
            rf_output_scale = config['model']['geometry']['rendering_network'].get('sigmoid_output_scale',1.0)
        geo_loss = geometry_module.GeoLoss(rf_loss_scale_mod=1/rf_output_scale, **config['train']['geo_loss'])
        self.loss = NeILFLoss(rgb_loss_type, 
                              lambertian_weighting, smoothness_weighting, trace_weighting, var_weighting, 
                              geo_loss, self.phase, 
                              mono_neilf=config['model']['use_ldr_image'],
                              remove_black=remove_black,
                              trace_grad_scale=config['train'].get('trace_grad_scale', 0))
        
        # create optimizer and scheduler
        self._create_optimizer()

        # load pre-trained model
        self.start_iteration = 0
        if (not no_load) and (is_continue or self.phase in ['mat', 'joint']):
            self._load_checkpoint(is_continue)

        self.num_pixel_samples = config['train']['num_pixel_samples']
        self.total_pixels = self.dataset.total_pixels
        self.image_resolution = self.dataset.image_resolution
        self.n_batches = len(self.train_dataloader)
        self.training_iterations = config['train']['training_iterations']
        self.plot_frequency = config['eval']['plot_frequency']
        self.save_frequency = config['eval']['save_frequency']
        self.eval_downsample = 4

        # backup config
        with open(os.path.join(self.timestamp_folder, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    def train(self):

        # progress bar
        progress_bar = tqdm.trange(self.start_iteration, self.training_iterations, dynamic_ncols=True)

        # self.model.eval()
        # self.validate(0)

        # training
        self.model.train()
        for iteration, (model_input, ground_truth) in enumerate(self.train_dataloader):
            
            # progress iteration, start from 1
            progress_iter = self.start_iteration + iteration + 1

            # transfer input to gpu
            model_input = {k: v.cuda() for k, v in model_input.items()}

            # run network
            model_outputs = self.model(model_input)
            
            # compute loss
            loss = self.loss(model_outputs, ground_truth, progress_iter)

            # optimize 
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # log message
            current_lr = self.scheduler.get_last_lr()[0]
            num_traced_pixels = model_outputs['render_masks'].sum().item()
            log_msg = f'{loss.item():.4f}, {current_lr:.1e}, {self.num_pixel_samples}/{num_traced_pixels}'
            # if self.use_ldr_image:
            #     log_msg = log_msg + f', gamma = {self.model.gamma.item():.4f}'
            if model_outputs.get('trace_render_rgb') is not None:
                log_msg = log_msg + f'/{model_outputs["trace_render_rgb"].shape[0]}'
                log_msg = log_msg + f', # trace {model_outputs["trace_render_rgb"].shape[0]}'
            progress_bar.update(1)
            progress_bar.set_description(log_msg)
            is_final = progress_iter >= self.training_iterations
            if is_final:
                progress_bar.close()

            # save model
            if progress_iter % self.save_frequency == 0 or is_final:
                self._save_checkpoint(progress_iter, is_final=is_final)

            # validate and save plot
            if progress_iter % self.plot_frequency == 0 or is_final:
                del model_input, ground_truth, model_outputs, loss
                cleanup()
                self.model.eval()
                self.validate(progress_iter)
                cleanup()
                self.model.train()

            # finish training
            self.scheduler.step()
            if is_final:
                break

    def validate(self, iteration):

        if iteration >= self.training_iterations:
            downsample = None
            total_pixels = self.total_pixels
        else:
            downsample = self.eval_downsample
            total_pixels = self.total_pixels // (downsample**2)
        model_input, ground_truth = self.dataset.get_validation_data(downsample)

        model_input = {k: v.cuda() for k, v in model_input.items()}

        # split inputs
        split_inputs = general.split_neilf_input(model_input, total_pixels)

        required_outputs = {
            'geo': ['geo_rgb', 'points', 'normals', 'render_masks'],
            'mat': ['rgb_values', 'base_color', 'roughness', 'metallic', 'render_masks'],
        }
        required_outputs['joint'] = required_outputs['geo'] + required_outputs['mat']

        # generate outputs
        split_outputs = []
        for split_input in split_inputs:
            with torch.no_grad():
                split_output = self.model(split_input)
            required_output = required_outputs[self.phase]
            if self.phase == 'mat' and 'rgb_var' in split_output:
                required_output.append('rgb_var')
            split_outputs.append(
                {k:split_output[k].detach().cpu() for k in required_output})

        # merge output
        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = general.merge_neilf_output(
            split_outputs, total_pixels, batch_size)

        if self.phase in ['geo', 'joint']:
            self.plot_geo(iteration, model_outputs, ground_truth)
        if self.phase in ['mat', 'joint']:
            self.plot_mat(iteration, model_outputs, ground_truth)
    
    def plot_geo(self, iteration, model_outputs, ground_truth):
        is_last = iteration >= self.training_iterations

        # mesh
        mesh_resolution = self.config['eval'].get('full_resolution', 512) if is_last else self.config['eval'].get('resolution', 100)
        self.model.geometry.get_surface_trace(self.plots_folder, iteration, self.dataset.scale_mat, resolution=mesh_resolution)

        # image size
        H = self.image_resolution[0] // (1 if is_last else self.eval_downsample)
        W = self.image_resolution[1] // (1 if is_last else self.eval_downsample)
        batch_size = ground_truth['rgb'].shape[0]

        # rendered mask
        mask = model_outputs['render_masks'].reshape([batch_size, H, W, 1]).float()

        # geo render image
        rgb_eval = model_outputs['geo_rgb']
        rgb_eval = rgb_eval.reshape([batch_size, H, W, 3])
        # if not self.use_ldr_image: 
        #     rgb_eval = general.hdr2ldr(rgb_eval)
        # rgb_eval = rgb_eval * mask + (1 - mask)

        # geo properties
        points = model_outputs['points'].reshape([batch_size, H, W, 3]) * mask
        normals = model_outputs['normals'].reshape([batch_size, H, W, 3]) * mask

        # create figure to plot
        rgb_plot = torch.cat([rgb_eval*2-1, points, normals, mask.repeat(1,1,1,3)*2-1], dim=1)  # [V, 4H, W, 3]
        rgb_plot = rgb_plot.permute(1,0,2,3).reshape(-1, batch_size*W, 3)  # [4H, VW, 3]

        # save figure to file
        io.save_image(f'{self.plots_folder}/geo_{iteration}.exr', rgb_plot)

    def plot_mat(self, iteration, model_outputs, ground_truth):
        is_last = iteration >= self.training_iterations

        # image size
        H = self.image_resolution[0] // (1 if is_last else self.eval_downsample)
        W = self.image_resolution[1] // (1 if is_last else self.eval_downsample)
        batch_size = ground_truth['rgb'].shape[0]

        # rendered mask
        mask = model_outputs['render_masks'].reshape([batch_size, H, W, 1]).float()

        # estimated image
        rgb_eval = model_outputs['rgb_values']
        rgb_eval = rgb_eval.reshape([batch_size, H, W, 3])
        if not self.use_ldr_image: 
            rgb_eval = general.hdr2ldr(rgb_eval)
        rgb_eval = rgb_eval * mask + (1 - mask)

        # gt image
        rgb_gt = ground_truth['rgb']
        rgb_gt = rgb_gt.reshape([batch_size, H, W, 3])
        if not self.use_ldr_image: 
            rgb_gt = general.hdr2ldr(rgb_gt)
        rgb_gt = rgb_gt * mask + (1 - mask)

        # estimated BRDF
        base_eval = model_outputs['base_color'].reshape([batch_size, H, W, 3])
        base_eval = base_eval * mask + (1 - mask)
        roug_eval = model_outputs['roughness'].repeat([1, 3])
        roug_eval = roug_eval.reshape([batch_size, H, W, 3])
        meta_eval = model_outputs['metallic'].repeat([1, 3])
        meta_eval = meta_eval.reshape([batch_size, H, W, 3])

        # create figure to plot
        rgb_plot = torch.cat([rgb_eval, rgb_gt, base_eval, roug_eval, meta_eval], dim=1)    # [V, H * 5, W, 3]
        rgb_plot = rgb_plot.permute([0, 2, 1, 3])                                           # [V, W, H * 5, 3]
        rgb_plot = rgb_plot.reshape([-1, rgb_plot.shape[2], rgb_plot.shape[3]])             # [V * W, 5 * H, 3]
        rgb_plot = rgb_plot.permute([1, 0, 2])                                              # [5 * H, V * W, 3]
        rgb_plot = (rgb_plot.clamp(0, 1).detach().numpy() * 255).astype(np.uint8)           # [5 * H, V * W, 3]

        # save figure to file
        io.save_image(f'{self.plots_folder}/render_{iteration}.jpg', rgb_plot)

        if iteration == 0 and self.phase == 'mat' and 'rgb_var' in model_outputs:
            rgb_plot = model_outputs['rgb_var'].reshape(batch_size, H, W, 1)
            rgb_plot = rgb_plot.permute([0, 2, 1, 3])
            rgb_plot = rgb_plot.reshape([-1, rgb_plot.shape[2], rgb_plot.shape[3]])
            rgb_plot = rgb_plot.permute([1, 0, 2])
            rgb_plot = rgb_plot.detach().numpy()
            io.save_image(f'{self.plots_folder}/var_{iteration}.exr', rgb_plot)

        # calculate PSNR
        psnr = 0
        for i in range(batch_size):
            psnr += general.calculate_psnr(rgb_gt[i].numpy(), rgb_eval[i].numpy())
        psnr /= batch_size
        print (f'Validation at iteration: {iteration}, PSNR: {psnr.item():.4f}')
        with open(os.path.join(self.timestamp_folder, 'eval.csv'), 'a') as f:
            f.write(f'{iteration},{psnr.item()}\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_folder', type=str, 
                        help='Input folder of images, cameras and geometry files.')
    parser.add_argument('output_folder', type=str, 
                        help='Output folder for saving trained models and results')
    parser.add_argument('--config_path', type=str, 
                        default='./configs/config_dtu_volsdf_ngp.json')    
    parser.add_argument('--phase', type=str, default='geo,mat,joint',
                        help='The stages to be run')
    parser.add_argument('--no_load', default=False, action="store_true", 
                        help='If set, do not load weight from previous step.')
    # finetuning options
    parser.add_argument('--is_continue', default=False, action="store_true", 
                        help='If set, indicates fine-tuning from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str, 
                        help='The timestamp to be used if from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint to be used if from a previous run.')
    parser.add_argument('--seed', default=None, type=int,
                        help='Global seed.')
    args, unknown= parser.parse_known_args()

    assert len(unknown) % 2 == 0
    config_override = dict(zip(unknown[:-1:2],unknown[1::2]))
    assert all([k.startswith('--') for k in config_override])

    phases = args.phase.split(',')

    # run training
    for phase in phases:
        trainer = NeILFTrainer(input_folder=args.input_folder,
                            output_folder=args.output_folder,
                            config_path=args.config_path,
                            phase=phase,
                            no_load=args.no_load,
                            is_continue=args.is_continue,
                            timestamp=args.timestamp,
                            checkpoint=args.checkpoint,
                            config_override=config_override,
                            seed=args.seed)
        trainer.train()
        del trainer
        cleanup()
