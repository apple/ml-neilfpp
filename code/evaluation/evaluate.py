#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import argparse
import os
import torch
import numpy as np
import imageio
import json
from collections import defaultdict
from importlib import import_module

import sys
sys.path.append('../code')
from dataset.dataset import NeILFDataset
from model.neilf_brdf import NeILFModel
from utils import general, io

def evaluate(input_data_folder,
             output_model_folder,
             config_path,
             load_phase,
             timestamp,
             checkpoint,
             eval_nvs,
             eval_brdf,
             eval_lighting,
             export_mesh,
             export_nvs,
             export_brdf,
             export_lighting):

    assert os.path.exists(input_data_folder), "Data directory is empty"
    assert os.path.exists(output_model_folder), "Model directorty is empty"
    phase = 'joint'
    if timestamp == 'latest':
        timestamp = io.find_latest(os.path.join(output_model_folder, load_phase))
    assert timestamp, "Model directorty is empty"

    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    # load config file
    # config = io.load_config(os.path.join(output_model_folder, load_phase, timestamp, 'config.json'), phase)
    config = io.load_config(config_path, phase)

    # load input data and create evaluation dataset
    num_pixel_samples = config['train']['num_pixel_samples']
    eval_dataset = NeILFDataset(
        input_data_folder, num_pixel_samples=num_pixel_samples, mode='eval', **config['dataset'])

    geometry_module = import_module(config['model']['geometry_module'])
    kwargs = {}
    if config['model']['geometry_module'] == 'model.geo_fixmesh':
        from pyrt import PyRT
        ray_tracer = PyRT(*eval_dataset.tracer_mesh, 0)
        kwargs['ray_tracer'] = ray_tracer
    if config['model']['geometry_module'] == 'model.geo_volsdf':
        pass
    geometry = geometry_module.Geometry(config['model']['geometry'], phase, **kwargs)

    # create model
    model = NeILFModel(config['model'], geometry, phase)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # load model
    io.load_checkpoint(model, None, None, output_model_folder, '', load_phase, timestamp, checkpoint, True)

    # create evaluation folder
    eval_folder = os.path.join(output_model_folder, load_phase, timestamp, 'evaluation')
    general.mkdir_ifnotexists(eval_folder)

    # evaluate BRDFs and novel view renderings
    if eval_brdf or eval_nvs or export_nvs:

        # results = dict()
        results = defaultdict(lambda: defaultdict(dict))

        # get validation data in the dataset
        model_input, ground_truth = eval_dataset.get_validation_data()
        model_input = {k: v.cuda() for k, v in model_input.items()}

        # split inputs
        total_pixels = eval_dataset.total_pixels
        split_inputs = general.split_neilf_input(model_input, total_pixels)

        # generate outputs
        split_outputs = []
        for split_input in split_inputs:
            with torch.no_grad():
                split_output = model(split_input)
            split_outputs.append(
                {k:split_output[k].detach().cpu() for k in 
                ['rgb_values', 'points', 'normals', 'base_color',
                'roughness', 'metallic', 'render_masks']})

        # merge output
        num_val_images = len(eval_dataset.validation_indexes)
        model_outputs = general.merge_neilf_output(
            split_outputs, total_pixels, num_val_images)

        # image size
        H = eval_dataset.image_resolution[0]
        W = eval_dataset.image_resolution[1]

        # rendered mask
        mask = model_outputs['render_masks']
        mask = mask.reshape([num_val_images, H, W, 1]).float()
        mask_np = mask.numpy()

        # estimated image
        rgb_eval = model_outputs['rgb_values']
        rgb_eval = rgb_eval.reshape([num_val_images, H, W, 3])
        if not config['model']['use_ldr_image']: 
            rgb_eval = general.hdr2ldr(rgb_eval)
        rgb_eval = rgb_eval * mask + (1 - mask)
        rgb_eval_np = rgb_eval.numpy()

        # gt image
        rgb_gt = ground_truth['rgb']
        rgb_gt = rgb_gt.reshape([num_val_images, H, W, 3])
        if not config['model']['use_ldr_image']: 
            rgb_gt = general.hdr2ldr(rgb_gt)
        rgb_gt = rgb_gt * mask + (1 - mask)
        rgb_gt_np = rgb_gt.numpy()

        # evaluate novel view renderings
        if eval_nvs:
            for i in range(num_val_images):
                index = str(eval_dataset.validation_indexes[i])
                results['render']['psnr'][index] = general.calculate_psnr(
                    rgb_gt_np[i], rgb_eval_np[i], mask_np[i])
                # results['render']['ssim'][index] = general.calculate_ssim(
                #     rgb_gt_np[i], rgb_eval_np[i], mask_np[i])
                # results['render']['lpips'][index] = general.calculate_lpips(
                #     rgb_gt_np[i], rgb_eval_np[i], mask_np[i])

        if export_nvs:
            rgb_export = (rgb_eval_np.clip(0, 1) * 255).astype(np.uint8)
            for i in range(num_val_images):
                index = str(eval_dataset.validation_indexes[i])
                imageio.imwrite(os.path.join(eval_folder, f"nvs{index}.png"), rgb_export[i])

        # evaluate BRDFs
        if eval_brdf:

            # estimated BRDF
            base_eval = model_outputs['base_color'].reshape([num_val_images, H, W, 3])    
            base_eval = base_eval * mask + (1 - mask)
            roug_eval = model_outputs['roughness'].reshape([num_val_images, H, W, 1])    
            meta_eval = model_outputs['metallic'].reshape([num_val_images, H, W, 1])
            base_eval_np = base_eval.numpy()
            roug_eval_np = roug_eval.numpy()
            meta_eval_np = meta_eval.numpy()

            # gt BRDF
            base_gt = ground_truth['base_color'].reshape([num_val_images, H, W, 3])
            roug_gt = ground_truth['roughness'].reshape([num_val_images, H, W, 1])
            meta_gt = ground_truth['metallic'].reshape([num_val_images, H, W, 1])
            base_gt_np = base_gt.numpy()
            roug_gt_np = roug_gt.numpy()
            meta_gt_np = meta_gt.numpy()

            # adjust base color
            mask_bool_np = mask_np[...,0]>0.5
            est_med = np.median(base_eval_np[mask_bool_np])
            gt_med = np.median(base_gt_np[mask_bool_np])
            gamma = np.log(gt_med) / np.log(est_med)
            print(est_med, gt_med, gamma)
            base_eval_np = base_eval_np.clip(0,1) ** gamma

            for i in range(num_val_images):
                index = str(eval_dataset.validation_indexes[i])
                # base color
                results['base_color']['psnr'][index] = general.calculate_psnr(
                    base_gt_np[i], base_eval_np[i], mask_np[i])
                # results['base_color']['ssim'][index] = general.calculate_ssim(
                #     base_gt_np[i], base_eval_np[i], mask_np[i])
                # results['base_color']['lpips'][index] = general.calculate_lpips(
                #     base_gt_np[i], base_eval_np[i], mask_np[i])
                # roughness
                results['roughness']['psnr'][index] = general.calculate_psnr(
                    roug_gt_np[i], roug_eval_np[i], mask_np[i])
                # results['roughness']['ssim'][index] = general.calculate_ssim(
                #     roug_gt_np[i], roug_eval_np[i], mask_np[i])
                # results['roughness']['lpips'][index] = general.calculate_lpips(
                #     roug_gt_np[i], roug_eval_np[i], mask_np[i])
                # metallic
                results['metallic']['psnr'][index] = general.calculate_psnr(
                    meta_gt_np[i], meta_eval_np[i], mask_np[i])
                # results['metallic']['ssim'][index] = general.calculate_ssim(
                #     meta_gt_np[i], meta_eval_np[i], mask_np[i])
                # results['metallic']['lpips'][index] = general.calculate_lpips(
                #     meta_gt_np[i], meta_eval_np[i], mask_np[i])
    
        # calculate mean scores
        for item in results:
            for metric in results[item]:
                results[item][metric]['mean'] = 0
                for i in range(num_val_images):
                    index = str(eval_dataset.validation_indexes[i])
                    results[item][metric]['mean'] += results[item][metric][index]
                results[item][metric]['mean'] /= num_val_images
        
        # print results
        for item in results:
            print (item + ' evaluation:')
            for metric in results[item]:
                print ('  mean ' + metric + ': ' + str(results[item][metric]['mean']))

        # save results
        eval_report_path = os.path.join(eval_folder, 'report_evaluation.json')
        with open(eval_report_path, 'w') as eval_report:
            json.dump(results, eval_report, indent=4)

    if export_mesh:
        mesh_resolution = config['eval'].get('full_resolution', 512)
        model.geometry.get_surface_trace(eval_folder, 'export', eval_dataset.scale_mat, resolution=mesh_resolution)

    # export BRDF as texture maps
    if export_brdf:
        for obj_name, tex_coord, coord_shape, coord_mask in eval_dataset.tex_coords:
            split_outputs = []
            batch_size = 500000
            for start in range(0, tex_coord.shape[0], batch_size):
                end = start + batch_size
                split_coord = tex_coord[start:end].cuda()
                brdf_output = model.neilf_pbr.sample_brdfs(split_coord)
                brdf_output = [arr.detach().cpu().numpy() for arr in brdf_output]
                split_outputs.append(brdf_output)
            valid_output = [np.concatenate(out, axis=0) for out in zip(*split_outputs)]
            for i, (brdf_name, num_channels) in enumerate(zip(['base_color', 'roughness', 'metallic'], [3,1,1])):
                out_map = np.zeros(list(coord_shape) + [num_channels])
                # coord_mask = coord_mask.reshape([-1])
                out_map[coord_mask] = valid_output[i]
                if num_channels == 1:
                    out_map = out_map.squeeze(-1)
                imageio.imwrite(os.path.join(eval_folder, f"{obj_name}_{brdf_name}.png"), (out_map.clip(0,1)*255).astype(np.uint8))
    
    # TODO: more evaluation options
    if eval_lighting: 
        print ('eval_lighting is not implemented yet')

    if export_lighting:
        env_width = 640
        plot_env = config["eval"]["plot_env"]
        model_input, ground_truth = eval_dataset.validation_data
        W = eval_dataset.image_resolution[1]
        for i, p in enumerate(plot_env):
            plot_point = model_input["positions"][p[0], p[2]*W+p[1]]
            env_map, slf_map = [v.detach().cpu().numpy() for v in model.plot_point_env(plot_point.float().cuda(), env_width)]
            for n, map in [('env', env_map), ('slf', slf_map)]:
                map = map[:, ::-1]
                imageio.imwrite(os.path.join(eval_folder, f'{n}{i}.exr'), map)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_data_folder', type=str, 
                        help='input folder of images, cameras and geometry files.')
    parser.add_argument('output_model_folder', type=str, 
                        help='folder containing trained models, and for saving results')
    parser.add_argument('--config_path', type=str, 
                        default='./configs/config_dtu_volsdf_ngp.json')    
    parser.add_argument('--phase', type=str, choices=['geo', 'mat', 'joint'], default='joint')

    # checkpoint
    parser.add_argument('--timestamp', default='latest',
                        type=str, help='the timestamp of the model to be evaluated.')
    parser.add_argument('--checkpoint', default='latest',
                        type=str, help='the checkpoint of the model to be evaluated')
    
    # items to evaluate
    parser.add_argument('--eval_nvs', action='store_true', default=False, 
                        help="evaluate novel view renderings")
    parser.add_argument('--eval_brdf', action='store_true', default=False, 
                        help="evaluate BRDF maps at novel views")
    parser.add_argument('--eval_lighting', action='store_true', default=False, 
                        help="work in progress, not ready yet")
    parser.add_argument('--export_mesh', action='store_true', default=False, 
                        help="export mesh")
    parser.add_argument('--export_nvs', action='store_true', default=False, 
                        help="export novel view renderings")
    parser.add_argument('--export_brdf', action='store_true', default=False, 
                        help="export BRDF as texture maps")
    parser.add_argument('--export_lighting', action='store_true', default=False, 
                        help="export incident lights at certain positions")

    args = parser.parse_args()

    evaluate(args.input_data_folder,
             args.output_model_folder,
             args.config_path,
             args.phase,
             args.timestamp,
             args.checkpoint,
             args.eval_nvs,
             args.eval_brdf,
             args.eval_lighting,
             args.export_mesh,
             args.export_nvs,
             args.export_brdf,
             args.export_lighting)