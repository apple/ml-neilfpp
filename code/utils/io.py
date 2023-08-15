#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import numpy as np
import imageio
import pyexr
import skimage
import json
import torch

def load_gray_image(path):
    ''' Load gray scale image (both uint8 and float32) into image in range [0, 1] '''
    ext = os.path.splitext(path)[1]
    if ext in ['.png', '.jpg']:
        image = imageio.imread(path, mode='L')                      # [H, W]
        if image.dtype == np.float32 and image.max() > 1:
            image /= 255
    elif ext in ['.tiff', '.exr']:
        if ext == '.exr':
            # NOTE imageio read exr has artifact https://github.com/imageio/imageio/issues/517
            image = pyexr.read(path)
        else:
            image = imageio.imread(path)                                    # [H, W]
        if len(image.shape) > 2:
            print ('Reading rgbfloat32 image as gray, will use the first channel')
            image = image[:, :, 0]
    image = skimage.img_as_float32(image)
    return image

def load_gray_image_with_prefix(prefix):
    ''' Load image using prefix to support different data type '''
    exts = ['.png', '.jpg', '.tiff', '.exr']
    for ext in exts:
        path = prefix + ext
        if os.path.exists(path):
            return load_gray_image(path)
    print ('Does not exists any image file with prefix: ' + prefix)
    return None

def load_rgb_image(path):
    ''' Load RGB image (both uint8 and float32) into image in range [0, 1] '''
    ext = os.path.splitext(path)[1]
    if ext == '.exr':
        # NOTE imageio read exr has artifact https://github.com/imageio/imageio/issues/517
        image = pyexr.read(path)
    else:
        image = imageio.imread(path)
    if image.shape[-1] > 3:
        image = image[..., :3]                          # [H, W, 4] -> [H, W ,3]
    image = skimage.img_as_float32(image)
    return image


def load_rgb_image_with_prefix(prefix):
    ''' Load image using prefix to support different data type '''
    exts = ['.png', '.jpg', '.tiff', '.exr']
    for ext in exts:
        path = prefix + ext
        if os.path.exists(path):
            return load_rgb_image(path)
    print ('Does not exists any image file with prefix: ' + prefix)
    return None


def save_image(path, image):
    imageio.imwrite(path, image)

def load_mask_image(path):
    alpha = imageio.imread(path, mode='L')                      # [H, W]
    alpha = skimage.img_as_float32(alpha)
    object_mask = alpha > 127.5
    return object_mask

def load_tex_coord(path, mask=None):
    # NOTE imageio read exr has artifact https://github.com/imageio/imageio/issues/517
    # coord_image = imageio.imread(path)[..., 0:3]                    # [H, W, 4] -> [H, W ,3]
    coord_image = pyexr.read(path)
    if mask is not None:
        mask_image = imageio.imread(mask, mode='L') > 127.5
    else:
        mask_image = np.ones(coord_image.shape[:2], dtype=np.bool_)
    return coord_image, mask_image

def load_cams_from_sfmscene(path):

    # load json file
    with open(path) as f:
        sfm_scene = json.load(f)

    # camera parameters
    intrinsics = dict()
    extrinsics = dict()
    camera_info_list = sfm_scene['camera_track_map']['images']
    for i, (index, camera_info) in enumerate(camera_info_list.items()):
        # flg == 2 stands for valid camera 
        if camera_info['flg'] == 2:
            intrinsic = np.zeros((4, 4))
            intrinsic[0, 0] = camera_info['camera']['intrinsic']['focal'][0]
            intrinsic[1, 1] = camera_info['camera']['intrinsic']['focal'][1]
            intrinsic[0, 2] = camera_info['camera']['intrinsic']['ppt'][0]
            intrinsic[1, 2] = camera_info['camera']['intrinsic']['ppt'][1]
            intrinsic[2, 2] = intrinsic[3, 3] = 1
            extrinsic = np.array(camera_info['camera']['extrinsic']).reshape(4, 4)
            intrinsics[index] = intrinsic
            extrinsics[index] = extrinsic

    # load bbox transform
    bbox_transform = np.array(sfm_scene['bbox']['transform']).reshape(4, 4)

    # compute scale_mat for coordinate normalization
    scale_mat = bbox_transform.copy()
    scale_mat[[0,1,2],[0,1,2]] = scale_mat[[0,1,2],[0,1,2]].max() / 2
    
    # meta info
    image_list = sfm_scene['image_path']['file_paths']
    image_indexes = [str(k) for k in sorted([int(k) for k in image_list])]
    resolution = camera_info_list[image_indexes[0]]['size'][::-1]

    return intrinsics, extrinsics, scale_mat, image_list, image_indexes, resolution

def traversal(d, parent=[]):
    for k,v in d.items():
        if type(v) is dict:
            yield from traversal(v, parent+[k])
        else:
            yield parent+[k], v

def modify_dict(d, path, v, convert_type=False):
    curr = d
    for k in path[:-1]:
        if k not in curr:
            curr[k] = {}
        curr = curr[k]
    if convert_type and path[-1] in curr and type(curr[path[-1]]) != type(v):
        v = type(curr[path[-1]])(v)
    curr[path[-1]] = v

def load_config(path, phase):
    with open(path) as f:
        config = json.load(f)

    if 'phase' in config['train'] and phase in config['train']['phase']:
        print(f'apply config modification for {phase}')
        phase_mod = config['train']['phase'][phase]
        del config['train']['phase']

        for kpath, v in traversal(phase_mod):
            if os.environ.get('NEILFPP_SEP', '0') == '1' and kpath[-1] not in ['mat2geo_grad_scale']:
                continue
            modify_dict(config['train'], kpath, v)

    return config

def find_latest(path, exclude=[]):
    items = [i for i in os.listdir(path) if i not in exclude]
    if len(items) > 0:
        return sorted(items)[-1]
    else:
        raise ValueError('Cannot find runs.')

def load_checkpoint(
        model,
        optimizer,
        scheduler,
        output_folder,
        timestamp,
        phase,
        last_timestamp,
        last_checkpoint,
        is_continue
        ):
    ret = {}

    if is_continue:
        load_phase = phase
    elif phase == 'mat':
        load_phase = 'geo'
    elif phase == 'joint':
        load_phase = 'mat'
    else:
        raise ValueError
    load_folder = os.path.join(output_folder, load_phase)

    # find latest timestamp
    if last_timestamp == 'latest':
        last_timestamp = find_latest(load_folder, exclude=[timestamp])
    ret['last_timestamp'] = last_timestamp

    # load pre-trained model
    # paths
    prefix = str(last_checkpoint) + '.pth'
    last_checkpoint_folder = os.path.join(
        load_folder, last_timestamp, 'checkpoints')
    model_path = os.path.join(last_checkpoint_folder, 'ModelParameters', prefix)
    # load
    print(f'load {model_path}')
    model_params = torch.load(model_path)
    model_state_dict = model_params['model_state_dict']
    # filter parameters
    strict_load = True
    if (phase == 'geo') or (phase == 'mat' and not is_continue):
        model_state_dict = {k:v for k,v in model_state_dict.items() if k.startswith('geometry')}
        strict_load = False
    # set model
    model.load_state_dict(model_state_dict, strict=strict_load)
    # continue
    if is_continue and os.environ.get('NEILFPP_SEP', '0') != '1':
        for torch_module, key, path in [
            (optimizer, 'optimizer', 'Optimizer'),
            (scheduler, 'scheduler', 'Scheduler'),
        ]:
            if torch_module:
                load_path = os.path.join(last_checkpoint_folder, f'{path}Parameters', prefix)
                load_params = torch.load(load_path)
                torch_module.load_state_dict(load_params[f'{key}_state_dict'])
        ret['start_iteration'] = model_params['iteration']

    return ret

def save_checkpoint(model, optimizer, scheduler, checkpoint_folder, iteration, is_final=False):
    for torch_module, key, path in [
        (model, 'model', 'Model'),
        (optimizer, 'optimizer', 'Optimizer'),
        (scheduler, 'scheduler', 'Scheduler'),
    ]:
        if torch_module:
            if not is_final:
                torch.save(
                    {'iteration': iteration, f'{key}_state_dict': torch_module.state_dict()},
                    os.path.join(checkpoint_folder, f'{path}Parameters', f'{iteration}.pth'))
            torch.save(
                {'iteration': iteration, f'{key}_state_dict': torch_module.state_dict()},
                os.path.join(checkpoint_folder, f'{path}Parameters', 'latest.pth'))
