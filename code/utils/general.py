#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import torch
import numpy as np
from skimage.metrics import structural_similarity as SSIM, peak_signal_noise_ratio as PSNR
import lpips
from tqdm import tqdm, trange

lpips_func = None

def demask(masked_tensor, mask, bg=0):
    output = torch.full((mask.shape[0], *masked_tensor.shape[1:]), bg,  \
        dtype=masked_tensor.dtype, device=masked_tensor.device)
    output[mask] = masked_tensor
    return output

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def split_neilf_input(input, total_pixels):
    '''
     Split the input to fit Cuda memory for large resolution.
     Can decrease the value of split_num in case of cuda out of memory error.
     '''
    split_size = 1000
    for start_index in trange(0, total_pixels, split_size):
        indexes = slice(start_index, start_index+split_size)
        data = {k: v[:,indexes] for k, v in input.items() if k not in ['intrinsics', 'pose']}
        chunk_size = data['uv'].shape[1]
        data['intrinsics'] = input['intrinsics'].repeat([1, chunk_size, 1, 1])
        data['pose'] = input['pose'].repeat([1, chunk_size, 1, 1])
        yield data

def merge_neilf_output(res, total_pixels, batch_size):
    ''' Merge the split output. '''
    model_outputs = {}
    for entry in res[0]:
        if res[0][entry] is None:
            continue
        if len(res[0][entry].shape) == 1:
            model_outputs[entry] = torch.cat(
                [r[entry].reshape(batch_size, -1, 1) for r in res], 1
                ).reshape(batch_size * total_pixels)
        else:
            model_outputs[entry] = torch.cat(
                [r[entry].reshape(batch_size, -1, r[entry].shape[-1]) for r in res], 1
                ).reshape(batch_size * total_pixels, -1)
    return model_outputs

def calculate_ssim(img1, img2, mask=None):
    if mask is None: 
        mask = np.ones_like(img1)
    img1, img2 = [(arr * mask).astype(np.float64) for arr in [img1, img2]]
    return SSIM(img1, img2, multichannel=True, data_range=1.0)

def calculate_psnr(img1, img2, mask=None):
    if mask is None: 
        mask = np.ones_like(img1)
    img1, img2 = [(arr * mask).astype(np.float64) for arr in [img1, img2]]
    return PSNR(img1, img2, data_range=1.0)

def calculate_lpips(img1, img2, mask=None):
    global lpips_func
    if lpips_func is None:
        lpips_func = lpips.LPIPS(net='alex').cuda()
    if mask is not None:
        img1, img2 = [arr * mask for arr in [img1, img2]]
    if type(img1) == np.ndarray: img1 = torch.from_numpy(img1).float().cuda()
    if type(img2) == np.ndarray: img2 = torch.from_numpy(img2).float().cuda()
    img1, img2 = [arr * 2 - 1 for arr in [img1, img2]]
    img1, img2 = [arr.permute(2,0,1).unsqueeze(0) for arr in [img1, img2]]
    return lpips_func(img1, img2).item()

def hdr2ldr(img, scale=1):
    img = img * scale
    # img = 1 - np.exp(-3.0543 * img)  # Filmic
    img = (img * (2.51 * img + 0.03)) / (img * (2.43 * img + 0.59) + 0.14)  # ACES
    return img

class SoftClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, dL_dout):
        # identity
        return dL_dout.clone(), None, None

def soft_clip(input, min=None, max=None):
    return SoftClip.apply(input, min, max)

class ScaleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.set_materialize_grads(False)
        ctx.scale = scale
        return x
    @staticmethod
    def backward(ctx, g):
        if g is None:
            return None, None
        return g * ctx.scale, None

def scale_grad(x, scale, mask=None):
    if scale == 0:
        return x.detach()
    elif scale == 1:
        if mask is None:
            return x
        else:
            return torch.where(mask, x, x.detach())
    else:
        if mask is None:
            return ScaleGrad.apply(x, scale)
        else:
            return torch.where(mask, ScaleGrad.apply(x, scale), x.detach())

