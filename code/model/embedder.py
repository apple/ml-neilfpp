#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn

""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

class ScaledTCNN(nn.Module):
    def __init__(self, input_min, input_max, input_dim, config, dtype):
        super().__init__()
        self.fn = tcnn.Encoding(input_dim, config, dtype=dtype)
        self.input_min = input_min
        self.input_range = max(input_max - input_min, 1e-9)

    def forward(self, x):
        x.requires_grad_()
        x = (x - self.input_min) / self.input_range
        return self.fn(x)

class ComposedEmbedder(nn.Module):
    def __init__(self, config, input_dim=3):
        super().__init__()
        self.input_dim = input_dim
        func_map = {
            'Identity': self.identity_embedder,
            'Frequency': self.freq_embedder,
            'HashGrid': self.scaled_tcnn_embedder,
            'SphericalHarmonics': self.scaled_tcnn_embedder,
        }
        self.embedders = []
        self.n_feat_dims = 0
        for econf in config:
            efn_name = f'efn_{econf["otype"]}'
            efn, feat_num = func_map[econf['otype']](**econf)
            setattr(self, efn_name, efn)
            self.embedders.append(efn_name)
            self.n_feat_dims += feat_num
        self.n_output_dims = (input_dim if 'efn_Identity' in self.embedders else 0) + self.n_feat_dims
    
    def identity_embedder(self, otype, scale=1, offset=0, **kwargs):
        return lambda x: x*scale+offset, 0
    
    def freq_embedder(self, otype, n_frequencies, **kwargs):
        embed_kwargs = {
            'include_input': False,
            'input_dims': self.input_dim,
            'max_freq_log2': n_frequencies-1,
            'num_freqs': n_frequencies,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        embedder_obj = Embedder(**embed_kwargs)
        return embedder_obj.embed, embedder_obj.out_dim
    
    def scaled_tcnn_embedder(self, **config):
        input_min, input_max = config['input_range']
        if config['otype'] == 'HashGrid':
            config['per_level_scale'] = np.exp((np.log(config['max_resolution']) - np.log(config['base_resolution'])) / (config['n_levels'] - 1))
            del config['max_resolution']
        tcnn_embed_fn = ScaledTCNN(input_min, input_max, self.input_dim, config, dtype=torch.float)
        if config['otype'] == 'HashGrid':
            init_range = config.get('init_range', 1e-4)
            nn.init.uniform_(tcnn_embed_fn.fn.params, -init_range, init_range)
            # setattr(tcnn_embed_fn.fn.params, 'custom_meta', {})
            # tcnn_embed_fn.fn.params.custom_meta['optim_args'] = {'lr_scale': 10}
        return tcnn_embed_fn, tcnn_embed_fn.fn.n_output_dims

    def forward(self, x):
        return torch.cat([getattr(self, e)(x) for e in self.embedders], dim=-1)