#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn

from .embedder import get_embedder, ComposedEmbedder

class SineLayer(nn.Module):
    ''' Siren layer '''
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True, 
                 is_first=False, 
                 omega_0=30, 
                 weight_norm=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)

    def init_weights(self):
        if self.is_first:
            nn.init.uniform_(self.linear.weight, 
                             -1 / self.in_features * self.omega_0, 
                             1 / self.in_features * self.omega_0)
        else:
            nn.init.uniform_(self.linear.weight, 
                             -np.sqrt(3 / self.in_features), 
                             np.sqrt(3 / self.in_features))
        nn.init.zeros_(self.linear.bias)

    def forward(self, input):
        return torch.sin(self.linear(input))

class BRDFMLP(nn.Module):

    def __init__(
            self,
            in_dims,
            out_dims,
            dims,
            skip_connection=(),
            weight_norm=True,
            embed_config=[{'otype':'Identity'}],
            use_siren=True,
    ):
        super().__init__()
        self.init_range = np.sqrt(3 / dims[0])

        dims = [in_dims] + dims + [out_dims]
        first_omega = 30
        hidden_omega = 30

        self.embed_fn = ComposedEmbedder(embed_config)
        dims[0] += self.embed_fn.n_feat_dims

        self.num_layers = len(dims)
        self.skip_connection = skip_connection

        for l in range(0, self.num_layers - 1):

            if l + 1 in self.skip_connection:
                out_dim = dims[l + 1] - dims[0] 
            else:
                out_dim = dims[l + 1]

            siren_is_first = (l == 0) and (self.embed_fn.n_feat_dims == 0)
            is_last = (l == (self.num_layers - 2))
            
            if not is_last:
                if use_siren:
                    omega_0 = first_omega if siren_is_first else hidden_omega
                    lin = SineLayer(dims[l], out_dim, True, siren_is_first, omega_0, weight_norm)
                else:
                    lin = nn.Linear(dims[l], out_dim)
                    nn.init.zeros_(lin.bias)
                    if weight_norm:
                        lin = nn.utils.weight_norm(lin)
                    # if l == 0:
                    #     param = lin.weight_g if weight_norm else lin.weight
                    #     assert type(param) is nn.Parameter
                    #     setattr(param, 'custom_meta', {})
                    #     param.custom_meta['optim_args'] = {'weight_decay': 0.01}
                    lin = nn.Sequential(lin, nn.ReLU(inplace=True))
            else:
                lin = nn.Linear(dims[l], out_dim, bias=False)
                nn.init.uniform_(lin.weight, -1e-5, 1e-5)
                # nn.init.zeros_(lin.bias)
                if weight_norm:
                    lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.last_active_fun = nn.Tanh()

    def forward(self, points):
        init_x = self.embed_fn(points)
        x = init_x

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_connection:
                x = torch.cat([x, init_x], 1) / np.sqrt(2)
                
            x = lin(x)

        x = self.last_active_fun(x)

        return x


class SeparatedBRDFMLP(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.embed_fn = ComposedEmbedder(config['embed_config'])
        config['embed_config'] = [{'otype':'Identity'}]
        config['in_dims'] = self.embed_fn.n_output_dims
        assert config['out_dims'] == 5
        assert not config['use_siren']  # FIXME
        config['out_dims'] = 3; self.b_nn = BRDFMLP(**config)
        config['out_dims'] = 1; self.r_nn = BRDFMLP(**config)
        config['out_dims'] = 1; self.m_nn = BRDFMLP(**config)

    def forward(self, points):
        x = self.embed_fn(points)
        b = self.b_nn(x)
        r = self.r_nn(x)
        m = self.m_nn(x)
        return torch.cat([b,r,m], dim=1)


class NeILFMLP(nn.Module):

    def __init__(
            self,
            in_dims,   # 6
            out_dims,  # 3
            dims,
            dir_insert=(),
            pos_insert=(),
            embed_config_view=[{'otype':'Identity'}],
            embed_config=[{'otype':'Identity'}],
            use_siren=True,
            weight_norm=True,
            init_output=1.0,
            fused=False,
            last_act='Exponential',
            exp_act_base=-1,
            sigmoid_output_scale=1.0,
    ):
        super().__init__()
        self.init_range = np.sqrt(3 / dims[0])

        d_pos = 3
        d_dir = 3
        dims = [0] + dims + [out_dims]

        self.embeddir_fn = ComposedEmbedder(embed_config_view)
        d_dir += self.embeddir_fn.n_feat_dims
        self.embedpos_fn = ComposedEmbedder(embed_config)
        d_pos += self.embedpos_fn.n_feat_dims

        self.dir_insert = dir_insert
        self.pos_insert = pos_insert

        if 0 in self.dir_insert:
            dims[0] += d_dir
        if 0 in self.pos_insert:
            dims[0] += d_pos
        assert dims[0] > 0

        self.num_layers = len(dims)
        first_omega = 30
        hidden_omega = 30

        if last_act == 'Exponential':
            if exp_act_base != -1:
                self.last_active_fun = lambda x: exp_act_base ** x
                self.last_active_fun_inv = lambda x: np.log(x)/np.log(exp_act_base)
            else:
                self.last_active_fun = torch.exp
                self.last_active_fun_inv = np.log
        elif last_act == 'Sigmoid':
            if sigmoid_output_scale > 1:
                self.last_active_fun = lambda x: x.sigmoid() * sigmoid_output_scale
            else:
                self.last_active_fun = nn.Sigmoid()
            self.last_active_fun_inv = lambda x:0
        else:
            raise NotImplementedError

        if init_output < 0:
            init_output = 1.0

        self.fused = fused
        if fused:
            segments = self.pos_insert + self.dir_insert + [0, self.num_layers - 2]
            segments = sorted(list(set(segments)))
            assert all([v == dims[1] for v in dims[1:-1]])
            fused_config = {
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': "ReLU",
                'n_neurons': dims[1],
                'n_hidden_layers': 0,
            }
            for s in range(len(segments) - 1):
                fused_config['n_hidden_layers'] = segments[s+1] - segments[s] - 1
                out_dim = dims[segments[s+1]]
                if segments[s+1] in self.dir_insert:
                    out_dim -= d_dir
                if segments[s+1] in self.pos_insert:
                    out_dim -= d_pos
                fused_net = tcnn.Network(dims[segments[s]], out_dim, fused_config)
                setattr(self, f'fuse{s}', fused_net)
            self.segments = segments

            self.last_lin = nn.Linear(dims[-2], dims[-1])
            nn.init.uniform_(self.last_lin.weight, -1e-5, 1e-5)
            nn.init.constant_(self.last_lin.bias, self.last_active_fun_inv(init_output))
            if weight_norm:
                self.last_lin = nn.utils.weight_norm(self.last_lin)

        else:
            for l in range(0, self.num_layers - 1):

                out_dim = dims[l + 1]
                if l + 1 in self.dir_insert:
                    out_dim -= d_dir
                if l + 1 in self.pos_insert:
                    out_dim -= d_pos

                siren_is_first = (l == 0) and (self.embeddir_fn.n_feat_dims == 0) and (self.embedpos_fn.n_feat_dims == 0)
                is_last = (l == (self.num_layers - 2))

                if not is_last:
                    if use_siren:
                        omega_0 = first_omega if siren_is_first else hidden_omega
                        lin = SineLayer(dims[l], out_dim, True, siren_is_first, omega_0, weight_norm)
                    else:
                        lin = nn.Linear(dims[l], out_dim)
                        nn.init.zeros_(lin.bias)
                        if weight_norm:
                            lin = nn.utils.weight_norm(lin)
                        lin = nn.Sequential(lin, nn.ReLU(inplace=True))
                else:
                    lin = nn.Linear(dims[l], out_dim)
                    nn.init.uniform_(lin.weight, -1e-5, 1e-5)
                    nn.init.constant_(lin.bias, self.last_active_fun_inv(init_output))
                    if weight_norm:
                        lin = nn.utils.weight_norm(lin)

                setattr(self, "lin" + str(l), lin)

    def forward(self, points):

        view_embed = self.embeddir_fn(points[:, 3:6])
        pos_embed = self.embedpos_fn(points[:, 0:3])
        x = torch.zeros(view_embed.shape[0], 0, dtype=view_embed.dtype, device=view_embed.device)

        if self.fused:
            for s in range(len(self.segments) - 1):
                fuse = getattr(self, f'fuse{s}')

                if self.segments[s] in self.dir_insert:
                    x = torch.cat([x, view_embed], 1)

                if self.segments[s] in self.pos_insert:
                    x = torch.cat([x, pos_embed], 1)

                x = fuse(x)
            
            x = self.last_active_fun(self.last_lin(x.float()))

        else:
            for l in range(0, self.num_layers - 1):
                lin = getattr(self, "lin" + str(l))

                if l in self.dir_insert:
                    x = torch.cat([x, view_embed], 1)

                if l in self.pos_insert:
                    x = torch.cat([x, pos_embed], 1)

                x = lin(x)

            x = self.last_active_fun(x)

        return x
