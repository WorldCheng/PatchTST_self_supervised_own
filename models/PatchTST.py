__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        
        
        # model
        self.context_window = context_window
        self.multiscale_patch_lens = [8, 16, 32]
        self.freq_gating = FrequencyGatingUnit(num_scales=len(self.multiscale_patch_lens))
        backbone_kwargs = dict(
            c_in=c_in,
            context_window=context_window,
            target_window=target_window,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            pretrain_head=pretrain_head,
            head_type=head_type,
            individual=individual,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            verbose=verbose,
            **kwargs,
        )

        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = self._build_multiscale_backbones(backbone_kwargs, stride)
            self.model_res = self._build_multiscale_backbones(backbone_kwargs, stride)
        else:
            self.model = self._build_multiscale_backbones(backbone_kwargs, stride)

    def _build_multiscale_backbones(self, backbone_kwargs, default_stride):
        branches = nn.ModuleList()
        for patch_len in self.multiscale_patch_lens:
            patch_len = min(patch_len, self.context_window)
            stride = max(1, min(default_stride, patch_len))
            branches.append(PatchTST_backbone(patch_len=patch_len, stride=stride, **backbone_kwargs))
        return branches

    def _forward_multiscale(self, x, branches, weights):
        branch_outputs = [branch(x) for branch in branches]
        mixed_output = 0
        for idx, branch_output in enumerate(branch_outputs):
            mixed_output = mixed_output + branch_output * weights[:, idx].view(-1, 1, 1)
        return mixed_output
    
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        weights = self.freq_gating(x)
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self._forward_multiscale(res_init, self.model_res, weights)
            trend = self._forward_multiscale(trend_init, self.model_trend, weights)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self._forward_multiscale(x, self.model, weights)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x


class FrequencyGatingUnit(nn.Module):
    def __init__(self, num_scales:int):
        super().__init__()
        self.num_scales = num_scales
        self.gate = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, num_scales)
        )

    def forward(self, x):
        centered_x = x - x.mean(dim=1, keepdim=True)
        fft_energy = torch.fft.rfft(centered_x, dim=1).abs().pow(2).mean(dim=-1)

        n_freq_bins = fft_energy.shape[1]
        split_idx = max(1, n_freq_bins // 2)
        low_band = fft_energy[:, :split_idx].mean(dim=1)
        high_band = fft_energy[:, split_idx:].mean(dim=1) if split_idx < n_freq_bins else low_band.new_zeros(low_band.shape)

        total_energy = low_band + high_band + 1e-6
        low_ratio = low_band / total_energy
        high_ratio = high_band / total_energy
        ratio_feature = torch.stack([low_ratio, high_ratio], dim=-1)

        learned_logits = self.gate(ratio_feature)
        high_freq_bias = torch.stack([high_ratio, torch.zeros_like(high_ratio), -high_ratio], dim=-1)
        return torch.softmax(learned_logits + high_freq_bias, dim=-1)
