__all__ = ['PatchTST']

# Cell
from typing import Optional
import math
import torch
from torch import nn
from torch import Tensor

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto', padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type='flatten', verbose:bool=False, **kwargs):
        
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
        self.target_window = target_window
        self._resolve_multiscale_configs(configs, patch_len, stride)

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

        num_scales = len(self.multiscale_patch_lens)
        gating_kwargs = dict(
            num_scales=num_scales,
            channel_wise=self.channel_wise_gating,
            freq_feature_mode=self.freq_feature_mode,
            num_bands=self.freq_num_bands,
        )

        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            # trend/res use independent gating to decouple scale preference
            self.freq_gating_res = FrequencyGatingUnit(**gating_kwargs)
            self.freq_gating_trend = FrequencyGatingUnit(**gating_kwargs)
            self.model_trend = self._build_multiscale_backbones(backbone_kwargs)
            self.model_res = self._build_multiscale_backbones(backbone_kwargs)
        else:
            self.freq_gating = FrequencyGatingUnit(**gating_kwargs)
            self.model = self._build_multiscale_backbones(backbone_kwargs)

    def _resolve_multiscale_configs(self, configs, base_patch_len:int, base_stride:int):
        # patch lens: configurable, with safe clipping by context_window
        patch_lens = getattr(configs, 'multiscale_patch_lens', None)
        if patch_lens is None:
            patch_lens = [8, 16, 32]
        patch_lens = [max(1, min(int(p), self.context_window)) for p in list(patch_lens)]
        if len(patch_lens) == 0:
            raise ValueError('multiscale_patch_lens must contain at least one patch length.')
        self.multiscale_patch_lens = patch_lens

        # gating configs
        self.channel_wise_gating = bool(getattr(configs, 'channel_wise_gating', True))
        self.freq_feature_mode = (getattr(configs, 'freq_feature_mode', 'enhanced') or 'enhanced').lower()
        self.freq_num_bands = max(3, int(getattr(configs, 'freq_num_bands', 3)))

        # stride config: manual or auto-linked
        multiscale_strides = getattr(configs, 'multiscale_strides', None)
        if multiscale_strides is not None:
            multiscale_strides = list(multiscale_strides)
            if len(multiscale_strides) != len(self.multiscale_patch_lens):
                raise ValueError(
                    f'multiscale_strides length ({len(multiscale_strides)}) must match '
                    f'multiscale_patch_lens length ({len(self.multiscale_patch_lens)}).'
                )
            cleaned_strides = []
            for stride_val in multiscale_strides:
                stride_val = int(stride_val)
                if stride_val < 1:
                    raise ValueError('multiscale_strides must be positive integers.')
                cleaned_strides.append(stride_val)
            self.multiscale_strides = cleaned_strides
            self.stride_ratio = None
            self._stride_mode = 'manual'
        else:
            stride_ratio = getattr(configs, 'multiscale_stride_ratio', None)
            if stride_ratio is None:
                base_patch_len = max(1, int(base_patch_len))
                base_stride = max(1, int(base_stride))
                stride_ratio = max(1, int(round(base_patch_len / base_stride)))
            else:
                stride_ratio = max(1, int(stride_ratio))
            self.stride_ratio = stride_ratio
            self._stride_mode = 'auto'
            # stride follows patch_len proportionally
            self.multiscale_strides = [
                max(1, min(patch_len, patch_len // stride_ratio))
                for patch_len in self.multiscale_patch_lens
            ]

    def _build_multiscale_backbones(self, backbone_kwargs):
        branches = nn.ModuleList()
        for patch_len, stride in zip(self.multiscale_patch_lens, self.multiscale_strides):
            patch_len = max(1, min(int(patch_len), self.context_window))
            stride = max(1, int(stride))
            branches.append(PatchTST_backbone(patch_len=patch_len, stride=stride, **backbone_kwargs))
        return branches

    def _forward_multiscale(self, x, branches, weights):
        # x: [B, C, L], weights: [B, C, S] (channel-wise) or [B, S] (sample-wise)
        if weights.dim() == 2:
            weight_view = weights[:, None, :]  # [B, 1, S] for broadcast over channels
        else:
            weight_view = weights  # [B, C, S]

        mixed_output = None
        for idx, branch in enumerate(branches):
            branch_output = branch(x)  # [B, C, target_window]
            scale_weight = weight_view[:, :, idx].unsqueeze(-1)  # [B, C, 1] or [B, 1, 1]
            weighted = branch_output * scale_weight
            mixed_output = weighted if mixed_output is None else mixed_output + weighted
        return mixed_output
    
    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            # series_decomp returns [B, L, C]; trend/res use independent gating
            res_init, trend_init = self.decomp_module(x)
            weights_res = self.freq_gating_res(res_init)
            weights_trend = self.freq_gating_trend(trend_init)

            res_init = res_init.permute(0, 2, 1)   # [B, C, L] for backbone
            trend_init = trend_init.permute(0, 2, 1)

            res = self._forward_multiscale(res_init, self.model_res, weights_res)
            trend = self._forward_multiscale(trend_init, self.model_trend, weights_trend)
            x = res + trend
            x = x.permute(0, 2, 1)    # [B, L, C]
        else:
            weights = self.freq_gating(x)
            x = x.permute(0, 2, 1)    # [B, C, L]
            x = self._forward_multiscale(x, self.model, weights)
            x = x.permute(0, 2, 1)    # [B, L, C]
        return x


class FrequencyGatingUnit(nn.Module):
    def __init__(self, num_scales:int, channel_wise:bool=True, freq_feature_mode:str='enhanced', num_bands:int=3):
        super().__init__()
        self.num_scales = num_scales
        self.channel_wise = channel_wise
        self.freq_feature_mode = (freq_feature_mode or 'enhanced').lower()
        self.num_bands = max(3, int(num_bands))
        self.eps = 1e-6

        self.feature_dim = self._feature_dim()
        hidden_dim = max(16, self.feature_dim * 4)
        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales)
        )

    def _feature_dim(self) -> int:
        # basic: low/mid/high; enhanced: + dominant idx + spectral entropy + peak ratio
        if self.freq_feature_mode == 'basic':
            return 3
        return 6

    @staticmethod
    def _split_sizes(total:int, parts:int):
        parts = max(1, parts)
        base = total // parts
        rem = total % parts
        return [base + (1 if i < rem else 0) for i in range(parts)]

    def _compute_band_ratios(self, energy:Tensor, total_energy:Tensor, n_freq_bins:int) -> Tensor:
        # energy: [B, F, C], total_energy: [B, C]
        num_bands = min(self.num_bands, n_freq_bins) if n_freq_bins > 0 else 1
        sizes = self._split_sizes(n_freq_bins, num_bands)
        band_ratios = []
        start = 0
        for size in sizes:
            end = start + size
            if size > 0:
                band_energy = energy[:, start:end, :].sum(dim=1)
            else:
                band_energy = total_energy.new_zeros(total_energy.shape)
            band_ratios.append(band_energy / total_energy)
            start = end
        return torch.stack(band_ratios, dim=-1)  # [B, C, num_bands]

    def _aggregate_low_mid_high(self, band_ratios:Tensor):
        # band_ratios: [B, C, num_bands]
        num_bands = band_ratios.shape[-1]
        if num_bands == 1:
            low = band_ratios[..., 0]
            mid = low.new_zeros(low.shape)
            high = low.new_zeros(low.shape)
            return low, mid, high
        if num_bands == 2:
            low = band_ratios[..., 0]
            mid = low.new_zeros(low.shape)
            high = band_ratios[..., 1]
            return low, mid, high

        sizes = self._split_sizes(num_bands, 3)
        idx = 0
        low = band_ratios[..., idx:idx + sizes[0]].sum(dim=-1)
        idx += sizes[0]
        mid = band_ratios[..., idx:idx + sizes[1]].sum(dim=-1)
        idx += sizes[1]
        high = band_ratios[..., idx:idx + sizes[2]].sum(dim=-1)
        return low, mid, high

    def _extract_frequency_features(self, x:Tensor):
        # x: [B, L, C]
        centered_x = x - x.mean(dim=1, keepdim=True)
        fft_energy = torch.fft.rfft(centered_x, dim=1).abs().pow(2)  # [B, F, C]
        n_freq_bins = fft_energy.shape[1]

        total_energy = fft_energy.sum(dim=1) + self.eps  # [B, C]
        band_ratios = self._compute_band_ratios(fft_energy, total_energy, n_freq_bins)
        low_ratio, mid_ratio, high_ratio = self._aggregate_low_mid_high(band_ratios)

        # dominant frequency index (normalized)
        if n_freq_bins > 1:
            dominant_idx = fft_energy.argmax(dim=1).float() / float(n_freq_bins - 1)
        else:
            dominant_idx = total_energy.new_zeros(total_energy.shape)

        # spectral entropy (normalized)
        if n_freq_bins > 1:
            prob = (fft_energy / total_energy.unsqueeze(1)).clamp_min(self.eps)
            spectral_entropy = -(prob * torch.log(prob)).sum(dim=1) / math.log(n_freq_bins)
        else:
            spectral_entropy = total_energy.new_zeros(total_energy.shape)

        # peak energy ratio
        peak_energy_ratio = fft_energy.max(dim=1).values / total_energy

        if self.freq_feature_mode == 'basic':
            features = torch.stack([low_ratio, mid_ratio, high_ratio], dim=-1)
        else:
            features = torch.stack(
                [low_ratio, mid_ratio, high_ratio, dominant_idx, spectral_entropy, peak_energy_ratio],
                dim=-1
            )

        if not self.channel_wise:
            # aggregate to sample-wise gating
            features = features.mean(dim=1)
            low_ratio = low_ratio.mean(dim=1)
            high_ratio = high_ratio.mean(dim=1)

        stats = {
            'low_ratio': low_ratio,
            'high_ratio': high_ratio,
        }
        return features, stats

    def _build_frequency_bias(self, low_ratio:Tensor, high_ratio:Tensor):
        # continuous bias from small-scale (high freq) to large-scale (low freq)
        base_bias = torch.linspace(1.0, -1.0, steps=self.num_scales, device=high_ratio.device, dtype=high_ratio.dtype)
        freq_skew = high_ratio - low_ratio  # >0 => more high-frequency energy
        if freq_skew.dim() == 2:
            return base_bias.view(1, 1, -1) * freq_skew.unsqueeze(-1)  # [B, C, S]
        return base_bias.view(1, -1) * freq_skew.unsqueeze(-1)  # [B, S]

    def forward(self, x:Tensor):
        features, stats = self._extract_frequency_features(x)
        if self.channel_wise:
            bsz, channels, feat_dim = features.shape
            logits = self.gate(features.reshape(bsz * channels, feat_dim)).reshape(bsz, channels, self.num_scales)
        else:
            logits = self.gate(features)
        freq_bias = self._build_frequency_bias(stats['low_ratio'], stats['high_ratio'])
        return torch.softmax(logits + freq_bias, dim=-1)
