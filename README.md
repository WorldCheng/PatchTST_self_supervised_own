# PatchTST_self_supervised_own

**多尺度与频域配置示例**
下面示例基于 `run_longExp.py`。多尺度参数使用空格分隔的列表形式。

自动联动 stride（根据 patch_len 自动生成 stride）:
```bash
python run_longExp.py --is_training 1 --model_id demo --model PatchTST \
  --seq_len 96 --pred_len 96 --enc_in 7 --c_out 7 \
  --multiscale_patch_lens 8 16 32 \
  --multiscale_stride_ratio 2 \
  --channel_wise_gating 1 \
  --freq_feature_mode enhanced \
  --freq_num_bands 4
```

手动配置 stride（每个 patch_len 对应一个 stride）:
```bash
python run_longExp.py --is_training 1 --model_id demo --model PatchTST \
  --seq_len 96 --pred_len 96 --enc_in 7 --c_out 7 \
  --multiscale_patch_lens 8 16 32 \
  --multiscale_strides 4 8 16 \
  --channel_wise_gating 1 \
  --freq_feature_mode basic \
  --freq_num_bands 3
```
