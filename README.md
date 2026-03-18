# 第一轮修改说明

## 修改目标
第一轮修改的目标是：**在不改动 PatchTST 模型结构的前提下，减少训练脚本中的无效计算，并补齐 AMP 混合精度支持**。

---

## 修改内容

### 1. 为 PatchTST / TST 单独设置训练路径

在 `exp_main.py` 的 `train()`、`vali()`、`test()` 中，新增了针对 `TST` 模型的单独分支。

#### 修改前
通用训练流程会统一执行：
- 搬运 `batch_x_mark`
- 搬运 `batch_y_mark`
- 构造 `dec_inp`

但对 PatchTST 来说，前向实际上只需要：

```python
outputs = self.model(batch_x)
```

#### 修改后
当模型属于 TST / PatchTST 时：
-只搬运 batch_x、batch_y
-不再搬运 batch_x_mark、batch_y_mark
-不再构造 dec_inp
前向统一使用：
```python
outputs = self.model(batch_x)
```
### 2. 训练阶段取消每个 epoch 的 test
#### 修改前
每个 epoch 结束后都会执行：vali、test
即每轮训练流程为：train + vali + test

#### 修改后
每个 epoch 结束后只执行：vali
即改为：train + vali
最终 test 只在训练完成、加载最佳模型后执行一次。
#### 作用
减少了每个 epoch 的额外评估开销，缩短了整体训练时间。

### 3. 验证阶段改为在 GPU 上直接计算 loss
#### 修改前
vali() 中会先将 outputs 和 batch_y 搬到 CPU，再计算 loss。

#### 修改后
改为：
在 GPU 上直接计算 loss = criterion(outputs, batch_y)
仅保存 loss.item()

#### 作用
减少了不必要的 GPU 到 CPU 数据拷贝，提高了验证阶段效率。

### 4. 补齐 AMP 混合精度支持
在第一轮修改中，统一补齐了 train()、vali()、test() 的 AMP 路径。
#### 训练阶段
当开启 --use_amp 时：
前向放在 autocast() 中执行
反向传播使用 GradScaler

#### 验证/测试阶段
当开启 --use_amp 时：
前向同样放在 autocast() 中执行
不使用 GradScaler

#### 作用
在 CUDA 环境下可实现：
更快的训练速度
更低的显存占用
