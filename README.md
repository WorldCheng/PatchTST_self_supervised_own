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

#### 修改后
当模型属于 TST / PatchTST 时：
-只搬运 batch_x、batch_y
-不再搬运 batch_x_mark、batch_y_mark
-不再构造 dec_inp
前向统一使用：
```python
outputs = self.model(batch_x)
