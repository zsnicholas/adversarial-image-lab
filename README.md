# 对抗样本图像实验室

本项目用于训练一个简单的 MNIST 卷积神经网络，并使用 FGSM 和 PGD 两种方法生成对抗样本，观察模型在干净样本和对抗样本上的准确率变化。

## 项目结构

```text
adversarial_image_lab/
|-- models/
|   `-- simple_cnn.py
|-- attacks/
|   |-- fgsm.py
|   `-- pgd.py
|-- eval/
|   `-- metrics.py
|-- utils/
|   `-- visualize.py
|-- train.py
|-- evaluate_attack.py
`-- README.md
```

## SimpleCNN 模型

模型文件位于：

```text
models/simple_cnn.py
```

这个模型用于 MNIST 手写数字分类。输入是单通道灰度图，大小为 `28 x 28`。

### Shape 变化

MNIST 输入：

```text
x: [64, 1, 28, 28]
```

第一层卷积后：

```text
conv1: [64, 16, 28, 28]
```

原因是：

```text
in_channels = 1
out_channels = 16
kernel_size = 3
padding = 1
```

所以高和宽保持为 `28 x 28`。

第一次池化后：

```text
pool: [64, 16, 14, 14]
```

第二层卷积后：

```text
conv2: [64, 32, 14, 14]
```

第二次池化后：

```text
pool: [64, 32, 7, 7]
```

展平后：

```text
x.view(x.size(0), -1): [64, 1568]
```

因为：

```text
32 x 7 x 7 = 1568
```

全连接层输出：

```text
fc1: [64, 128]
fc2: [64, 10]
```

最终输出：

```text
logits: [64, 10]
```

这里的 `10` 表示 MNIST 的 10 个类别：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

## FGSM 攻击

FGSM 代码位于：

```text
attacks/fgsm.py
```

FGSM 的核心思想是：沿着让分类 loss 变大的方向，直接修改输入图片。

核心公式：

```text
adv_images = images + epsilon * images.grad.sign()
```

关键步骤：

- `images.clone().detach()`：复制原图，并切断旧计算图。
- `images.requires_grad = True`：让 PyTorch 计算 loss 对输入图片的梯度。
- `loss.backward()`：反向传播，得到 `images.grad`。
- `images.grad.sign()`：只取梯度方向，不关心梯度大小。
- `torch.clamp(adv_images, 0, 1)`：保证 MNIST 像素值仍在 `[0, 1]` 范围内。

`epsilon` 越大，扰动越强，攻击通常越明显，模型准确率通常越低。

## PGD 攻击

PGD 代码位于：

```text
attacks/pgd.py
```

PGD 可以理解为多步版本的 FGSM。它不是只走一步，而是在允许的扰动范围内反复迭代。

重要参数：

- `epsilon`：总扰动范围，限制对抗样本不能离原图太远。
- `alpha`：每一步走多远。
- `steps`：一共迭代多少步。

PGD 的投影步骤：

```text
perturbation = torch.clamp(
    adv_images - original_images,
    min=-epsilon,
    max=epsilon
)
adv_images = original_images + perturbation
```

这个步骤保证扰动不会超过 `epsilon`。

一般来说，在相同 `epsilon` 下，PGD 比 FGSM 更强。

## 评估指标

评估代码位于：

```text
eval/metrics.py
```

`evaluate_clean` 计算模型在干净测试集上的准确率。

`evaluate_attack` 计算三个指标：

- `clean_acc`：干净测试集准确率。
- `robust_acc`：对抗样本上的准确率。
- `attack_success_rate`：攻击成功率，当前代码中计算方式为 `1 - robust_acc`。

示例实验结果格式：

| Model | Attack | Epsilon | Alpha | Steps | Clean Acc | Robust Acc | Attack Success Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| SimpleCNN | None | 0 | - | - | 98.5% | - | - |
| SimpleCNN | FGSM | 0.1 | - | 1 | 98.5% | 55.0% | 45.0% |
| SimpleCNN | FGSM | 0.3 | - | 1 | 98.5% | 18.0% | 82.0% |
| SimpleCNN | PGD | 0.3 | 0.01 | 40 | 98.5% | 3.0% | 97.0% |

解释方式：

- `Clean Acc` 高，说明模型在正常图片上表现好。
- FGSM 后 `Robust Acc` 下降，说明模型对一步攻击不鲁棒。
- `epsilon` 变大后 `Robust Acc` 更低，说明扰动越大，攻击越强。
- PGD 通常比 FGSM 的 `Robust Acc` 更低，说明 PGD 是更强的攻击。

## 可视化

可视化代码位于：

```text
utils/visualize.py
```

可视化会显示三张图：

- `Original Image`：原始图片。
- `Perturbation`：扰动，为了方便观察会放大显示。
- `Adversarial Image`：攻击后的图片。

观察重点：

- 人眼是否还能看出原来的数字。
- 扰动是否明显。
- 模型是否已经预测错误。

对抗攻击的关键现象是：人眼看起来差不多，但模型预测变错。

## 运行方法

### 1. 安装依赖

```bash
pip install torch torchvision matplotlib
```

### 2. 训练模型

```bash
python train.py --epochs 3
```

训练完成后会生成：

```text
simple_cnn_mnist.pth
```

这是后续 FGSM 和 PGD 评估要加载的模型权重。

### 3. 检查干净测试集准确率

```bash
python evaluate_attack.py --attack none --model-path simple_cnn_mnist.pth
```

你应该能看到类似输出：

```text
Model: SimpleCNN
Attack: None
Clean Acc: 98.xx%
```

### 4. 运行 FGSM 攻击检查

```bash
python evaluate_attack.py --attack fgsm --epsilon 0.3 --model-path simple_cnn_mnist.pth
```

如果想看对抗样本图片：

```bash
python evaluate_attack.py --attack fgsm --epsilon 0.3 --model-path simple_cnn_mnist.pth --visualize
```

### 5. 运行 PGD 攻击检查

```bash
python evaluate_attack.py --attack pgd --epsilon 0.3 --alpha 0.01 --steps 40 --model-path simple_cnn_mnist.pth
```

如果想看 PGD 生成的对抗样本：

```bash
python evaluate_attack.py --attack pgd --epsilon 0.3 --alpha 0.01 --steps 40 --model-path simple_cnn_mnist.pth --visualize
```

## 建议实验顺序

先训练模型：

```bash
python train.py --epochs 3
```

然后测试干净准确率：

```bash
python evaluate_attack.py --attack none
```

再测试 FGSM：

```bash
python evaluate_attack.py --attack fgsm --epsilon 0.1
python evaluate_attack.py --attack fgsm --epsilon 0.3
```

最后测试 PGD：

```bash
python evaluate_attack.py --attack pgd --epsilon 0.3 --alpha 0.01 --steps 40
```

正常情况下，你会看到：

```text
Clean Acc 较高
FGSM Robust Acc 明显下降
PGD Robust Acc 通常比 FGSM 更低
Attack Success Rate 上升
```
