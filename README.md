# SimpleNet-CBAM-PatchGuard

本项目围绕工业图像无监督异常检测任务展开，基于轻量判别式框架 SimpleNet 进行改进，构建 SimpleNet-CBAM-PatchGuard 模型。方法通过多尺度特征提取与双通道伪异常生成增强异常边界学习，引入 CBAM 注意力机制提升异常相关特征表达，并借鉴 PatchGuard 鲁棒思想，在判别特征层进行稳健约束，减弱噪声与离群特征干扰。项目在 MVTec AD 数据集上完成实验验证，可实现图像级异常检测与像素级异常定位，并输出异常热力图。

## 项目目录结构

```text
├── README.md          # 项目说明文件
├── backbones.py       # 主干特征提取网络相关定义
├── common.py          # 公共模块与基础组件
├── histogram.py       # 异常分数分布可视化相关代码
├── main.py            # 项目主程序入口
├── metrics.py         # 评价指标计算
├── resnet.py          # ResNet 网络结构定义
├── run.sh             # 训练/测试运行脚本
├── simplenet.py       # SimpleNet 及改进模型核心实现
├── test_mvtec.py      # MVTec AD 数据集测试脚本
└── utils.py           # 工具函数
```
## 项目声明
- 项目名称： 基于 SimpleNet 的无监督图像异常检测算法设计与实现
- 项目作者： 曾宇雯
- 作者单位： 暨南大学网络空间安全学院
- 开发语言： Python
- 核心模型： SimpleNet-CBAM-PatchGuard
- 核心技术： 无监督异常检测、SimpleNet、CBAM 注意力机制、PatchGuard 鲁棒约束、多尺度特征融合、双通道伪异常生成
- 项目用途： 本科毕业设计成果展示、学术交流与学习参考
