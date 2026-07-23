---
title: "LingBot-VLA 2.0: 从基础模型到实际机器人应用"
date: 2026-07-23T01:30:00+08:00
draft: false
tags: ["VLA", "机器人", "具身智能", "深度学习", "MoE"]
categories: ["技术文档"]
weight: 10
---

## 概述

**LingBot-VLA 2.0** 是一个面向实际机器人应用的 Vision-Language-Action（视觉-语言-动作）基础模型，由 Robbyant 团队开发。论文标题为 *"From Foundation to Application: Improving VLA Models in Practice"*。

相较于 LingBot-VLA 1.0，2.0 版本的三大核心改进：

1. **跨任务、跨本体泛化**：重新设计数据流水线，整合约 **60,000 小时** 预训练数据
2. **扩展动作空间**：统一表示支持手臂、末端执行器、夹爪、灵巧手、腰部、头部和移动底盘信号
3. **预测性动力学建模**：将未来预测作为代理任务，融合语义时间先验和几何线索

---

## 核心能力

| 维度 | 描述 |
|------|------|
| 模型尺寸 | **6B** 参数量 |
| 视觉主干 | Qwen3-VL-4B-Instruct |
| 动作专家 | 36 层 Qwen2 架构 + 稀疏 MoE |
| 动作维度 | 55 维标准化状态/动作向量 |
| 预训练数据 | ~60,000 小时（50,000h 机器人 + 10,000h 人类第一人称） |
| 机器人构型 | 20 种 |
| 推理速度 | RTX 4090D 上 ~**130ms**/步（10 步去噪） |
| 基础模型 | LingBot-VLA-V2-6B（原生深度版本） |

---

## 数据流水线

### 数据规模

LingBot-VLA 2.0 使用海量异构预训练语料：

- **50,000 小时** 机器人轨迹，覆盖 20 种机器人构型
- **10,000 小时** 第一人称人类操作视频

数据类型涵盖单臂、双臂、半人形、全人形及第一人称视角。

### 数据筛选

**机器人数据过滤策略：**
- 移除视频-状态不同步的样本
- 剔除模糊/遮挡视频
- 多视角对齐校验
- 异常速度/加速度/加加速度检测
- 静态信号片段去除

**第一人称视频过滤策略：**
- 保留以操作为中心的视频
- 重建并标准化手部轨迹
- 过滤不稳定的相机或手部运动估计

![数据分布概览](/images/lingbot-vla-v2/lingbot_vla2_data_demo.png)

![数据筛选与处理流程](/images/lingbot-vla-v2/lingbot_vla2_data_process.png)

---

## 模型架构

### 整体框架

LingBot-VLA 2.0 的核心架构分为三个主要部分：

1. **视觉语言主干（VLM Backbone）**：基于 Qwen3-VL-4B-Instruct，负责多模态理解
2. **动作专家（Action Expert）**：36 层 Qwen2 解码器，包含稀疏 MoE（Mixture-of-Experts）层
3. **双查询蒸馏（Dual-Query Distillation）**：从 LingBot-Depth 和 DINO-Video 蒸馏先验知识

![LingBot-VLA 2.0 整体架构](/images/lingbot-vla-v2/lingbot_vla2_framework.png)

### 视觉语言主干（Qwen3-VL-4B）

- 基于 **Qwen3-VL-4B-Instruct** 作为多模态编码器
- 支持图像和视频输入
- 使用 Flash Attention 2 加速
- 训练时可冻结视觉编码器，仅训练动作专家

### 统一动作表示

LingBot-VLA 2.0 将不同机器人的状态/动作映射到一个 **55 维** 标准化向量空间：

| 维度 | 含义 | 维度数 |
|------|------|:------:|
| Arm Joint Position | 手臂关节位置 | 14 |
| End-effector Pose | 末端执行器姿态 | 14 |
| Gripper Position | 夹爪位置 | 2 |
| Hand Joint Position | 灵巧手关节位置 | 12 |
| Waist Position | 腰部位置 | 4 |
| Head Position | 头部位置 | 2 |
| Mobility Signal | 移动信号 | 3 |
| Reserved | 预留维度 | 4 |

![55 维统一动作空间](/images/lingbot-vla-v2/lingbot_vla2_data_dimension.png)

### MoE Action Expert

为提高跨本体扩展能力，LingBot-VLA 2.0 在动作专家中引入了**稀疏 MoE 层**：

- **细粒度专家分割**（Fine-grained Expert Segmentation）
- **共享专家隔离**（Shared Expert Isolation）
- 允许通用先验和专长模式在相同计算预算下共存
- 支持 Token-Level Routing，每个 token 独立选择专家

**MoE 关键配置：**
- 专家数量：32（可配置）
- Top-K：1（每 token 选一个专家）
- 路由激活函数：Softmax 或 Sigmoid
- 损失函数：Sequence-wise Auxiliary Loss + Router Z-Loss


### 双查询蒸馏（Dual-Query Distillation）

LingBot-VLA 2.0 在输入序列中追加了**当前感知查询**和**未来感知查询**：

- **LingBot-Depth**：提供当前场景的几何结构先验（深度信息）
- **DINO-Video**：提供未来场景演化的语义时间先验

这些查询被蒸馏到 VLM 的隐藏状态中，鼓励模型进行因果推理，同时捕捉当前场景几何和未来场景变化。

![双查询蒸馏机制](/images/lingbot-vla-v2/lingbot_vla2_vis_distillation.png)

### 注意力机制

- 默认使用 **Flex Attention**（支持自定义 Block Mask）
- 支持 Flash Attention 2、SDPA、Eager 等多种后端
- 视觉前缀 + 动作后缀的联合注意力计算
- 使用 3D 旋转位置编码（mRoPE），支持图像、视频和文本混合输入

### Flow Matching 动作生成

动作生成采用 **Flow Matching** 框架：

- 训练目标：预测速度场 `v_t`，而非直接预测动作
- 正向过程：`x_t = t * noise + (1 - t) * actions`
- 推理过程：从高斯噪声开始，逐步去噪生成动作
- 默认 10 步去噪，可在速度和精度之间权衡

```python
# 训练时：MSE loss on velocity
x_t = time * noise + (1 - time) * actions
u_t = noise - actions  # ground truth velocity
loss = MSE(v_t_pred, u_t)
```

---

## 训练流程

### 环境配置

```bash
# PyTorch 2.8.0 + Python 3.12
# Flash Attention 2.8.3
conda create -n lingbotvla python=3.12
bash tools/create_train_env.sh
```

### 后训练（Post-Training）

以 RoboTwin 2.0 的 50 个任务为例：

```bash
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin.yaml \
  --data.train_path assets/training_data/robotwin.txt \
  --data.data_name multi \
  --train.output_dir output/
```

**后训练关键配置：**
- Sequence-wise Auxiliary Loss：`per_sequence` 模式，系数 `1e-3`
- Router Z-Loss 系数：`1e-4`
- 支持 Muon 优化器（相比 AdamW 收敛更好，但更慢）
- 可切换为 Loss-Free Routing（`bias_update_speed` 控制）

### 分布式训练

项目内置了完整的分布式训练支持：

| 方案 | 说明 |
|------|------|
| FSDP2 | 全分片数据并行 |
| Sequence Parallel | Ulysses 风格序列并行 |
| Expert Parallel | MoE 专家参数分片 |
| VeScale | 华为昇腾 NPU 适配 |

---

## 部署与推理

### 实时机器人部署

```bash
export QWEN3VL_PATH=path_to_Qwen3-VL-4B-Instruct
python -m deploy.lingbot_vla_v2_policy \
  --model_path path_to_checkpoint \
  --use_compile \
  --use_length 25 \
  --port port
```

- **RTX 4090D** 上单次推理约 **130ms**（10 步去噪）
- 支持 WebSocket 策略服务器，可集成到机器人控制框架
- 使用 `torch.compile` 进一步加速推理

### 开环评估

```bash
python scripts/open_loop_eval.py \
  --model_path path_to_posttraining_ckpt \
  --robo_name robotwin \
  --data_path path_to_validation_data \
  --use_length 50
```

---

## 性能表现

### GM-100 双臂操控

| 平台 | GR00T N1.7 | π₀ | LingBot-VLA 1.0 | **LingBot-VLA 2.0** |
|------|:----------:|:----:|:---------------:|:-------------------:|
| AgileX Cobot Magic | 36.3 / 17.8 | 59.1 / 32.2 | 58.2 / 30.0 | **66.2 / 34.4** |
| Galaxea R1Pro | 16.4 / 5.6 | 27.4 / 8.9 | 32.7 / 15.6 | **34.6 / 15.6** |

*指标为 进度分/成功率*

![GM-100 消融实验](/images/lingbot-vla-v2/lingbot_vla2_gm100_ablation_barplot.png)

### 长时程移动操控

| 机器人 | 任务 | 设置 | **LingBot-VLA 2.0** | π₀ |
|--------|------|:----:|:------------------:|:--:|
| Astribot S1 | 冰箱整理 | 域内 | **77.1 / 60.0** | 65.3 / 46.7 |
| Astribot S1 | 冰箱整理 | 域外 | **37.0 / 13.3** | 30.3 / 6.7 |
| Cobot Magic-ARX X5 | 炉灶清理 | 域内 | **84.3 / 66.7** | 79.9 / 60.0 |
| Cobot Magic-ARX X5 | 炉灶清理 | 域外 | **67.5 / 40.0** | 62.5 / 33.3 |

---

## 项目结构

```
lingbot-vla-v2/
├── lingbotvla/                    # 核心源码
│   ├── models/                    # 模型定义
│   │   └── vla/lingbot_vla/       # V2 主模型
│   │       ├── modeling_lingbot_vla_v2.py  # 主模型与 Flow Matching
│   │       ├── qwen2_action_expert.py      # MoE Action Expert
│   │       ├── qwen3vl_in_vla.py           # Qwen3-VL 适配
│   │       ├── configuration_lingbot_vla.py # 配置
│   │       └── flex_attention.py           # Flex Attention
│   ├── data/                      # 数据加载与处理
│   ├── distributed/               # 分布式训练（FSDP、EP、SP）
│   │   ├── moe/                   # MoE 通信
│   │   ├── fsdp2/                 # FSDP2 实现
│   │   └── sequence_parallel/     # 序列并行
│   ├── ops/                       # 高性能算子
│   │   ├── fused_moe.py           # 融合 MoE 算子
│   │   ├── group_gemm/            # Group GEMM
│   │   └── robby_moe.py           # Robbyant 自定义 MoE
│   ├── optim/                     # 优化器（Muon）
│   └── schedulers/flow_match.py   # Flow Matching 调度器
├── deploy/                        # 部署脚本
├── configs/                       # 配置文件
│   ├── vla/                       # 训练配置
│   └── robot_configs/             # 机器人参数配置
├── scripts/                       # 工具脚本
└── assets/                        # 论文图片与数据
```

---

## 引用

```bibtex
@article{lingbotvla2,
      title={From Foundation to Application: Improving VLA Models in Practice}, 
      author={Wei Wu and Fangjing Wang and Fan Lu and He Sun and Shi Liu and Yunnan Wang 
              and Yibin Yan and Yong Wang and Shuailei Ma and Xinyang Wang and Yibin Liu 
              and Shuai Yang and Tianxiang Zhou and Kejia Zhang and Lei Zhou and Cheng Su 
              and Nan Xue and Bin Tan and Han Zhang and Youchao Zhang and Fei Liao 
              and Xing Zhu and Yujun Shen and Kecheng Zheng},
      journal={arXiv preprint arXiv:2607.06403},
      year={2026}
}
```

## 相关链接

- [论文 PDF](https://arxiv.org/pdf/2607.06403)
- [项目主页](https://technology.robbyant.com/lingbot-vla-v2)
- [Hugging Face 模型](https://huggingface.co/collections/robbyant/lingbot-vla-v2)
- [ModelScope 模型](https://modelscope.cn/collections/Robbyant/LingBot-VLA-V2)
- [GitHub 仓库](https://github.com/Robbyant/lingbot-vla-v2)
- [LICENSE: Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
