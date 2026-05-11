# 06-02 VLM/VLA 论文导读

> 预计阅读：30 分钟 | 前置知识：03-02 视觉语言动作VLA

本文档收录视觉语言模型和视觉语言动作模型的核心论文。

---

## 1. VLM 核心论文

### 1.1 LLaVA — 视觉语言助手

```
Visual Instruction Tuning
作者: Liu et al. (University of Wisconsin-Madison)
年份: 2023
会议: NeurIPS 2023

核心问题:
  如何让 LLM 理解图像并遵循视觉指令

方法概述:
  两阶段训练:
  1. 预训练对齐: 冻结 CLIP + Llama，只训练线性投影层
  2. 指令微调: 冻结 CLIP，训练投影层 + LLM (LoRA)
  
  数据: 使用 GPT-4 生成视觉指令数据

关键创新:
  简单高效的 VLM 架构 (CLIP + Linear + Llama)
  使用 GPT-4 自动生成训练数据
  开源了模型和数据

实验结果:
  在多个 VQA 基准上接近 GPT-4V 水平
  开源 VLM 的标杆

与无人机的关联:
  可以分析无人机航拍图像
  支持中文，适合国内无人机场景
```

> 📄 arXiv: [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)

### 1.2 BLIP-2 — 高效 VLM

```
BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
作者: Li et al. (Salesforce Research)
年份: 2023
会议: ICML 2023

核心问题:
  如何高效地连接视觉和语言模型

方法概述:
  Q-Former: 可学习的查询 token
  从冻结的图像编码器中提取固定数量的视觉 token
  输入到冻结的 LLM 中

关键创新:
  Q-Former 架桥接视觉和语言
  极高的训练效率 (相比 Flamingo)
  多任务预训练

实验结果:
  在多个基准上达到 SOTA
  训练成本远低于同类方法

与无人机的关联:
  Q-Former 可以压缩视觉信息，适合边缘部署
```

> 📄 arXiv: [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)

### 1.3 GPT-4V — 商业 VLM 标杆

```
GPT-4V(ision) System Card
作者: OpenAI
年份: 2023

核心能力:
  - 图像理解和描述
  - 视觉推理和问答
  - 文档和图表理解
  - 多图像比较分析

与无人机的关联:
  可以分析复杂的航拍场景
  强大的推理能力适合任务规划
  但需要 API 调用，有延迟和成本
```

> 📄 Report: [GPT-4V(ision) System Card](https://cdn.openai.com/papers/GPTV_System_Card.pdf)

### 1.4 Gemini — 长上下文多模态

```
Gemini: A Family of Highly Capable Multimodal Models
作者: Google DeepMind
年份: 2023-2024

核心能力:
  - 原生多模态 (文本、图像、音频、视频)
  - 超长上下文 (1M-10M tokens)
  - 视频理解

与无人机的关联:
  可以处理长航时视频
  超长上下文可以包含完整飞行日志
  多模态原生支持简化系统架构
```

> 📄 arXiv: [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805)

---

## 2. VLA 核心论文

### 2.1 RT-1 — 大规模机器人数据

```
RT-1: Robotics Transformer for Real-World Control at Scale
作者: Brohan et al. (Google Research)
年份: 2022
会议: RSS 2023

核心问题:
  如何训练通用的机器人控制策略

方法概述:
  收集 130K 条机器人操作数据
  使用 Transformer 架构
  输入: 图像 + 语言指令
  输出: 离散化的机器人动作

关键创新:
  大规模机器人数据收集
  Transformer 用于机器人控制
  展示了数据规模的重要性

实验结果:
  在 700+ 任务上成功率 97%
  泛化到未见过的物体和指令

与无人机的关联:
  数据驱动的方法可以应用到无人机
  但需要大量无人机飞行数据
```

> 📄 arXiv: [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)

### 2.2 RT-2 — VLM 直接输出动作

```
RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
作者: Brohan et al. (Google DeepMind)
年份: 2023
会议: CoRL 2023

核心问题:
  能否让 VLM 直接输出机器人动作？

方法概述:
  在 VLM (PaLI-X/PaLM-E) 基础上微调
  将动作表示为文本 token
  例如: "12 5 8 0 0 0 1" → 动作序列

关键创新:
  首次将 VLM 扩展为 VLA
  利用互联网知识迁移到机器人
  单一模型处理视觉、语言和动作

实验结果:
  在未见过的物体和指令上泛化能力显著提升
  推理能力从互联网迁移到机器人

与无人机的关联:
  理论上可以直接用于无人机控制
  动作需要重新定义为飞行控制指令
```

### 2.3 OpenVLA — 开源 VLA

```
OpenVLA: An Open-Source Vision-Language-Action Model
作者: Kim et al. (Stanford)
年份: 2024
会议: arXiv

核心问题:
  提供开源的 VLA 基础模型

方法概述:
  基于 Llama-2 7B
  双视觉编码器: SigLIP + DINOv2
  动作输出: 离散化 token

关键创新:
  完全开源 (模型 + 数据 + 代码)
  可以用 LoRA 微调到新任务
  双编码器提升视觉理解

实验结果:
  在多个机器人基准上表现良好
  微调后可以适应新场景

与无人机的关联:
  可以直接用于无人机视觉导航
  开源便于二次开发
```

> 📄 arXiv: [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)

### 2.4 Octo — 通用机器人策略

```
Octo: An Open-Source Generalist Robot Policy
作者: Ghosh et al. (UC Berkeley)
年份: 2024
会议: arXiv

核心问题:
  通用的机器人控制策略

方法概述:
  基于 Transformer 架构
  在 Open X-Embodiment 数据集上训练
  支持多种机器人平台

关键创新:
  跨机器人迁移
  开源且可微调
  支持多种动作空间

与无人机的关联:
  理论上可以迁移到无人机平台
  需要定义无人机的动作空间
```

> 📄 arXiv: [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)

**CLIP 对比学习损失函数 (核心公式):**

```
L_CLIP = -(1/N) Σ_{i=1}^{N} [ log( exp(sim(I_i, T_i)/τ) / Σ_{j=1}^{N} exp(sim(I_i, T_j)/τ) ) ]

其中:
  I_i = 第 i 个图像的嵌入向量 (image embedding)
  T_i = 第 i 个文本的嵌入向量 (text embedding, 与 I_i 配对)
  τ   = 温度参数 (可学习), 控制 softmax 的锐度
  sim(I,T) = cosine(I, T) / (||I|| · ||T||) 余弦相似度
  N   = batch size

直觉: 拉近配对 (I_i, T_i) 的距离, 推远不配对 (I_i, T_j, j≠i) 的距离
→ 这使得视觉和语言共享同一个语义空间, 是所有 VLM/VLA 的基础
```

---

## 3. 遥感 VLM 论文

| 论文 | 年份 | 核心贡献 |
|------|------|---------|
| RSVQA | 2020 | 遥感视觉问答 |
| EarthGPT | 2024 | 遥感多模态大模型 |
| GeoChat | 2024 | 遥感对话模型 |
| LHRS-Bot | 2024 | 遥感视觉语言模型 |

---

## 思考题

1. LLaVA 的两阶段训练策略有什么优势？
2. RT-2 将动作表示为文本 token 有什么优缺点？
3. OpenVLA 的双编码器设计为什么有效？
4. 如何将 RT-2 的思想应用到无人机控制？
5. 遥感 VLM 与通用 VLM 有什么区别？

<details>
<summary>参考答案</summary>

**1.** 两阶段优势：(1) 第一阶段对齐视觉和语言空间，让 LLM 理解视觉输入；(2) 第二阶段微调指令遵循能力；(3) 分阶段训练更稳定，不容易遗忘；(4) 第一阶段数据需求少（图文对），第二阶段可以用合成数据。

**2.** 优点：(1) 统一了 VLM 和动作生成，单一模型更简洁；(2) 可以利用 VLM 预训练的视觉和语言理解能力；(3) 自然支持多模态输入。缺点：(1) 离散化损失精度；(2) 动作 token 序列长度有限制；(3) 推理速度受自回归生成限制。

**3.** 双编码器有效性：(1) SigLIP 擅长语义理解（理解图像内容）；(2) DINOv2 擅长空间特征（精确定位）；(3) 两者互补 — 机器人需要同时理解"是什么"和"在哪里"；(4) 实验证明双编码器优于单一编码器。

**4.** 应用到无人机：(1) 重新定义动作空间 — 从机械臂的 7DoF 改为无人机的 6DoF (x,y,z,roll,pitch,yaw)；(2) 收集无人机飞行数据；(3) 微调 RT-2/OpenVLA 到无人机动作空间；(4) 添加安全层验证动作。

**5.** 区别：(1) 视角 — 遥感是俯视，通用是平视；(2) 尺度 — 遥感目标小，通用目标大；(3) 术语 — 遥感有专业术语（NDVI、光谱）；(4) 任务 — 遥感关注分类/检测/变化检测，通用关注对话/描述。遥感 VLM 通常需要用遥感数据专门训练。

</details>
