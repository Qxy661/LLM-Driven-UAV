# 06-01 LLM 任务规划论文导读

> 预计阅读：30 分钟 | 前置知识：01-03 Agent架构设计、02-02 LLM任务分解

本文档收录并导读 LLM 用于机器人/无人机任务规划的核心论文，帮助读者快速了解该领域的研究脉络。

---

## 1. 论文导读方法

### 1.1 论文卡片格式

每篇论文按以下结构导读：

```
论文标题
├── 基本信息 (作者、年份、会议)
├── 核心问题 (解决什么问题)
├── 方法概述 (怎么解决的)
├── 关键创新 (与之前方法的区别)
├── 实验结果 (效果如何)
├── 与无人机的关联
└── 延伸阅读
```

---

## 2. 核心论文

### 2.1 SayCan — LLM 首次与机器人结合

```
Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
作者: Ahn et al. (Google Research)
年份: 2022
会议: arXiv (CoRL 2022)

核心问题:
  LLM 可以理解语言，但不知道机器人能做什么。
  例如 LLM 可能建议"拿起杯子"，但机器人没有抓手。

方法概述:
  Score = P(LLM|语言理解) × P(affordance|机器人能力)
  最终选择得分最高的动作。

  LLM 提供"应该做什么"（语言知识）
  affordance 提供"能做什么"（机器人能力）
  两者结合 = "应该且能做什么"

关键创新:
  首次将 LLM 的语言理解与机器人的物理能力结合
  不需要 LLM 微调，使用预训练模型

实验结果:
  在真实机器人上完成厨房任务，成功率 84%

与无人机的关联:
  LLM 可以理解"巡逻A区域"，但需要结合无人机能力
  affordance = 飞行能力、传感器能力、电量限制
```

### 2.2 Code as Policies — LLM 直接生成代码

```
Code as Policies: Language Model Programs for Embodied Control
作者: Liang et al. (Google Research)
年份: 2022
会议: ICRA 2023

核心问题:
  让 LLM 直接生成可执行的机器人控制代码

方法概述:
  用户: "把所有红色方块堆成一列"
  LLM 生成 Python 代码:
    red_blocks = detect(color="red")
    for i, block in enumerate(red_blocks):
        robot.pick(block)
        robot.place(target_position=(x, i*block_height))

关键创新:
  LLM 不只是输出动作序列，而是输出可执行代码
  代码可以包含循环、条件、变量等编程结构
  可以利用现有的 Python 生态系统

实验结果:
  在机械臂任务中，代码策略优于纯动作序列

与无人机的关联:
  LLM 可以生成完整的飞行控制脚本
  可以利用 Python 库处理复杂逻辑（如循环巡逻）
```

### 2.3 Inner Monologue — 闭环推理

```
Inner Monologue: Embodied Reasoning through Planning with Language Models
作者: Huang et al. (Google Research)
年份: 2022
会议: CoRL 2022

核心问题:
  LLM 规划需要环境反馈才能正确调整

方法概述:
  LLM → 动作 → 环境反馈 → LLM → 调整动作 → ...

  反馈来源:
  1. 视觉反馈 (成功了吗？)
  2. 语言反馈 (人类评价)
  3. 状态反馈 (机器人状态)

关键创新:
  将 LLM 推理变成闭环过程
  LLM 可以根据反馈调整计划

实验结果:
  在机器人任务中，闭环规划成功率显著高于开环

与无人机的关联:
  无人机任务执行中需要持续反馈
  "飞到建筑旁" → 摄像头反馈 → "偏了，向左调整"
```

### 2.4 PaLM-E — 具身多模态模型

```
PaLM-E: An Embodied Multimodal Language Model
作者: Driess et al. (Google DeepMind)
年份: 2023
会议: ICML 2023

核心问题:
  构建一个能同时理解语言、图像和机器人状态的大模型

方法概述:
  562B 参数的多模态模型
  输入: 文本 + 图像 + 传感器数据
  输出: 文本 (包括动作描述)

关键创新:
  首次在单一模型中融合语言、视觉和具身数据
  规模效应: 更大的模型在具身任务上也更好

实验结果:
  在多种机器人任务上达到 SOTA
  展示了正迁移: 机器人数据帮助语言理解

与无人机的关联:
  可以直接理解无人机摄像头图像 + 语言指令
  统一模型减少了系统复杂度
```

### 2.5 Voyager — LLM 驱动的自主探索

```
Voyager: An Open-Ended Embodied Agent with Large Language Models
作者: Wang et al. (NVIDIA)
年份: 2023
会议: arXiv

核心问题:
  让 LLM Agent 在开放世界中自主探索和学习

方法概述:
  三个核心组件:
  1. 课程生成器: LLM 决定下一步探索什么
  2. 代码生成器: LLM 生成可执行的技能代码
  3. 技能库: 存储和检索已学到的技能

关键创新:
  自主课程: Agent 自己决定学什么
  技能积累: 学到的技能可以复用
  持续改进: 从失败中学习

实验结果:
  在 Minecraft 中获得的物品是 ChatGPT 的 3.3 倍

与无人机的关联:
  无人机可以在飞行中自主学习新技能
  技能库可以跨任务复用
```

### 2.6 其他重要论文

| 论文 | 年份 | 核心贡献 |
|------|------|---------|
| ProgPrompt | 2022 | 程序化提示生成机器人计划 |
| TidyBot | 2023 | LLM 驱动的家居整理机器人 |
| Language to Rewards | 2023 | 语言到奖励函数的映射 |
| AdaPlanner | 2023 | 自适应 LLM 规划器 |
| GenSim | 2023 | LLM 生成仿真任务 |
| Eureka | 2023 | LLM 自动设计奖励函数 |
| RAP | 2023 | LLM 推理作为规划 |
| Tree of Thoughts | 2023 | 树搜索增强 LLM 推理 |

---

## 3. 无人机专用论文

### 3.1 LLM 用于无人机任务规划

| 论文 | 年份 | 核心贡献 |
|------|------|---------|
| ChatGPT for Robotics | 2023 | ChatGPT 用于机器人/无人机控制 |
| LLM-Drone | 2024 | LLM 驱动的无人机任务规划框架 |
| AirGPT | 2024 | 无人机专用 LLM Agent |
| Language-Guided UAV | 2024 | 语言引导的无人机导航 |

### 3.2 ChatGPT for Robotics

```
ChatGPT for Robotics: Design Principles and Model Abilities
作者: Vemprala et al. (Microsoft)
年份: 2023
会议: IEEE Access

核心问题:
  如何使用 ChatGPT 控制无人机

方法概述:
  设计 prompt 模板，让 ChatGPT 生成无人机控制代码
  使用函数库: takeoff(), land(), goto(), etc.

关键创新:
  自然语言到无人机控制的直接映射
  无需微调，使用 prompt engineering

实验结果:
  在简单任务上成功率 80%+
  在复杂任务上需要人类辅助

与本项目的关系:
  这是本项目最直接的参考论文
  本项目在此基础上增加了 VLM、安全层等
```

---

## 思考题

1. SayCan 的 affordance 机制如何应用到无人机场景？
2. Code as Policies 相比直接输出动作序列有什么优势？
3. Inner Monologue 的闭环推理对无人机任务规划有什么启示？
4. Voyager 的技能库思想如何应用到无人机系统？
5. 选择一篇论文，详细分析其方法并讨论在无人机场景中的适用性。

<details>
<summary>参考答案</summary>

**1.** 无人机的 affordance 包括：(1) 飞行能力 — 能飞多高、多快、多远；(2) 传感器能力 — 摄像头视野、LiDAR 范围；(3) 能量限制 — 电量决定可飞时间；(4) 环境限制 — 禁飞区、障碍物。LLM 规划的任务需要与这些 affordance 结合，过滤掉不可行的方案。

**2.** 优势：(1) 可以表达复杂逻辑（循环、条件）；(2) 可以调用 Python 生态系统（数值计算、图像处理）；(3) 可以定义变量和函数（代码复用）；(4) 可以处理异常（try-catch）。对于无人机，这意味着 LLM 可以生成完整的巡逻脚本，而不仅仅是航点列表。

**3.** 启示：(1) 无人机任务执行应该持续获取反馈（视觉、状态、人类输入）；(2) LLM 应该根据反馈调整计划；(3) 多源反馈可以互补（视觉看不清时用 LiDAR）；(4) 反馈延迟需要考虑（视觉反馈比状态反馈慢）。

**4.** 无人机技能库：(1) 每次成功完成任务后，将任务计划存入技能库；(2) 新任务来时，先检索相似技能；(3) 可以组合已有技能形成新技能；(4) 技能可以跨无人机共享。例如：学会"网格巡逻"技能后，新任务"巡逻停车场"可以直接复用。

**5.** 以 Voyager 为例：方法是 LLM 自主生成探索课程、编写技能代码、积累技能库。在无人机场景中：(1) 课程生成 — LLM 决定"学习在室内飞行"还是"学习避障"；(2) 代码生成 — 生成具体的飞行控制程序；(3) 技能库 — 存储已学会的飞行技能，新任务时检索复用。适用性好，但需要考虑安全约束。

</details>
