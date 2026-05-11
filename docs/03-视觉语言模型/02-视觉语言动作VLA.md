# 03-02 视觉语言动作模型 (VLA)

> 预计阅读：30 分钟 | 前置知识：03-01 VLM在无人机中的应用

本文档介绍视觉语言动作模型（Vision-Language-Action, VLA），这是将视觉感知和语言理解直接映射到机器人/无人机动作的端到端模型。

---

## 1. VLA 概述

### 1.1 从 VLM 到 VLA

```
传统 Pipeline (感知→决策→控制):
  图像 → VLM 理解 → LLM 规划 → 控制器执行
  多个模块，信息传递可能丢失

VLA (端到端):
  图像 + 语言指令 → VLA → 动作
  单一模型，直接从感知到行动
```

### 1.2 VLA 的核心思想

VLA 将视觉-语言模型的输出从"文本"扩展到"动作"：

| 模型类型 | 输入 | 输出 |
|---------|------|------|
| VLM | 图像 + 文本 | 文本（描述、回答） |
| VLA | 图像 + 文本 | 动作（位移、旋转、抓取） |

### 1.3 里程碑模型

| 模型 | 年份 | 机构 | 特点 |
|------|------|------|------|
| RT-1 | 2022 | Google | 大规模机器人数据训练 |
| RT-2 | 2023 | Google | VLM 直接输出动作 token |
| Octo | 2023 | UC Berkeley | 开源通用机器人策略 |
| OpenVLA | 2024 | Stanford | 开源 VLA，可微调 |
| π0 | 2024 | Physical Intelligence | 流匹配 VLA |

---

## 2. RT-2：视觉语言动作模型

### 2.1 架构

RT-2（Robotic Transformer 2, Brohan et al., 2023）的核心创新是将机器人动作表示为文本 token：

```
输入:
  图像: 机器人摄像头画面
  文本: "把红色方块放到蓝色碗里"

处理:
  图像 → ViT 编码 → 视觉 token
  文本 → Tokenizer → 文本 token
  [视觉 token] + [文本 token] → PaLI-X/PaLM-E → 动作 token

输出:
  动作 token: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
  例如: "12 5 8 0 0 0 1" → 向前移动12cm，向右5cm，向上8cm，抓取
```

### 2.2 动作离散化

```
连续动作空间 → 离散化为 256 个 bin

位移: -1.0 到 +1.0 → 256 个离散值
旋转: -π 到 +π → 256 个离散值
夹爪: 0 (开) 或 1 (关)

动作 token 序列: "a_128 a_135 a_140 a_128 a_128 a_128 a_255"
                  代表: [Δx=0, Δy=0.05, Δz=0.1, Δr=0, Δp=0, Δy=0, grip=close]
```

---

## 3. OpenVLA：开源视觉语言动作模型

### 3.1 架构

OpenVLA（Open Vision-Language-Action Model, Kim et al., 2024）是一个开源的 VLA：

```
┌─────────────────────────────────────────┐
│              OpenVLA 架构               │
│                                         │
│  ┌──────────┐    ┌──────────────────┐   │
│  │ SigLIP   │    │ DINOv2           │   │
│  │ (语义特征)│    │ (空间特征)        │   │
│  └────┬─────┘    └────────┬─────────┘   │
│       │                   │             │
│       └───────┬───────────┘             │
│               ▼                         │
│  ┌──────────────────────────────────┐   │
│  │  Projector (MLP)                 │   │
│  └──────────────┬───────────────────┘   │
│                 ▼                       │
│  ┌──────────────────────────────────┐   │
│  │  Llama-2 7B                      │   │
│  │  (生成动作 token 序列)            │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### 3.2 微调 OpenVLA 用于无人机

```python
# OpenVLA 微调示例
from openvla import OpenVLA
from transformers import AutoProcessor

# 加载预训练模型
model = OpenVLA.from_pretrained("openvla/openvla-7b")
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

# 准备无人机数据
# 数据格式: (image, instruction, action)
training_data = [
    {
        "image": "drone_view_001.jpg",
        "instruction": "向前飞行2米",
        "action": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [dx,dy,dz,dr,dp,dyaw,grip]
    },
    {
        "image": "drone_view_002.jpg",
        "instruction": "向右移动并降低高度",
        "action": [0.0, 0.15, -0.1, 0.0, 0.0, 0.0, 0.0]
    },
    # ... 更多数据
]

# 使用 LoRA 微调
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)

# 训练循环
for epoch in range(num_epochs):
    for batch in training_data:
        inputs = processor(batch["image"], batch["instruction"])
        outputs = model(**inputs, labels=batch["action"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

---

## 4. VLA 在无人机中的应用

### 4.1 应用场景

| 场景 | 输入 | 输出 VLA 动作 | 说明 |
|------|------|-------------|------|
| 视觉导航 | 摄像头图像 + "飞到红色建筑" | 航点/速度指令 | 语言引导的导航 |
| 目标跟踪 | 目标图像 + "跟踪那个人" | 跟随速度指令 | 视觉跟踪 |
| 精准降落 | 降落点图像 + "降落在标记处" | 降落调整指令 | 精确着陆 |
| 避障飞行 | 前方图像 + "穿过树林" | 避障+前进指令 | 复杂环境导航 |
| 编队飞行 | 队友图像 + "保持V字队形" | 编队调整指令 | 多机协同 |

### 4.2 无人机 VLA 系统设计

```python
class UAVVLAController:
    """基于 VLA 的无人机控制器"""

    def __init__(self, vla_model, flight_controller):
        self.vla = vla_model
        self.fc = flight_controller
        self.control_frequency = 10  # Hz

    def run(self, instruction):
        """执行语言指令控制的飞行"""
        while True:
            # 获取当前摄像头图像
            image = self.fc.get_camera_image()

            # VLA 推理
            action = self.vla.predict(image, instruction)

            # 将 VLA 动作转换为飞控指令
            flight_command = self._action_to_command(action)

            # 执行飞控指令
            self.fc.execute(flight_command)

            # 检查任务完成
            if self._is_task_complete(action, instruction):
                break

            # 控制频率
            time.sleep(1.0 / self.control_frequency)

    def _action_to_command(self, action):
        """将 VLA 输出转换为飞控命令"""
        # action = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
        return {
            "velocity_x": action[0] * 2.0,    # 缩放到合理范围
            "velocity_y": action[1] * 2.0,
            "velocity_z": action[2] * 1.0,
            "yaw_rate": action[5] * 1.0,
        }
```

---

## 5. 数据收集与训练

### 5.1 无人机 VLA 数据收集

```python
class UAVDataCollector:
    """无人机 VLA 数据收集器"""

    def __init__(self, flight_controller):
        self.fc = flight_controller
        self.dataset = []

    def collect_episode(self, instruction):
        """收集一个完整的操作序列"""
        episode = []

        while not self._is_done():
            # 记录当前状态
            image = self.fc.get_camera_image()
            state = self.fc.get_state()

            # 记录人类操作 (遥操作)
            human_action = self._get_human_input()

            # 存储数据点
            episode.append({
                "image": image,
                "instruction": instruction,
                "state": state,
                "action": human_action,
                "timestamp": time.time()
            })

            # 执行人类操作
            self.fc.execute(human_action)

        self.dataset.append(episode)
        return episode
```

### 5.2 数据格式

```python
# OpenVLA 兼容的数据格式
{
    "episode_id": "ep_001",
    "instruction": "飞到红色建筑旁边然后悬停",
    "steps": [
        {
            "image": "frame_0001.jpg",  # 或 base64 编码
            "state": {
                "position": [31.23, 121.47, 15.0],
                "velocity": [0.0, 0.0, 0.0],
                "battery": 95.0
            },
            "action": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        # ... 更多帧
    ]
}
```

---

## 6. VLA vs 传统控制

### 6.1 对比

| 维度 | VLA | 传统控制 |
|------|-----|---------|
| 泛化能力 | 强 (可处理未见过的指令) | 弱 (需预编程) |
| 精确性 | 中等 | 高 (PID等精确控制) |
| 实时性 | 较慢 (100ms+) | 快 (1ms) |
| 安全性 | 需要额外安全层 | 可形式化验证 |
| 可解释性 | 低 (黑盒) | 高 (白盒) |

### 6.2 混合架构

```
最佳实践: VLA 做高层决策 + 传统控制做底层执行

┌─────────────┐
│ VLA         │  → 生成目标位置/速度 (1-10Hz)
│ "飞到红建筑" │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 传统控制器  │  → PID/模型预测控制 (100Hz)
│ 位置/速度环 │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 飞控硬件    │  → 电机控制 (1000Hz)
│ PX4/ArduPilot│
└─────────────┘
```

---

## 思考题

1. VLA 的动作离散化（256 bins）对无人机控制精度有什么影响？如何改进？
2. 如何为无人机场景收集高质量的 VLA 训练数据？
3. VLA 和传统 PID 控制应该如何协作？各负责什么层级？
4. OpenVLA 的双编码器（SigLIP + DINOv2）设计有什么优势？
5. 如何评估 VLA 在无人机场景中的性能？设计评估方案。

<details>
<summary>参考答案</summary>

**1.** 256 bins 对于位移控制精度约为 1/128 的范围，对于无人机来说可能不够精确（特别是低速精细操作）。改进方案：(1) 增加 bin 数量到 1024；(2) 对不同维度使用不同精度（位移用更多 bins，旋转用较少）；(3) 使用连续动作输出（回归而非分类）；(4) 混合方案 — VLA 输出粗略目标，PID 做精细调整。

**2.** 数据收集方案：(1) 遥操作 — 人类通过手柄/VR 控制无人机，记录图像和动作；(2) 模拟器 — 在 Gazebo/AirSim 中收集，可自动生成大量数据；(3) 人类示范 — 人类操作员执行任务，多角度录制；(4) 数据增强 — 对已有数据做旋转、裁剪、颜色变换。关键：指令多样性（同一动作多种描述方式）。

**3.** 协作方式：VLA 负责高层语义决策（"飞到建筑旁"→目标坐标），频率 1-10Hz。传统 PID 负责底层精确控制（保持位置/速度），频率 100Hz+。VLA 处理不确定性（语义理解），PID 保证精确性。安全层位于两者之间，验证 VLA 输出的安全性。

**4.** SigLIP 擅长语义理解（理解图像内容和语言指令的关联），DINOv2 擅长空间特征（精确定位物体位置）。两者互补：SigLIP 帮助理解"红色建筑"是什么，DINOv2 帮助精确定位红色建筑在哪里。这种组合在需要同时理解语义和精确空间信息的任务中特别有效。

**5.** 评估方案：(1) 任务成功率 — 给定指令，是否成功完成；(2) 定位精度 — 到达目标位置的误差；(3) 安全性 — 碰撞/越界次数；(4) 泛化性 — 对未见过的指令/场景的表现；(5) 实时性 — 控制延迟。在仿真环境中可以安全地做大规模评估，再在实机上做小规模验证。

</details>
