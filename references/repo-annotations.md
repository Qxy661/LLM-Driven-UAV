# GitHub 仓库注释

本文档收录与 LLM 驱动无人机系统相关的 GitHub 仓库，按类别组织。

---

## 1. LLM 框架与工具

### LangChain

```
仓库: https://github.com/langchain-ai/langchain
Stars: 90K+
语言: Python

简介: LLM 应用开发框架，支持 Agent、Chain、Memory 等模式

与无人机的关联:
├── Agent 模块: 构建无人机任务规划 Agent
├── Tools 模块: 封装飞控 API 为工具
├── Memory 模块: 实现飞行历史记忆
└── Chain 模块: 构建多步推理链

关键代码位置:
├── langchain/agents/ — Agent 实现
├── langchain/tools/ — 工具定义
├── langchain/memory/ — 记忆系统
└── langchain/chains/ — 推理链
```

### LlamaIndex

```
仓库: https://github.com/run-llama/llama_index
Stars: 30K+
语言: Python

简介: LLM 数据索引和检索框架

与无人机的关联:
├── 知识库: 索引无人机操作手册、飞行规则
├── RAG: 基于文档的问答
└── 查询引擎: 自然语言查询飞行数据

关键代码位置:
├── llama_index/indices/ — 索引实现
├── llama_index/retrievers/ — 检索器
└── llama_index/query_engine/ — 查询引擎
```

### Ollama

```
仓库: https://github.com/ollama/ollama
Stars: 80K+
语言: Go

简介: 在本地运行 LLM 的简单工具

与无人机的关联:
├── 快速部署本地 LLM
├── 支持多种模型 (Llama, Qwen, etc.)
└── 提供 API 接口

使用方式:
  ollama run llama3
  curl http://localhost:11434/api/generate -d '{"model":"llama3","prompt":"..."}'
```

---

## 2. 视觉语言模型

### LLaVA

```
仓库: https://github.com/haotian-liu/LLaVA
Stars: 18K+
语言: Python

简介: 视觉语言助手，开源 VLM 标杆

与无人机的关联:
├── 航拍图像分析
├── 视觉问答
└── 场景描述

关键代码位置:
├── llava/model/ — 模型定义
├── llava/train/ — 训练代码
├── llava/eval/ — 评估代码
└── llava/serve/ — 服务部署

部署示例:
  python -m llava.serve.model_worker --model-path liuhaotian/llava-v1.6-vicuna-7b
```

### OpenVLA

```
仓库: https://github.com/openvla/openvla
Stars: 2K+
语言: Python

简介: 开源视觉语言动作模型

与无人机的关联:
├── 视觉到动作的端到端映射
├── 可微调到无人机动作空间
└── 支持 LoRA 微调

关键代码位置:
├── prismatic/models/ — 模型定义
├── prismatic/training/ — 训练代码
└── scripts/ — 使用脚本
```

### InternVL

```
仓库: https://github.com/OpenGVLab/InternVL
Stars: 5K+
语言: Python

简介: 多尺寸开源 VLM

与无人机的关联:
├── 多种尺寸 (1B-76B) 适合不同部署场景
├── 中文支持好
└── 支持多种视觉任务

关键代码位置:
├── internvl_chat/ — 对话模型
└── classification/ — 分类任务
```

---

## 3. 语音处理

### Whisper

```
仓库: https://github.com/openai/whisper
Stars: 65K+
语言: Python

简介: 通用语音识别模型

与无人机的关联:
├── 语音指令识别
├── 多语言支持
└── 噪声鲁棒性

使用方式:
  import whisper
  model = whisper.load_model("medium")
  result = model.transcribe("audio.mp3", language="zh")
```

### Coqui TTS

```
仓库: https://github.com/coqui-ai/TTS
Stars: 30K+
语言: Python

简介: 开源文本转语音库

与无人机的关联:
├── 语音反馈生成
├── 多语言支持
└── 支持多种 TTS 模型

使用方式:
  from TTS.api import tts
  tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST")
  tts.tts_to_file("收到指令，正在执行", file_path="output.wav")
```

---

## 4. 飞控与机器人

### MAVSDK

```
仓库: https://github.com/mavlink/MAVSDK
Stars: 800+
语言: C++/Python

简介: MAVLink SDK，用于与 PX4/ArduPilot 通信

与无人机的关联:
├── Python API 控制无人机
├── 支持起飞/降落/导航等操作
└── 与 ROS2 集成

使用方式 (Python):
  from mavsdk import System
  drone = System()
  await drone.connect()
  await drone.action.takeoff()
  await drone.action.goto_location(lat, lon, alt, yaw)
```

### AirSim

```
仓库: https://github.com/microsoft/AirSim
Stars: 16K+
语言: C++/Python

简介: 无人机/汽车仿真器

与无人机的关联:
├── 无人机飞行仿真
├── 传感器模拟 (相机、LiDAR)
├── Python API 控制
└── 与 ROS2 集成

使用方式:
  import airsim
  client = airsim.MultirotorClient()
  client.takeoffAsync().join()
  client.moveToPositionAsync(x, y, z, velocity).join()
```

### Gazebo

```
仓库: https://github.com/gazebosim/gz-sim
Stars: 1K+
语言: C++

简介: 机器人仿真平台

与无人机的关联:
├── 物理仿真
├── 传感器仿真
├── 与 ROS2 深度集成
└── PX4 SITL 支持
```

---

## 5. 推理引擎

### vLLM

```
仓库: https://github.com/vllm-project/vllm
Stars: 25K+
语言: Python/C++

简介: 高性能 LLM 推理引擎

与无人机的关联:
├── 云端 LLM 服务
├── PagedAttention 高效推理
└── OpenAI 兼容 API

使用方式:
  python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8B-Instruct
```

### llama.cpp

```
仓库: https://github.com/ggerganov/llama.cpp
Stars: 60K+
语言: C/C++

简介: 轻量级 LLM 推理引擎

与无人机的关联:
├── 边缘设备部署 (Jetson)
├── CPU/GPU 混合推理
├── GGUF 量化格式
└── 低资源占用

使用方式:
  ./llama-server -m model.gguf -c 4096 --host 0.0.0.0 --port 8080
```

### TensorRT-LLM

```
仓库: https://github.com/NVIDIA/TensorRT-LLM
Stars: 8K+
语言: Python/C++

简介: NVIDIA 官方 LLM 推理优化

与无人机的关联:
├── Jetson 平台优化
├── 最大化 GPU 利用率
└── 低延迟推理
```

---

## 6. 工具与辅助

### MediaPipe

```
仓库: https://github.com/google-ai-edge/mediapipe
Stars: 25K+
语言: C++/Python

简介: 谷歌的多媒体处理框架

与无人机的关联:
├── 手部追踪 (手势控制)
├── 姿态估计 (体态控制)
├── 人脸检测
└── 实时运行

使用方式:
  import mediapipe as mp
  hands = mp.solutions.hands.Hands()
  results = hands.process(image)
```

### Hugging Face Transformers

```
仓库: https://github.com/huggingface/transformers
Stars: 125K+
语言: Python

简介: 预训练模型库

与无人机的关联:
├── 加载各种 LLM/VLM
├── 模型量化 (GPTQ, AWQ, bitsandbytes)
├── 模型训练和微调
└── 推理服务

使用方式:
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
```

---

## 7. 学习资源

| 资源 | 链接 | 说明 |
|------|------|------|
| LangChain 文档 | python.langchain.com | Agent 开发教程 |
| PX4 文档 | docs.px4.io | 飞控系统文档 |
| ROS2 教程 | docs.ros.org | ROS2 入门教程 |
| Hugging Face | huggingface.co | 模型和数据集 |
| Papers With Code | paperswithcode.com | 论文和代码 |
