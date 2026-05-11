# 01-05 LLM 部署与优化

> 预计阅读：35 分钟 | 前置知识：01-01 大语言模型基础、Python 进阶

本文档介绍如何在资源受限的无人机边缘设备上部署和优化大语言模型，涵盖量化技术、推理引擎、以及 NVIDIA Jetson 平台的部署方案。

---

## 1. 部署挑战

### 1.1 为什么需要边缘部署？

| 部署方式 | 优势 | 劣势 |
|---------|------|------|
| 云端部署 | 算力充足，模型不受限 | 依赖网络，延迟高(100ms+)，隐私风险 |
| 边缘部署 | 低延迟(<50ms)，离线可用，隐私安全 | 算力受限，模型需压缩 |
| 混合部署 | 平衡性能和资源 | 架构复杂，需设计降级策略 |

### 1.2 无人机的资源约束

```
典型无人机机载计算平台:
┌─────────────────────────────────────┐
│ NVIDIA Jetson Orin Nano             │
│   CPU: 6核 Arm Cortex-A78AE        │
│   GPU: 1024 CUDA cores              │
│   内存: 8GB 统一内存                 │
│   功耗: 7-25W                       │
│   存储: 64GB eMMC + NVMe SSD        │
├─────────────────────────────────────┤
│ NVIDIA Jetson AGX Orin              │
│   CPU: 12核 Arm Cortex-A78AE       │
│   GPU: 2048 CUDA cores + 64 Tensor  │
│   内存: 64GB 统一内存                │
│   功耗: 15-60W                      │
│   存储: 64GB eMMC + NVMe SSD        │
└─────────────────────────────────────┘

对比:
  RTX 4090:  24GB VRAM, 450W, 826 TFLOPS (FP16)
  Orin Nano: 8GB 共享, 25W,  40  TFLOPS (FP16)
  Orin 64GB: 64GB 共享, 60W, 275 TFLOPS (FP16)
```

---

## 2. 模型量化

### 2.1 量化基础

量化是将模型权重从高精度（FP32/FP16）转换为低精度（INT8/INT4）的技术：

```
精度对比:
FP32: 32 bit, 范围 ±3.4×10³⁸, 精度最高
FP16: 16 bit, 范围 ±6.5×10⁴,  精度良好
INT8: 8 bit,  范围 -128~127,   精度可接受
INT4: 4 bit,  范围 -8~7,       精度有损

存储节省 (以 7B 模型为例):
FP32: 7B × 4 bytes = 28 GB
FP16: 7B × 2 bytes = 14 GB
INT8: 7B × 1 byte  = 7  GB
INT4: 7B × 0.5 byte = 3.5 GB
```

### 2.2 量化方法对比

| 方法 | 精度 | 速度 | 易用性 | 适用场景 |
|------|------|------|--------|---------|
| GPTQ | INT4/INT3 | 快 | 中等 | GPU 部署 |
| AWQ | INT4 | 快 | 简单 | GPU 部署 |
| GGUF | Q2-Q8 | 中等 | 简单 | CPU/混合推理 |
| bitsandbytes | INT8/INT4 | 中等 | 非常简单 | 快速实验 |
| SmoothQuant | INT8 | 快 | 中等 | 服务端部署 |
| FP8 | FP8 | 快 | 简单 | H100 GPU |

### 2.3 GGUF 量化

GGUF 是 llama.cpp 使用的量化格式，支持 CPU 和 GPU 混合推理：

```bash
# 下载模型
git clone https://huggingface.co/meta-llama/Llama-3-8B-Instruct

# 转换为 GGUF
python convert_hf_to_gguf.py Llama-3-8B-Instruct --outfile llama3-8b-f16.gguf

# 量化为不同精度
./llama-quantize llama3-8b-f16.gguf llama3-8b-q4_k_m.gguf Q4_K_M
./llama-quantize llama3-8b-f16.gguf llama3-8b-q8_0.gguf Q8_0
```

**GGUF 量化级别**:

| 量化类型 | 大小(7B) | 质量 | 速度 | 推荐场景 |
|---------|---------|------|------|---------|
| Q2_K | ~2.8 GB | 差 | 最快 | 仅测试 |
| Q4_K_M | ~4.1 GB | 良好 | 快 | 边缘部署首选 |
| Q5_K_M | ~4.8 GB | 很好 | 中等 | 质量优先 |
| Q6_K | ~5.5 GB | 优秀 | 中等 | 接近 FP16 |
| Q8_0 | ~7.0 GB | 极好 | 较慢 | 质量最高 |

### 2.4 AWQ 量化

AWQ（Activation-aware Weight Quantization, Lin et al., 2023）是一种基于激活值感知的权重量量方法：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 加载模型
model_path = "meta-llama/Llama-3-8B-Instruct"
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"  # GEMM kernel 更快
}

# 执行量化（需要校准数据）
model.quantize(tokenizer, quant_config=quant_config)

# 保存
model.save_quantized("llama3-8b-awq")
tokenizer.save_pretrained("llama3-8b-awq")
```

---

## 3. 推理引擎

### 3.1 推理引擎对比

| 引擎 | 特点 | 适用场景 | 吞吐量 |
|------|------|---------|--------|
| vLLM | PagedAttention, 高吞吐 | 服务端多用户 | 极高 |
| TensorRT-LLM | NVIDIA 官方优化 | NVIDIA GPU | 极高 |
| llama.cpp | CPU/GPU 混合, 轻量 | 边缘设备 | 中等 |
| Ollama | 简单易用的封装 | 快速原型 | 中等 |
| SGLang | 结构化生成优化 | Agent 场景 | 高 |

### 3.2 vLLM 部署

```bash
# 安装 vLLM
pip install vllm

# 启动 OpenAI 兼容的 API 服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B-Instruct \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --port 8000
```

```python
# 客户端调用（与 OpenAI API 完全兼容）
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "你是无人机任务规划助手"},
        {"role": "user", "content": "请规划一个检查校园操场的飞行任务"}
    ],
    temperature=0.3,
    max_tokens=500
)
print(response.choices[0].message.content)
```

### 3.3 TensorRT-LLM 部署

```bash
# 1. 转换模型
python examples/llama/convert_checkpoint.py \
    --model_dir ./Llama-3-8B-Instruct \
    --output_dir ./trt_ckpt \
    --dtype float16

# 2. 构建 TensorRT 引擎
trtllm-build \
    --checkpoint_dir ./trt_ckpt \
    --output_dir ./trt_engine \
    --gemm_plugin float16 \
    --max_batch_size 4 \
    --max_input_len 2048 \
    --max_output_len 512

# 3. 运行推理
python examples/run.py \
    --engine_dir ./trt_engine \
    --max_output_len 100 \
    --tokenizer_dir ./Llama-3-8B-Instruct
```

### 3.4 llama.cpp 在 Jetson 上部署

```bash
# 在 Jetson Orin 上编译 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 使用 CUDA 加速编译
make LLAMA_CUBLAS=1 LLAMA_CUDA_F16=1

# 运行量化模型
./llama-server \
    -m ./models/llama3-8b-q4_k_m.gguf \
    -c 4096 \
    -ngl 99 \
    --host 0.0.0.0 \
    --port 8080
```

---

## 4. Jetson 平台部署实战

### 4.1 环境准备

```bash
# Jetson Orin 环境配置
# 1. 安装 JetPack 6.0 (包含 CUDA 12.2, cuDNN 8.9, TensorRT 8.6)

# 2. 安装 Python 环境
sudo apt install python3-pip python3-venv
python3 -m venv ~/llm-env
source ~/llm-env/bin/activate

# 3. 安装 PyTorch for Jetson
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu122

# 4. 安装依赖
pip install transformers accelerate sentencepiece protobuf
```

### 4.2 Jetson 上运行 7B 模型

```python
# 在 Jetson Orin 64GB 上运行 Llama-3-8B (INT4 量化)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# INT4 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")

# 推理测试
prompt = "请用一句话描述无人机自主飞行的核心挑战。"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.3)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 预期性能:
# Jetson Orin 64GB: ~15-20 tokens/s (Llama-3-8B INT4)
# Jetson Orin Nano 8GB: ~5-8 tokens/s (Llama-3-8B INT4)
```

### 4.3 性能基准

在不同 Jetson 设备上的推理性能：

| 模型 | 量化 | Orin Nano 8GB | Orin 32GB | Orin 64GB |
|------|------|--------------|-----------|-----------|
| Llama-3-8B | Q4_K_M | ~5 tok/s | ~15 tok/s | ~18 tok/s |
| Llama-3-8B | FP16 | OOM | ~10 tok/s | ~14 tok/s |
| Phi-3-mini | Q4_K_M | ~15 tok/s | ~30 tok/s | ~35 tok/s |
| LLaVA-1.6-7B | INT4 | ~2 tok/s | ~8 tok/s | ~10 tok/s |

---

## 5. 优化技术

### 5.1 投机解码（Speculative Decoding）

```
原理: 用小模型快速"猜测"多个 token，大模型验证

小模型 (draft): 快速生成 5 个候选 token  → [t1, t2, t3, t4, t5]
大模型 (target): 并行验证这 5 个 token   → [✓, ✓, ✓, ✗, -]
                                          接受前 3 个，从第 4 个重新开始

加速比: 2-3x（取决于小模型和大模型的一致性）
```

### 5.2 持续批处理（Continuous Batching）

```
传统批处理:
  Batch 1: [请求A: 50 tokens, 请求B: 200 tokens]
  等待最慢的请求完成才能处理下一批
  浪费: 请求A 空等 150 步

持续批处理 (vLLM):
  Step 1: 处理 [A, B]
  Step 50: A 完成，插入新请求 C
  Step 100: C 完成，插入新请求 D
  Step 200: B 完成
  效率: GPU 始终满载
```

### 5.3 注意力优化

```
Flash Attention:
  标准注意力: 需要 O(n²) 内存存储完整注意力矩阵
  Flash Attention: 分块计算，避免存储完整矩阵
  效果: 内存减少 5-20x，速度提升 2-4x

PagedAttention (vLLM):
  将 KV Cache 分为固定大小的"页"
  按需分配，避免内存碎片
  效果: 吞吐量提升 2-4x
```

---

## 6. 云端-边缘混合架构

### 6.1 架构设计

```
┌──────────────────────────────────────────────────┐
│                    云端                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ 大模型   │  │ 模型仓库  │  │ 任务管理     │   │
│  │ 70B+     │  │ 模型更新  │  │ 历史分析     │   │
│  └──────────┘  └──────────┘  └──────────────┘   │
└────────────────────┬─────────────────────────────┘
                     │ 网络连接
                     │ (可断开)
┌────────────────────┴─────────────────────────────┐
│                  边缘 (无人机)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ 小模型   │  │ CV 模型   │  │ 飞控系统     │   │
│  │ 7B INT4  │  │ YOLO 等   │  │ PX4/ArduPilot│  │
│  └──────────┘  └──────────┘  └──────────────┘   │
└──────────────────────────────────────────────────┘
```

### 6.2 降级策略

```python
class HybridInference:
    def __init__(self):
        self.edge_model = load_edge_model()   # 7B INT4
        self.cloud_client = CloudLLMClient()  # GPT-4
        self.network_status = "online"

    def infer(self, prompt, priority="normal"):
        # 高优先级或复杂任务 → 云端
        if priority == "high" and self.network_status == "online":
            return self._cloud_infer(prompt)

        # 简单任务 → 边缘
        if self._is_simple_task(prompt):
            return self._edge_infer(prompt)

        # 默认尝试云端，失败则降级到边缘
        if self.network_status == "online":
            try:
                return self._cloud_infer(prompt, timeout=5.0)
            except (TimeoutError, NetworkError):
                self._log_degradation("cloud_timeout")
                return self._edge_infer(prompt)
        else:
            return self._edge_infer(prompt)

    def _edge_infer(self, prompt):
        # 边缘推理，可能需要简化提示
        simplified = self._simplify_prompt(prompt)
        return self.edge_model.generate(simplified)

    def _is_simple_task(self, prompt):
        # 判断任务复杂度
        return len(prompt) < 200 and "复杂" not in prompt
```

---

## 思考题

1. 如果要在 Jetson Orin Nano 8GB 上同时运行 LLM (语言推理) 和 VLM (视觉理解)，你会如何分配内存资源？
2. GGUF Q4_K_M 和 AWQ INT4 两种量化方式各有什么优缺点？在什么场景下选择哪种？
3. 设计一个无人机的云端-边缘混合推理系统，说明在什么情况下使用云端、什么情况下使用边缘。
4. 投机解码需要一个"小模型"来猜测 token。如何选择或训练这个小模型？
5. 在网络完全断开的情况下，如何保证无人机仍能执行基本的智能任务？

<details>
<summary>参考答案</summary>

**1.** 方案：(1) LLM 使用 GGUF Q4_K_M 约 4GB；(2) VLM 使用 INT4 量化约 4GB；(3) 不同时运行 —— 采用分时复用：飞行中运行轻量 CV 模型（YOLO，<1GB），需要深度理解时加载 VLM 分析当前帧，需要语言推理时加载 LLM；(4) 使用模型卸载（offload）技术，将不活跃的模型暂存到 NVMe SSD。

**2.** GGUF Q4_K_M 优势：支持 CPU+GPU 混合推理，不需要全放显存，适合资源极度受限场景；劣势：GPU 利用率不如纯 GPU 方案。AWQ INT4 优势：纯 GPU 推理更快，与 vLLM/TensorRT-LLM 兼容性好；劣势：需要完整模型放入 GPU 内存。选择：Jetson 用 GGUF（灵活），服务器用 AWQ（速度）。

**3.** 云端使用场景：(1) 复杂任务规划（需要 70B+ 模型推理能力）；(2) 任务前规划（不紧急）；(3) 正常网络条件下的一般推理。边缘使用场景：(1) 实时避障决策（<50ms 延迟要求）；(2) 网络中断时的降级推理；(3) 简单指令理解（"起飞"、"降落"）；(4) 隐私敏感场景。切换策略：网络正常时优先云端，超时 3 秒降级到边缘。

**4.** 选择策略：(1) 使用同系列的小模型，如 Llama-3-8B 的 draft model 用 Llama-3-1B；(2) 知识蒸馏：用大模型的输出训练小模型；(3) Medusa 方案：在大模型上附加多个预测头，无需独立小模型。无人机场景推荐 Medusa，因为不需要额外的模型加载。

**5.** 策略：(1) 预加载常用任务模板到边缘模型；(2) 设计基于规则的后备系统（当 LLM 不可用时切换到确定性逻辑）；(3) 关键操作（起飞、降落、返航）始终由硬编码逻辑控制，不依赖 LLM；(4) 缓存最近的 LLM 推理结果以供参考。

</details>
