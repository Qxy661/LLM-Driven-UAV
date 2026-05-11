# 01-03 Agent 架构设计

> 预计阅读：35 分钟 | 前置知识：01-01 大语言模型基础、01-02 LLM推理与提示工程

本文档介绍如何基于 LLM 构建智能体（Agent），涵盖主流 Agent 架构模式、记忆系统、规划模块，以及如何将 Agent 应用于无人机自主控制。

---

## 1. 什么是 LLM Agent？

### 1.1 Agent 的定义

LLM Agent 是一个以大语言模型为核心"大脑"，具备**感知、推理、规划、行动**能力的自主系统。它不只回答问题，而是能够：

- **感知环境**：接收来自传感器、用户输入、系统状态的信息
- **推理决策**：基于感知信息进行逻辑推理和判断
- **制定计划**：将复杂目标分解为可执行的步骤
- **执行行动**：调用工具或 API 实际改变环境状态
- **反思修正**：根据行动结果调整后续策略

### 1.2 Agent vs 传统 LLM 对话

| 维度 | 普通 LLM 对话 | LLM Agent |
|------|-------------|-----------|
| 交互模式 | 一问一答 | 持续自主运行 |
| 环境交互 | 无 | 主动感知和行动 |
| 记忆 | 仅当前上下文 | 短期 + 长期记忆 |
| 规划 | 一次性回答 | 多步规划与执行 |
| 工具使用 | 无或有限 | 调用多种外部工具 |
| 自主性 | 被动响应 | 主动决策 |

---

## 2. 主流 Agent 架构

### 2.1 ReAct 架构

ReAct（Reasoning + Acting）是最基础也最广泛使用的 Agent 架构：

```
┌──────────────────────────────────────────────┐
│                  ReAct Loop                  │
│                                              │
│   Thought → Action → Observation → Thought   │
│      ↑                                    │   │
│      └────────────────────────────────────┘   │
│                  (循环直到完成)                 │
└──────────────────────────────────────────────┘
```

**无人机场景示例**：

```
用户: "检查教学楼 A 栋北面墙体是否有裂缝"

Thought 1: 我需要先起飞，然后定位到 A 栋北面。
Action 1: takeoff(altitude=20)
Observation 1: ✅ 起飞成功，当前高度 20m，GPS: (31.23, 121.47)

Thought 2: 现在需要飞到 A 栋北面上方。
Action 2: goto_waypoint(lat=31.2305, lon=121.4710, alt=20)
Observation 2: ✅ 到达 A 栋北面

Thought 3: 我需要拍摄高分辨率照片用于裂缝检测。
Action 3: capture_image(resolution="high")
Observation 3: ✅ 照片已保存: img_20240115_001.jpg

Thought 4: 让我分析这张图片是否有裂缝。
Action 4: analyze_image("img_20240115_001.jpg", task="检测墙体裂缝")
Observation 4: 检测到疑似裂缝，位于建筑中部偏东，长度约2米

Thought 5: 发现了疑似裂缝，我需要多角度拍摄以确认。
Action 5: goto_waypoint(lat=31.2305, lon=121.4712, alt=15)
Observation 5: ✅ 降低高度并移动到东侧

Thought 6: 近距离拍摄确认。
Action 6: capture_image(resolution="high")
Observation 6: ✅ 照片已保存: img_20240115_002.jpg

Thought 7: 分析近距离照片。
Action 7: analyze_image("img_20240115_002.jpg", task="确认墙体裂缝详情")
Observation 7: 确认存在墙体裂缝，宽度约3-5mm，建议进一步人工检查

Thought 8: 任务完成，需要生成报告并返航。
Action 8: generate_report(findings="A栋北面墙体发现裂缝...")
Action 9: return_to_home()
```

### 2.2 Plan-and-Execute 架构

先制定完整计划，再逐步执行：

```python
class PlanAndExecuteAgent:
    def __init__(self, planner_llm, executor_llm):
        self.planner = planner_llm
        self.executor = executor_llm
        self.plan = []
        self.current_step = 0

    def run(self, goal: str) -> str:
        # 阶段 1: 制定计划
        self.plan = self._create_plan(goal)

        # 阶段 2: 逐步执行
        results = []
        for i, step in enumerate(self.plan):
            self.current_step = i
            result = self._execute_step(step)
            results.append(result)

            # 检查是否需要重新规划
            if self._should_replan(result):
                self.plan = self._replan(goal, results)
                # 从当前步骤重新开始

        return self._generate_summary(results)

    def _create_plan(self, goal):
        prompt = f"""你是一个无人机任务规划专家。
请将以下目标分解为可执行的飞行步骤：

目标：{goal}

要求：
1. 每个步骤必须是一个具体的飞行动作
2. 考虑安全约束（电量、高度限制、禁飞区）
3. 包含异常处理方案
4. 按执行顺序排列

输出格式：
- step 1: [动作描述]
- step 2: [动作描述]
..."""

        response = self.planner.generate(prompt)
        return self._parse_steps(response)

    def _execute_step(self, step):
        prompt = f"""执行以下无人机飞行步骤：
步骤 {self.current_step + 1}/{len(self.plan)}: {step}

当前状态：{self._get_current_state()}
可用工具：{self._get_available_tools()}

请决定调用哪个工具，传什么参数。"""

        return self.executor.generate(prompt)

    def _should_replan(self, result):
        # 如果执行失败或遇到意外情况，触发重新规划
        return "error" in result.lower() or "unexpected" in result.lower()
```

### 2.3 层级 Agent 架构

适合复杂的多无人机协同场景：

```
                    ┌─────────────┐
                    │  任务总管    │  ← 高层规划 LLM
                    │  (LLM)      │
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
     ┌──────┴──────┐ ┌────┴────┐  ┌──────┴──────┐
     │ 感知 Agent  │ │ 规划    │  │ 执行 Agent  │
     │ (VLM)       │ │ Agent   │  │ (LLM)       │
     └──────┬──────┘ │ (LLM)   │  └──────┬──────┘
            │        └────┬────┘         │
            │             │              │
     ┌──────┴──────┐      │       ┌──────┴──────┐
     │ 视觉分析    │      │       │ 飞控接口    │
     │ 语音识别    │      │       │ 检测系统    │
     └─────────────┘      │       └─────────────┘
                          │
                   ┌──────┴──────┐
                   │ 知识库/记忆 │
                   └─────────────┘
```

---

## 3. 记忆系统设计

### 3.1 记忆类型

Agent 的记忆系统可以分为三层：

| 记忆类型 | 类比 | 存储内容 | 生命周期 |
|---------|------|---------|---------|
| 感知记忆 | 感觉 | 当前传感器数据 | 毫秒级 |
| 工作记忆 | 短期记忆 | 当前任务上下文、对话历史 | 分钟级 |
| 长期记忆 | 长期记忆 | 飞行经验、历史任务、知识库 | 永久 |

### 3.2 向量数据库作为长期记忆

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

class UAVMemory:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            collection_name="uav_memory",
            embedding_function=self.embeddings
        )
        self.episodic_memory = []   # 任务经历
        self.semantic_memory = {}   # 知识条目

    def store_experience(self, task, result, lessons_learned):
        """存储任务经历"""
        doc = f"""
任务: {task}
结果: {result}
经验教训: {lessons_learned}
时间: {datetime.now().isoformat()}
"""
        self.vector_store.add_texts([doc])
        self.episodic_memory.append({
            "task": task, "result": result,
            "lessons": lessons_learned
        })

    def recall_similar(self, query, k=3):
        """检索相似经历"""
        results = self.vector_store.similarity_search(query, k=k)
        return results

    def get_relevant_context(self, current_task):
        """获取与当前任务相关的上下文"""
        similar = self.recall_similar(current_task)
        context = "相关历史经验:\n"
        for i, doc in enumerate(similar):
            context += f"{i+1}. {doc.page_content}\n"
        return context
```

### 3.3 飞行状态记忆

```python
class FlightStateMemory:
    """维护当前飞行状态的工作记忆"""

    def __init__(self):
        self.state = {
            "position": {"lat": 0, "lon": 0, "alt": 0},
            "battery": 100.0,
            "flight_mode": "idle",
            "mission_progress": 0.0,
            "warnings": [],
            "current_task": None,
            "completed_steps": [],
            "remaining_steps": [],
            "environment": {
                "wind_speed": 0,
                "visibility": "good",
                "temperature": 25
            }
        }

    def update(self, new_data):
        self.state.update(new_data)
        self._check_safety_constraints()

    def _check_safety_constraints(self):
        """自动检查安全约束"""
        if self.state["battery"] < 20:
            self.state["warnings"].append("电量低，建议返航")
        if self.state["environment"]["wind_speed"] > 10:
            self.state["warnings"].append("风速过大，注意安全")
        if self.state["position"]["alt"] > 120:
            self.state["warnings"].append("高度超过限制")

    def to_context_string(self):
        """转换为 LLM 可读的上下文字符串"""
        return f"""当前飞行状态:
- 位置: ({self.state['position']['lat']}, {self.state['position']['lon']})
- 高度: {self.state['position']['alt']}m
- 电量: {self.state['battery']}%
- 飞行模式: {self.state['flight_mode']}
- 当前任务: {self.state['current_task']}
- 已完成步骤: {self.state['completed_steps']}
- 待执行步骤: {self.state['remaining_steps']}
- 环境: 风速 {self.state['environment']['wind_speed']}m/s, 能见度 {self.state['environment']['visibility']}
- 警告: {self.state['warnings']}
"""
```

---

## 4. 规划模块

### 4.1 任务分解策略

LLM 可以将高层目标分解为可执行的子任务：

```
高层目标: "对整个校园进行安全巡逻"

分解层级:
Level 0: 校园安全巡逻
Level 1:
  ├── 1.1 起飞并进入巡逻模式
  ├── 1.2 巡逻东区
  ├── 1.3 巡逻西区
  ├── 1.4 巡逻北区
  ├── 1.5 巡逻南区
  └── 1.6 返航并降落

Level 2 (以 1.2 东区巡逻为例):
  ├── 1.2.1 飞到东区起始点
  ├── 1.2.2 沿东区主路巡逻
  ├── 1.2.3 检查东区停车场
  ├── 1.2.4 检查东区建筑群
  └── 1.2.5 移动到下一个区域

Level 3 (以 1.2.3 检查停车场为例):
  ├── 1.2.3.1 goto(东区停车场中心, alt=15)
  ├── 1.2.3.2 capture_image(resolution="high")
  ├── 1.2.3.3 analyze_image(task="检查可疑车辆")
  ├── 1.2.3.4 capture_image(resolution="high")  # 多角度
  └── 1.2.3.5 goto(东区建筑群入口)
```

### 4.2 依赖分析

任务步骤之间可能存在依赖关系：

```python
class TaskDependencyGraph:
    def __init__(self):
        self.tasks = {}
        self.dependencies = {}

    def add_task(self, task_id, description):
        self.tasks[task_id] = description
        self.dependencies[task_id] = []

    def add_dependency(self, task_id, depends_on):
        self.dependencies[task_id].append(depends_on)

    def get_executable_tasks(self, completed):
        """获取当前可执行的任务（所有依赖已完成）"""
        executable = []
        for task_id, deps in self.dependencies.items():
            if task_id not in completed and all(d in completed for d in deps):
                executable.append(task_id)
        return executable

    def get_parallel_groups(self):
        """识别可并行执行的任务组"""
        groups = []
        remaining = set(self.tasks.keys())
        completed = set()

        while remaining:
            executable = self.get_executable_tasks(completed)
            if not executable:
                break
            groups.append(executable)
            completed.update(executable)
            remaining -= set(executable)

        return groups


# 示例
graph = TaskDependencyGraph()
graph.add_task("takeoff", "起飞")
graph.add_task("goto_area_a", "飞往A区域")
graph.add_task("goto_area_b", "飞往B区域")
graph.add_task("scan_a", "扫描A区域")
graph.add_task("scan_b", "扫描B区域")
graph.add_task("report", "生成报告")

graph.add_dependency("goto_area_a", "takeoff")
graph.add_dependency("goto_area_b", "takeoff")
graph.add_dependency("scan_a", "goto_area_a")
graph.add_dependency("scan_b", "goto_area_b")
graph.add_dependency("report", "scan_a")
graph.add_dependency("report", "scan_b")

# 结果: goto_area_a 和 goto_area_b 可并行执行
```

---

## 5. LangChain Agent 实现

### 5.1 基于 LangChain 的 UAV Agent

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# 定义工具
def takeoff(altitude: str) -> str:
    alt = float(altitude)
    if alt > 120:
        return "错误: 高度超过120米限制"
    # 实际调用飞控 API
    return f"已起飞到 {alt} 米高度"

def goto_waypoint(params: str) -> str:
    # 解析参数: "lat, lon, alt"
    parts = params.split(",")
    lat, lon, alt = float(parts[0]), float(parts[1]), float(parts[2])
    return f"正在飞往 ({lat}, {lon}), 高度 {alt}m"

def capture_photo(_: str) -> str:
    return "照片已拍摄并保存"

def check_battery(_: str) -> str:
    return "当前电量: 85%, 预计可飞行时间: 25分钟"

# 创建工具列表
tools = [
    Tool(name="takeoff", func=takeoff,
         description="起飞到指定高度，输入为高度数值（米）"),
    Tool(name="goto_waypoint", func=goto_waypoint,
         description="飞往指定坐标，输入格式: 纬度,经度,高度"),
    Tool(name="capture_photo", func=capture_photo,
         description="拍摄照片，输入为任意字符串"),
    Tool(name="check_battery", func=check_battery,
         description="检查电池电量，输入为空字符串"),
]

# 创建 Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt = PromptTemplate.from_template("""你是一个专业的无人机飞行助手。

你可以使用以下工具：
{tools}

工具名称列表: {tool_names}

请使用以下格式：
Question: 用户的指令
Thought: 思考应该做什么
Action: 工具名称
Action Input: 工具输入
Observation: 工具返回的结果
... (重复 Thought/Action/Action Input/Observation)
Thought: 我现在知道最终答案
Final Answer: 对用户的最终回复

{agent_scratchpad}

Question: {input}""")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)

# 运行
result = executor.invoke({"input": "请帮我起飞到20米，然后飞到教学楼上空拍照"})
print(result["output"])
```

---

## 6. 错误处理与重试机制

在真实无人机场景中，LLM API 调用可能因网络波动、速率限制、服务超时等原因失败。Agent 必须具备健壮的错误处理能力，避免因单次调用失败导致整个任务中断。

### 6.1 指数退避重试装饰器

对于 LLM API 调用，使用指数退避（Exponential Backoff）策略自动重试瞬时错误：

```python
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0):
    """Decorator for LLM API calls with exponential backoff retry."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_type = type(e).__name__
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"Rate limited ({error_type}), retry {attempt+1}/{max_retries} in {delay:.1f}s")
                        time.sleep(delay)
                    elif "timeout" in str(e).lower() or "504" in str(e):
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"Timeout ({error_type}), retry {attempt+1}/{max_retries} in {delay:.1f}s")
                        time.sleep(delay)
                    elif "connection" in str(e).lower():
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(f"Connection error ({error_type}), retry {attempt+1}/{max_retries} in {delay:.1f}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"Non-retryable error: {error_type}: {e}")
                        raise
            logger.error(f"All {max_retries} retries exhausted")
            raise Exception(f"Max retries ({max_retries}) exceeded for {func.__name__}")
        return wrapper
    return decorator

# Usage example with safe JSON parsing
import json

def safe_llm_json(prompt, default=None):
    """Call LLM and parse JSON response with fallback."""
    try:
        response = llm_client.generate(prompt)
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning(f"JSON parse failed, using default")
        return default or {"action": "hover", "confidence": 0.0}
```

### 6.2 错误分类与处理策略

在无人机 Agent 中，不同类型的错误需要不同的处理策略：

| 错误类型 | 示例 | 是否可重试 | 处理策略 |
|---------|------|-----------|---------|
| 速率限制 | HTTP 429 | 是 | 指数退避重试 |
| 服务超时 | HTTP 504 / Timeout | 是 | 指数退避重试 |
| 网络中断 | ConnectionError | 是 | 退避重试 + 切换备用链路 |
| 参数错误 | HTTP 400 | 否 | 修正参数后重试 |
| 认证失败 | HTTP 401 | 否 | 刷新凭证 |
| 模型输出异常 | JSON 解析失败 | 是 | 重新生成 + 更严格提示 |
| 未知错误 | 其他异常 | 视情况 | 记录日志 + 降级处理 |

---

## 7. 自主性层级

### 7.1 自主性光谱

```
完全手动                                                      完全自主
  │                                                             │
  ▼                                                             ▼
遥控操作 → 辅助飞行 → 条件自主 → 高度自主 → 完全自主 → 超人自主
  │         │          │          │          │          │
  人工操控   自动悬停    按指令执行   自主规划    自主决策    超越人类
  全程手动   部分辅助    人工监督    人类审批    人类监控    自主进化
```

### 7.2 LLM Agent 的自主性级别

| 级别 | 描述 | LLM 角色 | 人类角色 |
|------|------|---------|---------|
| L1 | 建议模式 | 提供建议 | 人类做所有决策 |
| L2 | 审批模式 | 生成计划 | 人类审批后执行 |
| L3 | 监督模式 | 自主执行 | 人类监督关键节点 |
| L4 | 委托模式 | 自主执行 | 人类事后审计 |
| L5 | 自主模式 | 全自主 | 仅设定目标 |

**当前推荐**：在无人机场景中，建议使用 **L2-L3 级别**，保留人类在关键决策点的审批权。

---

## 思考题

1. ReAct 和 Plan-and-Execute 两种 Agent 架构各适合什么类型的无人机任务？举例说明。
2. 设计一个 UAV Agent 的记忆系统，说明短期记忆和长期记忆分别存储什么信息。
3. 如何处理 Agent 在执行过程中遇到的意外情况（如电量骤降、GPS 信号丢失）？
4. 为什么建议无人机场景使用 L2-L3 自主性级别？L4-L5 级别可能带来什么风险？
5. 如何评估一个 UAV Agent 的性能？设计 3-5 个评估指标。

<details>
<summary>参考答案</summary>

**1.** ReAct 适合需要实时感知和响应的任务，如搜索救援（需要根据发现动态调整）、实时跟踪。Plan-and-Execute 适合结构化任务，如按固定路线巡检、多区域测绘（可以提前规划好完整路线）。ReAct 更灵活但可能效率低，Plan-and-Execute 更高效但缺乏应变能力。

**2.** 短期记忆：当前 GPS 坐标、高度、电量、当前任务步骤、最近几轮对话、最近的传感器读数。长期记忆：历史任务记录和结果、该区域的地形信息、禁飞区地图、以往遇到的问题和解决方案、该无人机的性能特征。短期记忆存在内存中快速访问，长期记忆存在向量数据库中支持语义检索。

**3.** 设计异常处理链：(1) 监控模块持续检测异常；(2) 检测到异常时中断当前任务；(3) LLM 评估异常严重程度；(4) 低风险：调整策略继续；(5) 中风险：暂停任务，等待人类指令；(6) 高风险：立即执行应急预案（如电量低自动返航）。关键是为高风险场景设置**硬编码的安全规则**，不依赖 LLM 决策。

**4.** L2-L3 保留了人类监督，可以防止 LLM 的"幻觉"或错误推理导致危险操作。L4-L5 的风险：(a) LLM 可能误解环境信息做出错误决策；(b) 在异常情况下 LLM 的应急能力未经验证；(c) 法规和伦理要求人类对无人机行为负责；(d) 出事故时责任归属不清。

**5.** 评估指标：(1) 任务完成率 — 成功完成的任务占比；(2) 安全性 — 违反安全约束的次数；(3) 效率 — 完成任务的时间和能耗；(4) 指令理解准确率 — 正确理解用户意图的比例；(5) 异常处理成功率 — 遇到异常后成功恢复的比例。

</details>
