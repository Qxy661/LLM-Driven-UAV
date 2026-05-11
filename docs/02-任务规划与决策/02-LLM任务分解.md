# 02-02 LLM 任务分解

> 预计阅读：30 分钟 | 前置知识：02-01 自然语言任务理解、01-03 Agent架构设计

本文档介绍如何利用 LLM 将复杂的高层任务目标分解为可执行的子任务序列，涵盖层级分解、依赖分析和子目标生成。

---

## 1. 任务分解的必要性

### 1.1 为什么需要分解？

```
高层目标: "对校园进行全面安全巡逻"

问题: 飞控系统无法直接理解"全面安全巡逻"
      它只能执行: 起飞、goto(坐标)、hover(秒)、land() 等原子操作

需要: 将高层语义目标 → 分解为 → 底层飞行动作序列
```

### 1.2 分解的层级

| 层级 | 描述 | 示例 |
|------|------|------|
| L0 目标层 | 用户的高层意图 | "校园安全巡逻" |
| L1 任务层 | 主要任务阶段 | "巡逻东区"、"巡逻西区" |
| L2 子任务层 | 具体动作序列 | "飞到东区入口"、"拍照" |
| L3 原子操作层 | 飞控可执行指令 | "goto(31.23,121.47,15)" |

---

## 2. LLM 任务分解方法

### 2.1 基于 Prompt 的分解

```python
class LLMTaskDecomposer:
    """使用 LLM 进行任务分解"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def decompose(self, goal: str, context: dict) -> list:
        prompt = f"""你是一个无人机任务规划专家。
请将以下高层目标分解为可执行的子任务序列。

高层目标: {goal}

当前上下文:
- 位置: ({context['lat']}, {context['lon']})
- 高度: {context['altitude']}m
- 电量: {context['battery']}%
- 可用传感器: {context['sensors']}
- 已知地标: {context['landmarks']}

分解要求:
1. 每个子任务必须是飞控系统可执行的操作
2. 考虑电量限制和安全约束
3. 标注子任务之间的依赖关系
4. 估计每个子任务的耗时
5. 包含异常处理方案

可用的飞行动作:
- takeoff(altitude)
- goto(lat, lon, alt)
- hover(seconds)
- capture_photo(resolution)
- record_video(duration)
- scan_area(corner_points)
- return_to_home()
- land()

请输出 JSON 格式的任务列表。"""

        response = self.llm.generate(prompt)
        return json.loads(response)
```

### 2.2 分解示例

```python
# 输入: "检查教学楼A和B的屋顶状况"
# 输出:
{
    "task_name": "教学楼屋顶检查",
    "estimated_time": 600,  # 秒
    "steps": [
        {
            "id": 1,
            "action": "takeoff",
            "params": {"altitude": 25},
            "duration": 15,
            "dependencies": []
        },
        {
            "id": 2,
            "action": "goto",
            "params": {"lat": 31.2305, "lon": 121.4710, "alt": 25},
            "duration": 30,
            "dependencies": [1]
        },
        {
            "id": 3,
            "action": "scan_area",
            "params": {"area": "教学楼A屋顶", "pattern": "grid", "spacing": 5},
            "duration": 120,
            "dependencies": [2]
        },
        {
            "id": 4,
            "action": "capture_photo",
            "params": {"resolution": "high", "count": "multiple"},
            "duration": 30,
            "dependencies": [3]
        },
        {
            "id": 5,
            "action": "goto",
            "params": {"lat": 31.2310, "lon": 121.4720, "alt": 25},
            "duration": 25,
            "dependencies": [4]
        },
        {
            "id": 6,
            "action": "scan_area",
            "params": {"area": "教学楼B屋顶", "pattern": "grid", "spacing": 5},
            "duration": 120,
            "dependencies": [5]
        },
        {
            "id": 7,
            "action": "capture_photo",
            "params": {"resolution": "high", "count": "multiple"},
            "duration": 30,
            "dependencies": [6]
        },
        {
            "id": 8,
            "action": "return_to_home",
            "params": {},
            "duration": 45,
            "dependencies": [7]
        },
        {
            "id": 9,
            "action": "land",
            "params": {},
            "duration": 15,
            "dependencies": [8]
        }
    ]
}
```

---

## 3. 依赖分析与执行调度

### 3.1 依赖图构建

```python
class TaskDependencyGraph:
    """任务依赖图"""

    def __init__(self):
        self.tasks = {}
        self.edges = {}  # task_id -> [dependent_task_ids]

    def add_task(self, task_id, task_info):
        self.tasks[task_id] = task_info
        self.edges[task_id] = []

    def add_dependency(self, task_id, depends_on):
        """task_id 依赖 depends_on"""
        if depends_on not in self.edges:
            self.edges[depends_on] = []
        self.edges[depends_on].append(task_id)

    def get_execution_order(self):
        """拓扑排序获取执行顺序"""
        in_degree = {t: 0 for t in self.tasks}
        for deps in self.edges.values():
            for d in deps:
                in_degree[d] += 1

        queue = [t for t, d in in_degree.items() if d == 0]
        order = []

        while queue:
            # 可以并行执行的任务
            current_level = []
            for task in queue:
                current_level.append(task)
                order.append(task)

            queue = []
            for task in current_level:
                for dependent in self.edges[task]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        return order

    def get_parallel_groups(self):
        """获取可并行执行的任务组"""
        in_degree = {t: 0 for t in self.tasks}
        for deps in self.edges.values():
            for d in deps:
                in_degree[d] += 1

        queue = [t for t, d in in_degree.items() if d == 0]
        groups = []

        while queue:
            groups.append(queue[:])
            next_queue = []
            for task in queue:
                for dependent in self.edges[task]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_queue.append(dependent)
            queue = next_queue

        return groups
```

### 3.2 多无人机并行调度

```python
class MultiUAVScheduler:
    """多无人机任务调度器"""

    def __init__(self, uav_count: int):
        self.uav_count = uav_count
        self.uav_status = {i: {"idle": True, "position": None} for i in range(uav_count)}

    def schedule(self, dependency_graph: TaskDependencyGraph):
        """将任务分配给多架无人机"""
        parallel_groups = dependency_graph.get_parallel_groups()
        assignments = []

        for group in parallel_groups:
            available_uavs = [u for u, s in self.uav_status.items() if s["idle"]]
            group_assignments = []

            for i, task_id in enumerate(group):
                uav_id = available_uavs[i % len(available_uavs)]
                group_assignments.append({
                    "uav": uav_id,
                    "task": task_id,
                    "task_info": dependency_graph.tasks[task_id]
                })
                self.uav_status[uav_id]["idle"] = False

            assignments.append(group_assignments)

            # 等待组内所有任务完成后，标记无人机空闲
            for a in group_assignments:
                self.uav_status[a["uav"]]["idle"] = True

        return assignments
```

---

## 4. 迭代式分解

### 4.1 逐层细化策略

```python
class IterativeDecomposer:
    """迭代式任务分解 — 先粗后细"""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.max_depth = 3

    def decompose_recursive(self, goal, depth=0):
        if depth >= self.max_depth:
            return [{"action": goal, "type": "atomic"}]

        # LLM 判断是否需要进一步分解
        check_prompt = f"""判断以下任务是否可以直接由飞控系统执行:
任务: "{goal}"

如果可以直接执行（如"起飞"、"飞到某坐标"、"拍照"），回答 "atomic"
如果需要进一步分解，回答 "decompose"
"""
        decision = self.llm.generate(check_prompt).strip()

        if "atomic" in decision:
            return [{"action": goal, "type": "atomic"}]

        # 进一步分解
        decompose_prompt = f"""将以下任务分解为 2-5 个子任务:
任务: "{goal}"
要求: 每个子任务应该是更具体、更可执行的操作。
"""
        subtasks = self.llm.generate(decompose_prompt)
        subtask_list = self._parse_subtasks(subtasks)

        # 递归分解每个子任务
        result = []
        for subtask in subtask_list:
            result.extend(self.decompose_recursive(subtask, depth + 1))

        return result
```

### 4.2 分解深度控制

```
深度 0: "校园安全巡逻"
深度 1: ["巡逻东区", "巡逻西区", "巡逻北区", "巡逻南区"]
深度 2: ["飞到东区", "沿东区主路飞行", "检查停车场", ...]
深度 3: ["goto(31.23,121.47,15)", "hover(10)", "capture_photo()", ...]
                                                          ↑
                                                    到达原子操作，停止分解
```

---

## 5. 世界知识增强

### 5.1 知识库辅助分解

```python
class KnowledgeEnhancedDecomposer:
    """使用知识库增强任务分解"""

    def __init__(self, llm_client, knowledge_base):
        self.llm = llm_client
        self.kb = knowledge_base

    def decompose(self, goal, location_context):
        # 1. 检索相关知识
        relevant_knowledge = self.kb.search(goal, k=5)

        # 2. 获取区域信息
        area_info = self.kb.get_area_info(location_context["area"])

        prompt = f"""请分解以下无人机任务。

任务: {goal}
位置信息: {area_info}
相关知识: {relevant_knowledge}

在分解时请考虑:
1. 该区域的地理特征（建筑高度、障碍物）
2. 已知的禁飞区
3. 历史任务中的经验教训
4. 最优的飞行路径

输出 JSON 格式的任务列表。"""

        return json.loads(self.llm.generate(prompt))
```

---

## 6. 失败处理与备选方案

### 6.1 分解时生成备选方案

```python
def decompose_with_alternatives(self, goal):
    prompt = f"""分解任务并为关键步骤提供备选方案。

任务: {goal}

输出格式:
{{
    "primary_plan": [...],
    "alternatives": {{
        "step_id": {{
            "reason": "失败原因",
            "alternative_steps": [...]
        }}
    }}
}}

示例备选场景:
- 目标位置无法到达（障碍物/禁飞区）
- 电量不足无法完成全部任务
- 天气突变需要提前返航
- 目标未找到需要扩大搜索范围"""

    return json.loads(self.llm.generate(prompt))
```

---

## 思考题

1. 比较基于 Prompt 的任务分解和基于规则的任务分解，各有什么优缺点？
2. 如何处理 LLM 分解出的任务序列在执行过程中某个步骤失败的情况？
3. 在多无人机协同场景中，如何优化任务分配以最小化总任务时间？
4. 迭代式分解的深度应该如何控制？过深或过浅各有什么问题？
5. 如何评估 LLM 任务分解的质量？设计评估指标。

<details>
<summary>参考答案</summary>

**1.** Prompt 分解优点：灵活，能处理开放域任务，不需要预定义规则；缺点：输出不稳定，可能产生不可执行的步骤。规则分解优点：确定性高，输出始终可执行；缺点：覆盖范围有限，无法处理新场景。最佳实践：LLM 分解 + 规则验证，取两者之长。

**2.** 策略：(1) 在分解阶段就生成备选方案；(2) 执行失败时，将失败信息反馈给 LLM，请求重新规划剩余步骤；(3) 对于关键失败（如电量不足），触发硬编码的应急预案（立即返航）；(4) 使用 Plan-and-Execute 架构支持动态重规划。

**3.** 优化策略：(1) 按任务位置聚类，减少无人机飞行距离；(2) 考虑每架无人机的电量和位置，就近分配；(3) 对于可并行的任务，均匀分配到各无人机；(4) 使用 LLM 分析任务依赖，最大化并行度。

**4.** 过深：分解出过多细碎步骤，增加执行开销和出错概率，且 LLM 可能产生语义漂移。过浅：分解不充分，子任务仍然不够具体，飞控无法执行。控制方法：(1) 设定最大深度（通常 3-4 层）；(2) LLM 判断是否为原子操作；(3) 检查子任务是否能映射到预定义的飞行动作原语。

**5.** 评估指标：(1) 可执行性 — 分解出的步骤是否都能映射到飞行动作；(2) 完整性 — 是否覆盖了目标的所有要求；(3) 效率 — 步骤数和总耗时是否合理；(4) 安全性 — 是否遵守安全约束；(5) 正确性 — 执行后是否真正完成了原始目标。可以用人工标注的 ground truth 做对比评估。

</details>
