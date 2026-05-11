# 02-03 LLM 路径规划辅助

> 预计阅读：25 分钟 | 前置知识：02-02 LLM任务分解

本文档介绍如何利用 LLM 辅助无人机路径规划，包括航点生成、地形推理和禁飞区感知。

---

## 1. 传统路径规划 vs LLM 辅助

### 1.1 对比

| 维度 | 传统方法 (A*, RRT) | LLM 辅助 |
|------|-------------------|---------|
| 输入 | 地图 + 起终点 | 自然语言描述 |
| 环境理解 | 栅格/拓扑地图 | 语义理解 |
| 障碍处理 | 几何避障 | 语义避障（"绕过人群"） |
| 优化目标 | 距离/时间最短 | 多目标（安全、效率、视野） |
| 泛化能力 | 需要预建地图 | 可处理未知环境 |
| 实时性 | 毫秒级 | 秒级（需 LLM 推理） |

### 1.2 混合架构

```
最优方案: LLM 做高层路径规划 + 传统算法做底层避障

用户: "从A飞到B，避开建筑群，飞高一点"

LLM 规划 (秒级):
  → 生成粗略航点序列 [W1, W2, W3, W4]
  → 标注语义约束 "远离建筑群"

传统算法 (毫秒级):
  → A*/RRT* 在航点间生成精细路径
  → 实时避障（传感器数据）
  → 保证安全性

┌───────────┐     ┌───────────┐     ┌───────────┐
│ LLM 规划  │────→│ 航点优化  │────→│ 传统规划  │
│ (语义层)  │     │ (航点间)  │     │ (执行层)  │
└───────────┘     └───────────┘     └───────────┘
```

---

## 2. LLM 航点生成

### 2.1 基于语言描述的航点生成

```python
class LLMWaypointGenerator:
    """LLM 驱动的航点生成器"""

    def __init__(self, llm_client, map_client):
        self.llm = llm_client
        self.map = map_client

    def generate_waypoints(self, task_description, start_pos, constraints):
        # 获取区域地图信息
        area_info = self.map.get_area_description(start_pos, radius=500)

        prompt = f"""你是一个无人机路径规划专家。
请根据任务描述生成飞行航点。

任务: {task_description}
起始位置: ({start_pos['lat']}, {start_pos['lon']})
区域信息: {area_info}
约束条件: {constraints}

请生成航点序列，每个航点包含:
- latitude: 纬度
- longitude: 经度
- altitude: 飞行高度(米)
- speed: 飞行速度(m/s)
- action: 到达后执行的动作
- reason: 选择该航点的原因

输出 JSON 格式。"""

        waypoints = json.loads(self.llm.generate(prompt))

        # 验证航点安全性
        validated = self._validate_waypoints(waypoints)
        return validated

    def _validate_waypoints(self, waypoints):
        """验证航点是否安全"""
        validated = []
        for wp in waypoints:
            # 检查禁飞区
            if self.map.is_no_fly_zone(wp["latitude"], wp["longitude"]):
                wp = self._find_alternative(wp)
            # 检查障碍物
            if self.map.has_obstacle(wp["latitude"], wp["longitude"], wp["altitude"]):
                wp = self._adjust_altitude(wp)
            validated.append(wp)
        return validated
```

### 2.2 航点生成示例

```python
# 输入: "沿着河道巡逻，重点检查桥梁"
# 输出:
{
    "waypoints": [
        {"lat": 31.230, "lon": 121.470, "alt": 20, "speed": 5,
         "action": "takeoff", "reason": "起飞点"},
        {"lat": 31.231, "lon": 121.471, "alt": 15, "speed": 3,
         "action": "hover_and_scan", "reason": "河道起点"},
        {"lat": 31.232, "lon": 121.472, "alt": 10, "speed": 2,
         "action": "capture_photo", "reason": "第一座桥梁，降低高度拍照"},
        {"lat": 31.233, "lon": 121.473, "alt": 15, "speed": 3,
         "action": "hover_and_scan", "reason": "河道中段"},
        {"lat": 31.234, "lon": 121.474, "alt": 10, "speed": 2,
         "action": "capture_photo", "reason": "第二座桥梁"},
        {"lat": 31.235, "lon": 121.475, "alt": 15, "speed": 5,
         "action": "return_to_home", "reason": "返航"}
    ]
}
```

---

## 3. 地形推理

### 3.1 LLM 地形理解

```python
def terrain_reasoning(self, image_description, flight_context):
    """基于图像描述进行地形推理"""

    prompt = f"""基于以下航拍图像描述，分析地形特征并给出飞行建议。

图像描述: {image_description}
当前飞行参数: {flight_context}

请分析:
1. 地形类型（平坦/丘陵/山地/水域）
2. 障碍物分布（建筑/树木/电线）
3. 适合的飞行高度
4. 潜在风险点
5. 推荐的飞行路径

输出 JSON 格式。"""

    return json.loads(self.llm.generate(prompt))
```

### 3.2 地形推理示例

```python
# 输入图像描述: "图像显示一条河流穿过城市区域，河两侧是密集的建筑群，
#               河上有一座桥梁，桥梁附近有施工区域"

# LLM 推理输出:
{
    "terrain_type": "城市河流区域",
    "obstacles": [
        {"type": "建筑群", "risk": "high", "advice": "保持建筑群上方飞行"},
        {"type": "桥梁", "risk": "medium", "advice": "可从桥下或桥侧通过"},
        {"type": "施工区域", "risk": "high", "advice": "避开，可能有塔吊等高大障碍物"}
    ],
    "recommended_altitude": 25,
    "risk_points": [
        {"location": "施工区域", "reason": "可能有临时高大障碍物"},
        {"location": "桥梁下方", "reason": "GPS信号可能不稳定"}
    ],
    "recommended_path": "沿河道中心线上方飞行，避开施工区域"
}
```

---

## 4. 禁飞区感知

### 4.1 禁飞区数据库

```python
class NoFlyZoneManager:
    """禁飞区管理器"""

    def __init__(self):
        self.zones = []  # 禁飞区列表

    def load_zones(self, zones_data):
        """加载禁飞区数据"""
        for zone in zones_data:
            self.zones.append({
                "name": zone["name"],
                "type": zone["type"],  # permanent, temporary, conditional
                "polygon": zone["polygon"],  # 多边形顶点
                "altitude_range": zone.get("altitude_range", (0, 999)),
                "active_time": zone.get("active_time", "always"),
                "reason": zone.get("reason", "")
            })

    def check_point(self, lat, lon, alt):
        """检查某点是否在禁飞区内"""
        for zone in self.zones:
            if self._point_in_polygon(lat, lon, zone["polygon"]):
                if zone["altitude_range"][0] <= alt <= zone["altitude_range"][1]:
                    return {
                        "in_zone": True,
                        "zone_name": zone["name"],
                        "zone_type": zone["type"],
                        "reason": zone["reason"]
                    }
        return {"in_zone": False}

    def get_nearby_zones(self, lat, lon, radius_km=5):
        """获取附近的禁飞区"""
        nearby = []
        for zone in self.zones:
            dist = self._distance_to_zone(lat, lon, zone["polygon"])
            if dist < radius_km:
                nearby.append({**zone, "distance_km": dist})
        return sorted(nearby, key=lambda z: z["distance_km"])
```

### 4.2 LLM 禁飞区提示

```python
def plan_with_nfz_awareness(self, goal, position):
    """在任务规划中融入禁飞区信息"""
    nearby_zones = self.nfz_manager.get_nearby_zones(
        position["lat"], position["lon"]
    )

    prompt = f"""规划无人机路径时必须避开禁飞区。

任务: {goal}
当前位置: ({position['lat']}, {position['lon']})
附近禁飞区: {json.dumps(nearby_zones, ensure_ascii=False)}

重要规则:
1. 航点不得位于任何禁飞区内
2. 路径不得穿越禁飞区
3. 在禁飞区附近飞行时增加安全距离
4. 如任务目标位于禁飞区内，必须拒绝并说明原因

请生成安全的航点序列。"""

    return json.loads(self.llm.generate(prompt))
```

---

## 5. 多目标路径优化

### 5.1 LLM 路径评估

```python
def evaluate_path(self, waypoints, criteria):
    """使用 LLM 评估路径质量"""

    prompt = f"""评估以下无人机飞行路径的质量。

航点: {json.dumps(waypoints, ensure_ascii=False)}

评估标准:
1. 安全性 (0-10): 避障、禁飞区、应急处理
2. 效率性 (0-10): 路径长度、时间消耗
3. 覆盖性 (0-10): 任务目标覆盖程度
4. 可行性 (0-10): 电量、天气条件

请对每个标准打分并给出改进建议。"""

    return json.loads(self.llm.generate(prompt))
```

---

## 思考题

1. LLM 路径规划和传统路径规划算法（如 A*）应该如何协作？各负责什么层级？
2. 如何处理 LLM 生成的航点位于禁飞区边缘的情况？
3. 在没有精确地图信息的情况下，LLM 如何利用视觉信息进行路径规划？
4. 如何平衡路径的安全性和效率？LLM 能否理解这种权衡？
5. 设计一个完整的 LLM 辅助路径规划系统，从接收任务到输出可执行路径。

<details>
<summary>参考答案</summary>

**1.** LLM 负责高层语义规划：理解任务目标、生成粗略航点序列、考虑语义约束（"远离人群"、"靠近建筑"）。传统算法负责底层精细规划：在航点间生成几何路径、实时避障、优化飞行轨迹。两者通过航点列表连接。

**2.** 策略：(1) 验证阶段自动将禁飞区边缘的航点向外移动安全距离（如 50 米）；(2) 如果移动后无法完成任务，通知用户并建议替代方案；(3) LLM 规划时在 prompt 中强调"所有航点必须距离禁飞区边界至少 X 米"；(4) 使用禁飞区 buffer zone 作为硬约束。

**3.** 使用 VLM 分析实时摄像头画面，提取地形特征（建筑、道路、空地），将视觉描述输入 LLM 进行路径推理。结合 SLAM 构建的局部地图和 LLM 的语义理解，在未知环境中逐步探索和规划。

**4.** LLM 可以理解多目标权衡，通过在 prompt 中明确权重来引导："安全性权重 0.6，效率权重 0.4"。也可以让 LLM 生成多条候选路径，分别标注安全性和效率评分，由用户或规则选择最优方案。

**5.** 完整系统：(1) 接收自然语言任务 → (2) 意图解析 → (3) 查询禁飞区和地图信息 → (4) LLM 生成粗略航点 → (5) 安全验证（禁飞区、障碍物、电量） → (6) 传统算法优化航点间路径 → (7) LLM 评估整体路径质量 → (8) 输出可执行路径给飞控系统。

</details>
