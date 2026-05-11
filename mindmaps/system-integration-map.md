# 系统集成全景图

```mermaid
graph TB
    subgraph 云端服务
        A[大语言模型<br/>GPT-4/Llama-70B]
        B[大视觉模型<br/>GPT-4V/LLaVA-34B]
        C[任务管理平台]
        D[数据分析服务]
    end

    subgraph 边缘计算 (无人机机载)
        E[小语言模型<br/>Llama-7B INT4]
        F[小视觉模型<br/>LLaVA-7B INT4]
        G[实时 CV<br/>YOLO/SAM]
        H[ROS2 节点集群]
    end

    subgraph 飞控系统
        I[PX4/ArduPilot]
        J[传感器融合]
        K[电机控制]
    end

    subgraph 传感器
        L[RGB 相机]
        M[红外相机]
        N[LiDAR]
        O[IMU/GPS]
        P[麦克风阵列]
    end

    subgraph 通信链路
        Q[WiFi]
        R[4G/5G]
        S[MAVLink]
    end

    subgraph 人机交互
        T[语音控制]
        U[手势控制]
        V[移动 App]
        W[地面站]
    end

    %% 云端连接
    A <-->|API| R
    B <-->|API| R
    C <-->|任务管理| R
    D <-->|数据分析| R

    %% 边缘连接
    E <-->|推理| H
    F <-->|推理| H
    G <-->|检测| H
    H <-->|控制| S

    %% 飞控连接
    I <-->|MAVLink| S
    J <-->|传感器数据| I
    K <-->|PWM| I

    %% 传感器连接
    L -->|图像| H
    M -->|热成像| H
    N -->|点云| H
    O -->|状态| H
    P -->|音频| H

    %% 人机交互
    T -->|语音| P
    U -->|手势| L
    V -->|指令| Q
    W -->|指令| Q
```

## 系统层级

| 层级 | 组件 | 延迟要求 | 可靠性要求 |
|------|------|---------|-----------|
| 云端 | 大模型推理 | <5s | 高 |
| 通信 | 4G/5G/WiFi | <500ms | 中 |
| 边缘 | 小模型推理 | <3s | 高 |
| ROS2 | 节点通信 | <100ms | 很高 |
| 飞控 | PX4/ArduPilot | <1ms | 极高 |
| 传感器 | 数据采集 | <10ms | 极高 |

## 数据流

```
传感器 → 预处理 → VLM/CV 分析 → LLM 决策 → 任务规划 → 飞控执行 → 传感器反馈
```

## 故障降级策略

```
正常模式: 云端大模型 + 边缘小模型
    ↓ 网络中断
降级模式 1: 仅边缘小模型
    ↓ 边缘模型故障
降级模式 2: 规则引擎
    ↓ 规则引擎故障
降级模式 3: 飞控安全模式 (悬停/返航)
```
