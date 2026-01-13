# 云边协同使用指南

本文档介绍HiAgent的云边协同功能，该功能允许边缘节点将轨迹数据上传到云端进行LLM评估，并接收高质量的总结反馈。

## 架构概述

```
边缘端 (Edge)                    云端 (Cloud)
┌─────────────────┐         ┌─────────────────────┐
│   Trajectory    │         │   Cloud Server      │
│   Summarizer    │◄────────┤   (FastAPI)         │
│                 │         │                     │
│  CloudConnector │────────►│  CloudEvaluator     │
│  (HTTP/WS)      │         │  (OpenAI/GLM LLM)   │
└─────────────────┘         └─────────────────────┘
```

## 目录结构

```
cloud/
├── api/                    # API接口层
│   ├── main.py            # FastAPI主服务器
│   └── start_cloud_server.py  # 服务器启动脚本
├── core/                   # 核心功能
│   ├── evaluator.py       # 轨迹评估器
│   └── llm_providers.py   # LLM提供者
├── src/                    # 协议定义
│   └── protocol.py        # 云边通信协议
├── .env                    # 环境配置
├── requirements.txt        # Python依赖
├── main.py                # 主入口文件
└── README.md              # 本文档
```

## 功能特性

### 边缘侧功能
- **本地总结**：快速的本地轨迹总结，确保实时响应
- **多模型支持**：支持OpenAI GPT和GLM模型
- **智能缓存**：缓存云端总结结果，减少重复计算
- **异步上传**：轨迹数据异步上传到云端，不阻塞执行
- **降级机制**：云端不可用时自动降级到本地总结

### 云端功能
- **多LLM支持**：同时支持OpenAI和GLM（智谱AI）模型
- **LLM评估**：使用强大的云端LLM进行深度轨迹分析
- **结构化总结**：生成包含关键洞察和改进建议的结构化总结
- **多客户端支持**：同时服务多个边缘节点
- **智能缓存**：避免重复评估相似轨迹

## 快速开始

### 1. 云端服务器设置

#### 安装依赖
```bash
cd cloud
pip install -r requirements.txt
```

#### 配置环境变量
编辑 `cloud/.env` 文件：

```bash
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# GLM API配置（智谱AI）
GLM_API_KEY=your_glm_api_key_here

# 服务器配置
CLOUD_HOST=0.0.0.0
CLOUD_PORT=8000

# LLM配置
# 支持格式：
# - OpenAI: "gpt-4", "gpt-3.5-turbo"
# - GLM: "glm-4", "glm:glm-4", "glm:glm-3-turbo"
CLOUD_MODEL=gpt-4

# 通用参数
TEMPERATURE=0.3
MAX_TOKENS=1000
TIMEOUT=30

# 日志配置
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE=true  # 是否输出到文件
LOG_TO_CONSOLE=true  # 是否输出到控制台
LOG_MAX_FILE_SIZE_MB=10  # 单个日志文件最大大小(MB)
LOG_BACKUP_COUNT=5  # 保留的日志文件备份数量
```

#### 启动云端服务器

```bash
python main.py
```
服务器启动后会显示配置信息，并监听指定的端口。

### 2. 边缘节点配置

#### 安装依赖
```bash
# OpenAI支持
pip install openai

# GLM支持（可选）
pip install zhipuai
```

#### 配置环境变量
编辑项目根目录的 `.env` 文件：
```bash
# OpenAI API配置
OPENAI_API_KEY=your_openai_api_key_here

# GLM API配置（可选）
GLM_API_KEY=your_glm_api_key_here

# 启用云边协同
ENABLE_CLOUD_COLLABORATION=true

# 云端服务器地址
CLOUD_WEBSOCKET_URL=ws://your-cloud-server:8000/ws

# 本地LLM模型配置
# 支持格式：
# - OpenAI: "gpt-4", "gpt-3.5-turbo"
# - GLM: "glm-4", "glm:glm-4", "glm:glm-3-turbo"
LOCAL_MODEL=gpt-3.5-turbo
```

#### 运行评估
```bash
# 正常运行评估脚本，会自动使用云边协同
bash evaluate_model.sh
```

## 模型支持

### OpenAI模型
- gpt-3.5-turbo
- gpt-4
- gpt-4-turbo-preview

### GLM模型（智谱AI）
- glm-4.6（最强性能模型）
- glm-4.5-air（平衡性能和成本）
- glm-4.6v（视觉理解模型）
- glm-4
- glm-3-turbo

## API接口

### HTTP REST API

#### 1. 健康检查
```http
GET /health
```

#### 2. 获取提供者信息
```http
GET /provider/info
```

响应示例：
```json
{
  "provider": "openai",
  "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
  "base_url": "https://api.openai.com/v1"
}
```

#### 3. 评估轨迹
```http
POST /evaluate
```

#### 4. 评估历史
```http
GET /history?limit=100
```

### WebSocket API

连接地址：`ws://localhost:8000/ws`

## 配置说明

### 环境变量

#### 云端服务器（cloud/.env）

**API配置**：
- `OPENAI_API_KEY`: OpenAI API密钥
- `OPENAI_API_BASE`: OpenAI API基础URL（可选）
- `GLM_API_KEY`: GLM API密钥（可选）

**服务器配置**：
- `CLOUD_HOST`: 服务器监听地址（默认：0.0.0.0）
- `CLOUD_PORT`: 服务器监听端口（默认：8000）

**LLM配置**：
- `CLOUD_MODEL`: 使用的LLM模型
  - OpenAI格式：`gpt-4`, `gpt-3.5-turbo`
  - GLM格式：`glm-4`, `glm:glm-4`

**通用参数**：
- `TEMPERATURE`: 采样温度（0-1，默认：0.3）
- `MAX_TOKENS`: 最大token数（默认：1000）
- `TIMEOUT`: 请求超时时间（秒，默认：30）

#### 边缘节点（项目根目录/.env）

**API配置**：
- `OPENAI_API_KEY`: OpenAI API密钥
- `GLM_API_KEY`: GLM API密钥

**云边协同配置**：
- `ENABLE_CLOUD_COLLABORATION`: 是否启用云边协同（true/false）
- `CLOUD_WEBSOCKET_URL`: 云端WebSocket URL

**本地模型配置**：
- `LOCAL_MODEL`: 本地LLM模型
  - OpenAI格式：`gpt-4`, `gpt-3.5-turbo`
  - GLM格式：`glm-4`, `glm:glm-4`

## 模型切换

### 切换云端模型
修改 `cloud/.env` 中的 `CLOUD_MODEL`：

```bash
# 使用OpenAI GPT-4
CLOUD_MODEL=gpt-4

# 使用GLM-4
CLOUD_MODEL=glm:glm-4

# 使用GLM-3-turbo
CLOUD_MODEL=glm:glm-3-turbo
```

### 切换边缘本地模型
修改项目根目录的 `.env` 中的 `LOCAL_MODEL`：

```bash
# 使用OpenAI
LOCAL_MODEL=gpt-3.5-turbo

# 使用GLM
LOCAL_MODEL=glm:glm-4
```

## 日志管理

### 日志文件位置

所有日志文件存储在 `cloud/logs/` 目录下：
- `cloud_server.log` - 主日志文件
- `cloud_server_error.log` - 错误日志文件

### 日志轮转

日志文件会自动轮转：
- 单个日志文件最大 10MB（可配置）
- 保留最近5个备份文件（可配置）
- 超过限制的旧日志文件会被自动删除

### 查看日志

```bash
# 查看最新日志
tail -f logs/cloud_server.log

# 查看错误日志
tail -f logs/cloud_server_error.log

# 使用日志管理工具
python utils/cleanup_logs.py --list  # 列出所有日志文件
python utils/cleanup_logs.py --cleanup  # 清理30天前的日志
```

## API文档

启动服务器后，访问 `http://localhost:8000/docs` 查看自动生成的交互式API文档。

## 故障排查

### 常见问题

1. **模型初始化失败**
   - 检查对应的API密钥是否正确
   - 确认已安装相应的Python库
   - OpenAI: `pip install openai`
   - GLM: `pip install zhipuai`

2. **连接失败**
   - 检查云端服务器是否运行
   - 确认网络连接和防火墙设置
   - 验证WebSocket URL配置

3. **API调用失败**
   - 检查API密钥是否有足够的额度
   - 确认模型名称是否正确
   - 检查网络连接

4. **总结质量不高**
   - 调整TEMPERATURE参数（较低值更确定性）
   - 增加MAX_TOKENS以获得更详细的总结
   - 尝试不同的模型

## 性能优化建议

1. **模型选择**
   - 实时场景：使用gpt-3.5-turbo或glm-3-turbo
   - 高质量场景：使用gpt-4或glm-4

2. **缓存策略**
   - 相似轨迹使用缓存总结
   - 定期清理过期缓存

3. **网络优化**
   - 使用压缩减少传输量
   - 调整超时时间

## 开发指南

### 添加新的LLM提供者

1. 在 `llm_providers.py` 中创建新的提供者类
2. 继承 `LLMProvider` 抽象基类
3. 实现必要的方法
4. 在 `LLMProviderFactory` 中注册新的提供者

### 自定义提示词

修改 `cloud_evaluator.py` 中的提示词模板以适应不同的使用场景。

## 参考链接

- [OpenAI API文档](https://platform.openai.com/docs/api-reference)
- [GLM API文档](https://open.bigmodel.cn/dev/api)
- [智谱AI快速开始](https://docs.bigmodel.cn/cn/guide/start/quick-start)