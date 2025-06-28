# 🚀 Gemini API 轮询服务

一个免费、简单、高性能的 Gemini API 轮询代理服务，提供 OpenAI 兼容的 API 接口，支持智能负载均衡、思考模式、使用统计等高级功能。

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## ✨ 核心功能

- 🔄 **OpenAI 兼容 API** - 完全兼容 OpenAI SDK，无缝替换
- 🧠 **思考模式支持** - 启用模型内部推理，提高复杂查询质量
- ⚖️ **智能负载均衡** - 多API密钥轮询，最大化吞吐量
- 📊 **实时使用监控** - 详细的请求统计和性能指标
- 🎯 **提示词注入** - 自动为请求添加自定义指令
- 🔐 **多用户管理** - 生成和管理多个访问密钥
- 🌐 **Web 管理界面** - 美观的 Streamlit 管理面板
- ☁️ **云端部署** - 支持 Render、Railway 等平台
- 🇨🇳 **中国友好** - 优化的网络配置，支持大陆访问

## 🚀 快速开始

### 方法一：一键部署到 Render（推荐）

1. **Fork 此仓库**到你的 GitHub 账户

2. **注册 Render 账户**
   - 访问 [render.com](https://render.com)
   - 使用 GitHub 账户登录

3. **部署服务**
   - 点击 "New +" → "Web Service"
   - 选择你 Fork 的仓库
   - 配置部署设置：
     ```
     Name: gemini-api-proxy
     Environment: Python 3
     Region: Oregon (US West)
     Build Command: pip install -r requirements.txt
     Start Command: python run_server.py
     ```

4. **等待部署完成**（约3-5分钟）

5. **配置 API 密钥**
   ```bash
   curl -X POST https://your-app.onrender.com/admin/config/gemini-key \
        -H "Content-Type: application/json" \
        -d '{"key": "your-gemini-api-key"}'
   ```

6. **生成用户密钥**
   ```bash
   curl -X POST https://your-app.onrender.com/admin/config/user-key \
        -H "Content-Type: application/json" \
        -d '{"name": "My API Key"}'
   ```

### 方法二：本地开发

1. **克隆项目**
   ```bash
   git clone https://github.com/Arain119/gemini-api-proxy.git
   cd gemini-api-proxy
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **启动服务**
   ```bash
   python run_server.py
   ```

4. **启动管理界面**
   ```bash
   streamlit run main.py
   ```

5. **访问服务**
   - API 服务: http://localhost:8000
   - 管理界面: http://localhost:8501

## 📋 项目结构

```
gemini-api-proxy/
├── api_server.py           # FastAPI 主服务
├── database.py             # 数据库管理
├── main.py                 # Streamlit 管理界面
├── run_server.py           # 启动脚本
├── requirements.txt        # Python 依赖
├── render.yaml            # Render 部署配置
├── .gitignore             # Git 忽略文件
└── README.md              # 项目文档
```

## 🔧 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `PORT` | 服务端口 | `8000` |
| `RENDER_EXTERNAL_URL` | Render 部署URL | 自动生成 |
| `API_BASE_URL` | API 基础地址 | `http://localhost:8000` |

### 系统配置

通过管理界面或 API 调用配置：

- **默认模型**: `gemini-2.5-flash` 或 `gemini-2.5-pro`
- **请求超时**: 60秒
- **最大重试**: 3次
- **负载均衡**: `least_used` 或 `round_robin`

## 📚 API 使用指南

### OpenAI SDK 兼容

```python
import openai

# 配置客户端
client = openai.OpenAI(
    api_key="your-generated-user-key",
    base_url="https://your-app.onrender.com/v1"
)

# 基础对话
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "user", "content": "你好，请介绍一下自己"}
    ]
)

print(response.choices[0].message.content)
```

### 流式响应

```python
stream = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "写一个Python函数"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### 思考模式

```python
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "解决这个数学问题：..."}],
    thinking_config={
        "thinking_budget": 8192,
        "include_thoughts": True
    }
)
```

### cURL 示例

```bash
# 基础请求
curl -X POST https://your-app.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-user-key" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# 流式请求
curl -X POST https://your-app.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-user-key" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }' \
  --no-buffer
```

## 🌐 支持的模型

| 模型名称 | 描述 | 思考模式 | 适用场景 |
|----------|------|----------|----------|
| `gemini-2.5-flash` | 快速响应版本 | ✅ | 日常对话、快速查询 |
| `gemini-2.5-pro` | 专业增强版本 | ✅ | 复杂推理、专业分析 |

## 🛠️ 高级功能

### 1. 思考模式配置

```python
# 通过 API 配置思考模式
response = requests.post(f"{API_BASE_URL}/admin/config/thinking", json={
    "enabled": True,
    "budget": 16384,  # -1=自动, 0=禁用, 1-32768=固定预算
    "include_thoughts": False
})
```

### 2. 提示词注入

```python
# 配置全局提示词注入
response = requests.post(f"{API_BASE_URL}/admin/config/inject-prompt", json={
    "enabled": True,
    "content": "你是一个专业的AI助手，请用中文回答。",
    "position": "system"  # system, user_prefix, user_suffix
})
```

### 3. 使用统计监控

```python
# 获取使用统计
stats = requests.get(f"{API_BASE_URL}/admin/stats").json()
print(f"今日请求: {stats['usage_stats']['gemini-2.5-flash']['day']['requests']}")
```

### 4. 健康检查和监控

```python
# 健康检查
health = requests.get(f"{API_BASE_URL}/health").json()
print(f"服务状态: {health['status']}")
print(f"可用密钥: {health['available_keys']}")

# 详细状态
status = requests.get(f"{API_BASE_URL}/status").json()
print(f"内存使用: {status['memory_usage_mb']:.1f}MB")
print(f"运行时间: {status['uptime_seconds']}秒")
```

## 🎨 管理界面功能

Web 管理界面提供以下功能：

- 📊 **实时监控** - 服务状态、使用率图表
- 🔑 **密钥管理** - 添加 Gemini 密钥、生成用户密钥
- 🤖 **模型配置** - 查看模型状态、调整限制
- ⚙️ **系统设置** - 思考模式、提示词注入配置

## 🔐 安全特性

- **API 密钥验证** - 所有请求需要有效的用户密钥
- **速率限制** - 基于模型的 RPM/TPM/RPD 限制
- **请求日志** - 详细的使用记录和审计
- **错误处理** - 优雅的错误处理和重试机制

## 📈 性能优化

### 自动唤醒机制

Render 免费版在15分钟无活动后会休眠，项目内置智能唤醒机制：

1. **内置定时器** - 每14分钟自动ping服务
2. **外部监控** - 推荐配置 UptimeRobot 监控
3. **智能重试** - 客户端自动处理冷启动延迟

### 负载均衡

```python
# 配置负载均衡策略
strategies = ["least_used", "round_robin"]
```

### 连接池优化

```python
# HTTP 连接池配置
async with httpx.AsyncClient(
    timeout=30.0,
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100,
        keepalive_expiry=30
    ),
    http2=True
) as client:
    # API 调用
```

## 🌍 中国大陆访问优化

### 1. 服务器地区选择
- ✅ 推荐：美国西海岸（Oregon、California）
- ✅ 备选：新加坡、日本
- ❌ 避免：美国东海岸、欧洲

### 2. CDN 加速配置

**使用 Cloudflare（推荐）:**

1. 注册 Cloudflare 账户
2. 添加域名到 Cloudflare
3. 配置 DNS 记录指向 Render 服务
4. 启用代理模式（橙色云朵）
5. SSL/TLS 设置为"灵活"

**配置示例:**
```dns
类型: A
名称: api
IPv4: [Render服务IP]
代理状态: 已代理
```

## 🐛 故障排除

### 常见问题

**1. 服务无响应**
```bash
# 检查服务状态
curl https://your-app.onrender.com/health

# 手动唤醒
curl https://your-app.onrender.com/wake
```

**2. API 调用失败**
- 检查用户密钥是否有效
- 确认 Authorization 头格式：`Bearer your-key`
- 查看错误日志获取详细信息

**3. 思考模式不工作**
- 确认使用支持思考的模型（2.5系列）
- 检查思考配置是否启用
- 验证思考预算设置

**4. 中国访问慢**
- 配置 Cloudflare CDN
- 选择美国西海岸服务器
- 增加客户端超时时间

### 调试工具

```python
# 获取调试信息
debug_info = requests.get(f"{API_BASE_URL}/debug/info").json()

# 获取服务指标
metrics = requests.get(f"{API_BASE_URL}/metrics").json()

# 查看系统状态
status = requests.get(f"{API_BASE_URL}/status").json()
```

## 📊 监控和维护

### 外部监控服务

**UptimeRobot 配置:**
1. 注册 [uptimerobot.com](https://uptimerobot.com)
2. 创建 HTTP 监控：
   ```
   URL: https://your-app.onrender.com/wake
   监控间隔: 5分钟
   ```

**Cron-job.org 配置:**
1. 注册 [cron-job.org](https://cron-job.org)
2. 创建定时任务：
   ```
   URL: https://your-app.onrender.com/wake
   间隔: */14 * * * * (每14分钟)
   ```

### 日志管理

```python
# 清理旧日志
from database import Database
db = Database()
deleted_count = db.cleanup_old_logs(days=30)
print(f"清理了 {deleted_count} 条日志记录")

# 数据库备份
success = db.backup_database("backup_20240101.db")
if success:
    print("数据库备份成功")
```

## 🚢 部署选项

### Render.com（推荐）
- ✅ 免费 750 小时/月
- ✅ 无需信用卡
- ❌ 15分钟无活动后休眠

### Railway.app
- ✅ $5 免费额度/月
- ✅ 无强制休眠
- ❌ 需要监控用量

### Streamlit Cloud
- ✅ 完全免费
- ✅ 适合管理界面
- ❌ 仅支持 Streamlit 应用

### 自托管
- ✅ 完全控制
- ✅ 无使用限制
- ❌ 需要运维经验

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境设置

1. Fork 项目并克隆
2. 安装开发依赖：`pip install -r requirements.txt`
3. 运行测试：`python -m pytest tests/`
4. 提交前检查代码格式：`black . && flake8`

### 提交规范

- `feat:` 新功能
- `fix:` 错误修复
- `docs:` 文档更新
- `style:` 代码格式
- `refactor:` 重构
- `test:` 测试相关

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的 Python Web 框架
- [Streamlit](https://streamlit.io/) - 快速构建数据应用
- [Render](https://render.com/) - 简单的云部署平台
- [Google Gemini](https://ai.google.dev/) - 强大的AI模型


<div align="center">

**⭐ 如果这个项目对你有帮助，请给个 Star！⭐**

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/gemini-api-proxy&type=Date)](https://star-history.com/#yourusername/gemini-api-proxy&Date)

</div>
