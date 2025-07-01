# Gemini API 轮询代理服务

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

一个专为**Gemini多Key轮询**设计的 Gemini API 代理服务，通过智能轮询多个 API Key 突破单Key限制，提供 OpenAI 兼容接口和完整的管理界面。无需服务器，一键免费部署到 Render 平台，立即获得公网访问地址。

> 🔥 **核心价值：N个Key = N倍限制！** 
> 
> 单个gemini-2.5-pro限制100RPM？轮询10个Key瞬间变成1000RPM！  
> 告别API限制，享受丝滑AI体验！

## ✨ 核心特性

- 🔄 **智能轮询**：多个 Gemini API Key 自动轮询，突破单Key请求限制
- 📈 **倍增额度**：N个Key = N倍请求限制，告别额度不够的烦恼  
- 🛡️ **高可用性**：单个Key失效不影响服务，自动故障转移
- ⚖️ **负载均衡**：支持轮询(round-robin)和最少使用(least-used)策略
- 🚀 **一键部署**：Fork 仓库后直接在 Render 部署，10分钟获得公网地址
- 💰 **完全免费**：使用 Render 免费层，无需支付内网穿透或映射服务器费用
- 🎯 **OpenAI 兼容**：完全兼容 OpenAI SDK，无需修改现有代码
- 📊 **可视化管理**：Streamlit 构建的直观管理界面
- ⚡ **高性能**：FastAPI + 异步处理，支持流式响应
- 🔐 **安全可靠**：用户密钥管理、使用统计、速率限制

### 轮询策略

- **🔄 Round Robin（轮询）**：按顺序循环使用每个Key，平均分配负载
- **📊 Least Used（最少使用）**：优先使用使用量最少的Key，智能负载均衡

## 🎯 快速开始

### 1. Fork 本仓库

点击页面右上角的 **Fork** 按钮，将本项目复制到你的 GitHub 账户下。

### 2. 一键部署前后端服务

#### 2.1 创建 Render 账户
1. 访问 [Render.com](https://render.com)
2. 使用 GitHub 账户登录（推荐）或邮箱注册
3. **需要MasterCard或VisaCard验证身份！但完全免费**

#### 2.2 一键部署 Blueprint
1. 在 Render 控制台点击 **"New +"** → **"Blueprint"**
2. 选择 **"Connect a repository"**
3. 找到你刚刚 Fork 的 `gemini-api-proxy` 仓库，点击 **"Connect"**
4. 配置 Blueprint 参数：

```yaml
Name: gemini-api-services  # 自定义Blueprint名称
Branch: main
```

5. Render 会自动识别 `render.yaml` 文件并显示将要创建的服务：
   - ✅ **gemini-api-proxy** (后端API服务)
   - ✅ **gemini-proxy-admin** (前端管理界面)

6. 点击 **"Apply"** 开始部署

#### 2.3 等待部署完成
- ⏱️ **首次部署时间**：约5-10分钟
- 📊 **部署进度**：可在Dashboard中实时查看两个服务的构建状态
- ✅ **完成标志**：两个服务都显示绿色的"Live"状态

### 3. 配置服务连接

由于Render免费层的限制，需要手动配置前后端连接：

#### 3.1 获取后端地址
1. 在Render Dashboard中找到 **gemini-api-proxy** 服务
2. 复制其完整URL，格式类似：`https://gemini-api-proxy-xxx.onrender.com`

#### 3.2 配置前端环境变量
1. 点击进入 **gemini-proxy-admin** 服务
2. 转到 **"Environment"** 标签页
3. 添加环境变量：
   ```
   Key: API_BASE_URL
   Value: https://gemini-api-proxy-xxx.onrender.com
   ```
4. 点击 **"Save Changes"**
5. 前端会自动重新部署（约2-3分钟）

#### 3.2 访问管理界面
前端部署完成后，你会获得另一个地址，这就是你的管理界面地址。

## 🔧 配置指南

### 1. 添加多个 Gemini API Key

轮询的核心是配置多个API Key，建议至少添加3-5个Key：

1. 访问前端管理界面
2. 进入 **"密钥管理"** → **"Gemini 密钥"** 页面
3. **逐个添加**多个 Gemini API Key：

```
Key 1: AIzaSyXXXXXXXXXXXXXXXXXXXXXX
Key 2: AIzaSyYYYYYYYYYYYYYYYYYYYYYY  
Key 3: AIzaSyZZZZZZZZZZZZZZZZZZZZZZ
... 更多Key
```

4. API Key 获取方式：
   - 访问 [Google AI Studio](https://makersuite.google.com/app/apikey)
   - 登录并创建新的 API Key
   - 复制密钥（格式：`AIzaSy...`）
   - **建议创建多个项目，每个项目生成一个Key**

**💡 配置建议：**
- ✅ **至少3个Key**：保证基本的轮询效果
- ✅ **5-10个Key**：获得更高的请求限制和稳定性
- ✅ **不同项目的Key**：降低同时被限制的风险

### 2. 配置轮询策略

1. 进入 **"系统设置"** → **"轮询配置"** 页面
2. 选择轮询策略：
   - **Round Robin**：按顺序轮询，负载分配均匀
   - **Least Used（推荐）**：智能选择使用量最少的Key
3. 设置故障转移：自动跳过失效的Key

### 3. 生成用户访问密钥

1. 在管理界面进入 **"密钥管理"** → **"用户密钥"** 页面
2. 点击 **"生成新密钥"**，输入密钥名称
3. **立即保存生成的密钥**（格式：`sk-...`），它不会再次显示

### 3. 配置思考模式

1. 进入 **"系统设置"** → **"思考模式"** 页面
2. 启用思考模式以获得更好的推理能力
3. 选择合适的思考预算：
   - **自动**：让模型自动决定
   - **2.5flash最大思考预算 (24k)**：快速响应
   - **2.5pro最大思考预算 (32k)**：深度思考

## 📡 使用 API

配置完成后，你就可以使用 OpenAI SDK 访问轮询代理了。系统会自动在多个 Gemini Key 之间进行轮询，提供更高的请求限制和稳定性。

### 轮询效果展示

```python
# 假设你配置了5个Key，每个Key限制100 RPM
# 轮询后总限制 = 5 × 1000 = 500 RPM

import openai
import asyncio
import time

client = openai.OpenAI(
    api_key="sk-key",  # 你生成的用户密钥
    base_url="https://your-service-name.onrender.com/v1"  # 你的后端地址
)

# 高并发测试 - 轮询自动分配负载
async def test_polling_performance():
    tasks = []
    start_time = time.time()
    
    # 同时发送100个请求
    for i in range(100):
        task = asyncio.create_task(
            client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": f"请求 #{i}"}]
            )
        )
        tasks.append(task)
    
    # 等待所有请求完成
    responses = await asyncio.gather(*tasks)
    end_time = time.time()
    
    print(f"✅ 100个并发请求完成，耗时：{end_time - start_time:.2f}秒")
    print(f"🔄 系统自动在{len(responses)}个Key之间轮询分配")

# 运行测试
# asyncio.run(test_polling_performance())
```

### Python 示例

```python
import openai

client = openai.OpenAI(
    api_key="sk-your-user-key",  # 你生成的用户密钥
    base_url="https://your-service-name.onrender.com/v1"  # 你的后端地址
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "user", "content": "你好！"}
    ]
)

print(response.choices[0].message.content)
```

### Node.js 示例

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: 'sk-your-user-key',
  baseURL: 'https://your-service-name.onrender.com/v1',
});

const response = await openai.chat.completions.create({
  model: 'gemini-2.5-flash',
  messages: [{ role: 'user', content: '你好！' }],
});

console.log(response.choices[0].message.content);
```

### cURL 示例

```bash
curl -X POST "https://your-service-name.onrender.com/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-user-key" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "你好！"}]
  }'
```

**💡 轮询优势：**
- 🚀 **线性扩展**：N个Key = N倍限制，突破单Key瓶颈
- 🛡️ **高可用性**：单Key失效不影响整体服务
- ⚖️ **负载均衡**：请求均匀分配，避免单点过载

## 🎛️ 管理功能

### 控制台
- 📊 **轮询效率监控**：实时查看每个Key的使用分布
- 📈 **聚合请求统计**：所有Key的总体请求量图表
- 🔍 **Key状态分析**：快速识别失效或过载的Key
- 💹 **限制倍增显示**：直观展示轮询带来的限制提升

### 密钥管理
- 🔑 **多Key轮询管理**：批量添加、启用/禁用Gemini API Key
- 🔄 **轮询状态监控**：查看每个Key的轮询参与状态  
- 👤 **用户访问密钥生成**：为客户端生成访问令牌
- 🚦 **智能故障检测**：自动识别并跳过失效Key

### 轮询配置
- ⚖️ **负载均衡策略**：Round Robin / Least Used 策略切换
- 🎯 **Key权重设置**：为不同Key设置不同的使用权重
- 🛡️ **故障转移配置**：设置Key失效时的自动切换策略
- 📊 **使用率阈值**：单Key使用率预警和保护

### 模型配置
- ⚙️ **单Key限制设置**：配置每个Key的RPM/TPM限制
- 📊 **总限制计算**：自动计算轮询后的总体限制  
- 🔧 **模型状态管理**：启用/禁用特定模型的轮询

### 系统设置
- 🧠 思考模式配置
- 📝 提示词注入
- 📋 系统状态监控

## 🆓 Render 免费层说明

### 免费额度
- ⏰ **运行时间**：每月 750 小时
- 📶 **带宽**：每月 100GB 出站流量
- 💾 **数据库**：1GB PostgreSQL（90天）
- 🌍 **域名**：免费 `.onrender.com` 子域名
- 🔒 **HTTPS**：自动 SSL 证书

### 限制说明
- 🛌 **休眠机制**：15分钟无请求后自动休眠
- ⚡ **冷启动**：休眠后首次请求需要15-30秒唤醒
- 🔄 **自动重启**：系统可能随时重启服务

### 保活机制
本项目内置保活功能，每14分钟自动发送请求保持服务活跃，减少休眠时间。

## 🌐 自定义域名（可选）

Render 免费层支持自定义域名：

1. 在服务设置中添加自定义域名
2. 在域名提供商处添加 CNAME 记录
3. Render 自动提供 SSL 证书

## 🔧 高级配置

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `API_BASE_URL` | 后端API地址 | - |
| `PORT` | 服务端口 | 自动分配 |
| `PYTHONUNBUFFERED` | Python输出缓冲 | 1 |

### render.yaml 配置

```yaml
services:
  - type: web
    name: gemini-api-proxy
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: python run_server.py
    healthCheckPath: /health
    
    envVars:
      - key: PYTHONUNBUFFERED
        value: "1"
```

## 🚨 注意事项

### 轮询最佳实践
- 🔑 **Key数量建议**：3-10个Key为最佳，过多Key管理复杂
- 🌍 **Key来源分散**：使用不同Google账号和项目创建Key
- 📊 **监控使用率**：定期检查Key使用分布，确保轮询效果
- 🔄 **及时替换**：发现失效Key立即替换，保持轮询池健康
- ⚖️ **策略选择**：高并发场景用Round Robin，日常使用Least Used

### 安全提醒
- 🔐 用户密钥仅显示一次，请立即保存
- 🚫 不要在客户端直接使用 Gemini API Key
- 🔒 定期轮换所有 API 密钥
- 🛡️ 多Key轮询降低单点安全风险

### 常见问题

**Q: 需要多少个Key才有轮询效果？**
A: 至少2个Key，推荐3-5个Key获得最佳平衡。

**Q: 某个Key失效了怎么办？**
A: 系统自动跳过失效Key，继续使用其他Key，服务不中断。

**Q: 如何知道轮询是否在工作？**
A: 在管理界面可以看到每个Key的使用分布和轮询状态。

**Q: 服务访问很慢怎么办？**
A: 这是 Render 免费层的冷启动特性，等待15-30秒即可恢复正常。

**Q: 如何避免服务休眠？**
A: 项目内置保活机制，**即使是免费层也不会休眠！**

**Q: 可以商用吗？**
A: 本项目采用 CC BY-NC 4.0 许可证，仅允许非商业使用。

## 🛠️ 本地开发

### 环境要求
- Python 3.8+
- pip

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动后端
```bash
python run_server.py
```

### 启动前端
```bash
streamlit run main.py
```

## 📄 许可证

本项目采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) 许可证。

- ✅ 允许：分享、修改、分发
- ❌ 禁止：商业使用
- 📝 要求：署名原作者

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request


## ⭐ 如果这个项目对你有帮助，请给个 Star ⭐️

## 🙏 致谢

- [Google Gemini](https://deepmind.google/technologies/gemini/) - 强大的AI模型
- [Render](https://render.com) - 优秀的免费部署平台
- [FastAPI](https://fastapi.tiangolo.com/) - 现代Python Web框架
- [Streamlit](https://streamlit.io/) - 快速构建数据应用
