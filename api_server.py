import asyncio
import json
import time
import uuid
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator, Union, Any
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError, validator
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from database import Database

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# 全局变量
start_time = time.time()
request_count = 0


# 思考配置模型
class ThinkingConfig(BaseModel):
    thinking_budget: Optional[int] = None  # 0-32768, 0=禁用思考, None=自动
    include_thoughts: Optional[bool] = False  # 是否在响应中包含思考过程

    class Config:
        extra = "allow"

    @validator('thinking_budget')
    def validate_thinking_budget(cls, v):
        if v is not None:
            if not isinstance(v, int) or v < 0 or v > 32768:
                raise ValueError("thinking_budget must be an integer between 0 and 32768")
        return v


# 请求/响应模型
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]  # 支持字符串和数组格式

    class Config:
        # 允许额外字段，提高兼容性
        extra = "allow"

    @validator('content')
    def validate_content(cls, v):
        """验证并标准化content字段"""
        if isinstance(v, str):
            return v
        elif isinstance(v, list):
            # 将数组格式转换为字符串
            text_parts = []
            for item in v:
                if isinstance(item, dict):
                    if item.get('type') == 'text' and 'text' in item:
                        text_parts.append(item['text'])
                    elif 'text' in item:  # 兼容其他可能的格式
                        text_parts.append(item['text'])
                elif isinstance(item, str):
                    text_parts.append(item)
            return ' '.join(text_parts) if text_parts else ""
        else:
            raise ValueError("content must be string or array of content objects")

    def get_text_content(self) -> str:
        """获取纯文本内容"""
        # 由于validator已经将content标准化为字符串，这里直接返回
        return self.content if isinstance(self.content, str) else str(self.content)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None
    # 思考配置
    thinking_config: Optional[ThinkingConfig] = None

    class Config:
        # 允许额外字段，提高与OpenAI SDK的兼容性
        extra = "allow"

    # 自定义验证器，确保参数在合理范围内
    def __init__(self, **data):
        # 参数范围验证
        if 'temperature' in data and data['temperature'] is not None:
            data['temperature'] = max(0.0, min(2.0, data['temperature']))
        if 'top_p' in data and data['top_p'] is not None:
            data['top_p'] = max(0.0, min(1.0, data['top_p']))
        if 'n' in data and data['n'] is not None:
            data['n'] = max(1, min(4, data['n']))
        if 'max_tokens' in data and data['max_tokens'] is not None:
            data['max_tokens'] = max(1, min(8192, data['max_tokens']))

        super().__init__(**data)


# 内存缓存用于RPM/TPM限制
class RateLimitCache:
    def __init__(self):
        self.cache: Dict[str, Dict[str, List[tuple]]] = {}  # 改为按模型缓存
        self.lock = asyncio.Lock()

    async def add_usage(self, model_name: str, requests: int = 1, tokens: int = 0):
        async with self.lock:
            if model_name not in self.cache:
                self.cache[model_name] = {'requests': [], 'tokens': []}

            current_time = time.time()
            self.cache[model_name]['requests'].append((current_time, requests))
            self.cache[model_name]['tokens'].append((current_time, tokens))

    async def get_current_usage(self, model_name: str, window_seconds: int = 60) -> Dict[str, int]:
        async with self.lock:
            if model_name not in self.cache:
                return {'requests': 0, 'tokens': 0}

            current_time = time.time()
            cutoff_time = current_time - window_seconds

            # 清理过期记录
            self.cache[model_name]['requests'] = [
                (t, v) for t, v in self.cache[model_name]['requests']
                if t > cutoff_time
            ]
            self.cache[model_name]['tokens'] = [
                (t, v) for t, v in self.cache[model_name]['tokens']
                if t > cutoff_time
            ]

            # 计算总和
            total_requests = sum(v for _, v in self.cache[model_name]['requests'])
            total_tokens = sum(v for _, v in self.cache[model_name]['tokens'])

            return {'requests': total_requests, 'tokens': total_tokens}


# 保持唤醒机制（仅在Render环境启用）
async def keep_alive():
    """保持服务唤醒"""
    try:
        render_url = os.getenv('RENDER_EXTERNAL_URL')
        if render_url:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.get(f"{render_url}/wake")
                logger.info("Keep-alive ping sent successfully")
    except Exception as e:
        logger.warning(f"Keep-alive ping failed: {e}")


# 全局变量
db = Database()
rate_limiter = RateLimitCache()
scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler
    # 启动时的操作
    logger.info("🚀 Gemini API Proxy starting...")
    logger.info(f"📊 Available API keys: {len(db.get_available_gemini_keys())}")
    logger.info(f"🌍 Environment: {'Render' if os.getenv('RENDER_EXTERNAL_URL') else 'Local'}")

    # 启动保持唤醒调度器（仅在Render环境）
    render_url = os.getenv('RENDER_EXTERNAL_URL')
    if render_url:
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            keep_alive,
            'interval',
            minutes=14,  # 在15分钟睡眠前保持唤醒
            id='keep_alive',
            max_instances=1
        )
        scheduler.start()
        logger.info("⏰ Keep-alive scheduler started (14min interval)")

    yield

    # 关闭时的操作
    if scheduler:
        scheduler.shutdown()
        logger.info("⏰ Scheduler shutdown")
    logger.info("🛑 API Server shutting down...")


app = FastAPI(
    title="Gemini API Proxy",
    description="A high-performance proxy for Gemini API with OpenAI compatibility",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求计数中间件
@app.middleware("http")
async def count_requests(request: Request, call_next):
    global request_count
    request_count += 1

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # 记录请求日志
    logger.info(f"📡 {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")

    return response


# 全局异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误"""
    logger.warning(f"❌ Request validation error: {exc}")

    # 提取具体的错误信息
    error_details = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        msg = error["msg"]
        error_details.append(f"{field}: {msg}")

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": f"Request validation failed: {'; '.join(error_details)}",
                "type": "invalid_request_error",
                "code": "request_validation_error"
            }
        }
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """处理Pydantic验证错误"""
    logger.warning(f"❌ Pydantic validation error: {exc}")

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": f"Data validation failed: {str(exc)}",
                "type": "invalid_request_error",
                "code": "data_validation_error"
            }
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"💥 Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": "server_error"
            }
        }
    )


# 辅助函数
def get_actual_model_name(request_model: str) -> str:
    """获取实际使用的模型名称"""
    supported_models = db.get_supported_models()

    # 如果请求的是支持的模型，直接使用
    if request_model in supported_models:
        logger.info(f"✅ Using requested model: {request_model}")
        return request_model

    # 否则使用默认模型
    default_model = db.get_config('default_model_name', 'gemini-2.5-flash')
    logger.info(f"⚠️ Unsupported model: {request_model}, using default: {default_model}")
    return default_model


def inject_prompt_to_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """向消息中注入prompt"""
    inject_config = db.get_inject_prompt_config()

    if not inject_config['enabled'] or not inject_config['content']:
        return messages

    content = inject_config['content']
    position = inject_config['position']

    # 创建消息副本
    new_messages = messages.copy()

    if position == 'system':
        # 添加或更新system消息
        system_msg = None
        for i, msg in enumerate(new_messages):
            if msg.role == 'system':
                system_msg = msg
                break

        if system_msg:
            # 更新现有system消息
            new_content = f"{content}\n\n{system_msg.get_text_content()}"
            new_messages[i] = ChatMessage(role='system', content=new_content)
        else:
            # 添加新的system消息到开头
            new_messages.insert(0, ChatMessage(role='system', content=content))

    elif position == 'user_prefix':
        # 在第一个user消息前添加
        for i, msg in enumerate(new_messages):
            if msg.role == 'user':
                original_content = msg.get_text_content()
                new_content = f"{content}\n\n{original_content}"
                new_messages[i] = ChatMessage(role='user', content=new_content)
                break

    elif position == 'user_suffix':
        # 在最后一个user消息后添加
        for i in range(len(new_messages) - 1, -1, -1):
            if new_messages[i].role == 'user':
                original_content = new_messages[i].get_text_content()
                new_content = f"{original_content}\n\n{content}"
                new_messages[i] = ChatMessage(role='user', content=new_content)
                break

    return new_messages


def get_thinking_config(request: ChatCompletionRequest) -> Dict:
    """根据配置生成思考配置"""
    thinking_config = {}

    # 从数据库获取全局配置
    global_thinking_enabled = db.get_config('thinking_enabled', 'true').lower() == 'true'
    global_thinking_budget = int(db.get_config('thinking_budget', '-1'))  # -1 表示自动
    global_include_thoughts = db.get_config('include_thoughts', 'false').lower() == 'true'

    # 如果全局禁用思考，直接返回禁用配置
    if not global_thinking_enabled:
        return {"thinkingBudget": 0}

    # 如果请求中有思考配置，优先使用请求的配置
    if request.thinking_config:
        if request.thinking_config.thinking_budget is not None:
            thinking_config["thinkingBudget"] = request.thinking_config.thinking_budget
        elif global_thinking_budget >= 0:
            thinking_config["thinkingBudget"] = global_thinking_budget

        if request.thinking_config.include_thoughts is not None:
            thinking_config["includeThoughts"] = request.thinking_config.include_thoughts
        elif global_include_thoughts:
            thinking_config["includeThoughts"] = global_include_thoughts
    else:
        # 使用全局配置
        if global_thinking_budget >= 0:
            thinking_config["thinkingBudget"] = global_thinking_budget
        if global_include_thoughts:
            thinking_config["includeThoughts"] = global_include_thoughts

    return thinking_config


def openai_to_gemini(request: ChatCompletionRequest) -> Dict:
    """将OpenAI格式转换为Gemini格式"""
    contents = []

    for msg in request.messages:
        # 确保获取文本内容
        text_content = msg.get_text_content()

        if msg.role == "system":
            # Gemini没有system角色，将其转换为user消息
            contents.append({
                "role": "user",
                "parts": [{"text": f"[System]: {text_content}"}]
            })
        elif msg.role == "user":
            contents.append({
                "role": "user",
                "parts": [{"text": text_content}]
            })
        elif msg.role == "assistant":
            contents.append({
                "role": "model",
                "parts": [{"text": text_content}]
            })

    # 构建基本请求
    gemini_request = {
        "contents": contents,
        "generationConfig": {
            "temperature": request.temperature,
            "topP": request.top_p,
            "candidateCount": request.n,
        }
    }

    # 添加思考配置
    thinking_config = get_thinking_config(request)
    if thinking_config:
        gemini_request["generationConfig"]["thinkingConfig"] = thinking_config

    # 添加其他参数
    if request.max_tokens:
        gemini_request["generationConfig"]["maxOutputTokens"] = request.max_tokens

    if request.stop:
        gemini_request["generationConfig"]["stopSequences"] = request.stop

    return gemini_request


def extract_thoughts_and_content(gemini_response: Dict) -> tuple[str, str]:
    """从Gemini响应中提取思考过程和最终内容"""
    thoughts = ""
    content = ""

    for candidate in gemini_response.get("candidates", []):
        parts = candidate.get("content", {}).get("parts", [])

        for part in parts:
            if "text" in part:
                # 检查是否为思考部分
                if part.get("thought", False):
                    thoughts += part["text"]
                else:
                    content += part["text"]

    return thoughts, content


def gemini_to_openai(gemini_response: Dict, request: ChatCompletionRequest, usage_info: Dict = None) -> Dict:
    """将Gemini响应转换为OpenAI格式"""
    choices = []

    # 提取思考过程和内容
    thoughts, content = extract_thoughts_and_content(gemini_response)

    for i, candidate in enumerate(gemini_response.get("candidates", [])):
        # 构建响应消息
        message_content = content if content else ""

        # 如果有思考过程且配置要求包含，添加到响应中
        if thoughts and request.thinking_config and request.thinking_config.include_thoughts:
            message_content = f"**Thinking:**\n{thoughts}\n\n**Response:**\n{content}"

        choices.append({
            "index": i,
            "message": {
                "role": "assistant",
                "content": message_content
            },
            "finish_reason": map_finish_reason(candidate.get("finishReason", "STOP"))
        })

    # 构建响应
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": choices,
        "usage": usage_info or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }

    return response


def map_finish_reason(gemini_reason: str) -> str:
    """映射Gemini的结束原因到OpenAI格式"""
    mapping = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "OTHER": "stop"
    }
    return mapping.get(gemini_reason, "stop")


async def select_gemini_key_and_check_limits(model_name: str) -> Optional[Dict]:
    """选择可用的Gemini Key并检查模型限制"""
    available_keys = db.get_available_gemini_keys()

    if not available_keys:
        logger.warning("⚠️ No available Gemini keys found")
        return None

    # 获取模型配置（已经包含计算的总限制）
    model_config = db.get_model_config(model_name)
    if not model_config:
        logger.error(f"❌ Model config not found for: {model_name}")
        return None

    # 记录限制信息
    logger.info(
        f"📈 Model {model_name} limits: RPM={model_config['total_rpm_limit']}, TPM={model_config['total_tpm_limit']}, RPD={model_config['total_rpd_limit']}")
    logger.info(f"🔑 Available API keys: {len(available_keys)}")

    # 检查模型级别的限制（使用总限制）
    current_usage = await rate_limiter.get_current_usage(model_name)

    if (current_usage['requests'] >= model_config['total_rpm_limit'] or
            current_usage['tokens'] >= model_config['total_tpm_limit']):
        logger.warning(
            f"🚫 Model {model_name} has reached rate limits: requests={current_usage['requests']}/{model_config['total_rpm_limit']}, tokens={current_usage['tokens']}/{model_config['total_tpm_limit']}")
        return None

    # 检查RPD限制（使用总限制）
    day_usage = db.get_usage_stats(model_name, 'day')
    if day_usage['requests'] >= model_config['total_rpd_limit']:
        logger.warning(
            f"🚫 Model {model_name} has reached daily request limit: {day_usage['requests']}/{model_config['total_rpd_limit']}")
        return None

    # 负载均衡策略选择Key
    strategy = db.get_config('load_balance_strategy', 'least_used')

    if strategy == 'round_robin':
        # 简单轮询，选择第一个可用的
        selected_key = available_keys[0]
    else:
        # least_used策略，选择最少使用的Key（基于Key ID的使用分布）
        selected_key = available_keys[0]

    logger.info(f"🎯 Selected API key #{selected_key['id']} for model {model_name}")

    # 返回选中的Key和模型配置
    return {
        'key_info': selected_key,
        'model_config': model_config
    }


async def make_gemini_request_with_retry(
        gemini_key: str,
        gemini_request: Dict,
        model_name: str,
        max_retries: int = 3
) -> Dict:
    """带重试的Gemini API请求"""
    timeout = float(db.get_config('request_timeout', '60'))

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

                response = await client.post(
                    gemini_url,
                    json=gemini_request,
                    headers={"x-goog-api-key": gemini_key}
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    error_detail = response.json() if response.content else {"error": {"message": "Unknown error"}}
                    if attempt == max_retries - 1:  # 最后一次尝试
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=error_detail.get("error", {}).get("message", "Unknown error")
                        )
                    else:
                        logger.warning(f"🔄 Request failed (attempt {attempt + 1}), retrying...")
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                        continue

        except httpx.TimeoutException as e:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=504, detail="Request timeout")
            else:
                logger.warning(f"⏰ Request timeout (attempt {attempt + 1}), retrying...")
                await asyncio.sleep(2 ** attempt)
                continue
        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))
            else:
                logger.warning(f"❌ Request failed (attempt {attempt + 1}): {str(e)}, retrying...")
                await asyncio.sleep(2 ** attempt)
                continue

    raise HTTPException(status_code=500, detail="Max retries exceeded")


async def stream_gemini_response(
        gemini_key: str,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        key_info: Dict,
        model_name: str
) -> AsyncGenerator[str, None]:
    """处理Gemini的流式响应"""
    # 🔥 关键修复：添加 alt=sse 参数，这是官方文档要求的
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
    timeout = float(db.get_config('request_timeout', '60'))
    max_retries = int(db.get_config('max_retries', '3'))

    logger.info(f"🌊 Starting stream request to: {url}")

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                        "POST",
                        url,
                        json=gemini_request,
                        headers={"x-goog-api-key": gemini_key}
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        error_msg = error_text.decode() if error_text else "Unknown error"
                        logger.error(f"❌ Stream request failed with status {response.status_code}: {error_msg}")
                        yield f"data: {json.dumps({'error': {'message': error_msg, 'type': 'api_error', 'code': response.status_code}})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    stream_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                    created = int(time.time())
                    total_tokens = 0
                    thinking_sent = False
                    has_content = False
                    processed_lines = 0

                    logger.info(f"✅ Stream response started, status: {response.status_code}")

                    try:
                        # 🔥 按照官方文档的SSE格式处理
                        async for line in response.aiter_lines():
                            processed_lines += 1

                            if not line:
                                continue

                            # 记录原始行以便调试
                            if processed_lines <= 5:  # 只记录前几行
                                logger.debug(f"📝 Stream line {processed_lines}: {line[:100]}...")

                            # 官方SSE格式：每行以 "data: " 开头
                            if line.startswith("data: "):
                                json_str = line[6:]  # 去掉 "data: " 前缀

                                # 检查结束标志
                                if json_str.strip() == "[DONE]":
                                    logger.info("🏁 Received [DONE] signal from stream")
                                    break

                                if not json_str.strip():
                                    continue

                                try:
                                    data = json.loads(json_str)

                                    # 处理候选响应
                                    for candidate in data.get("candidates", []):
                                        content_data = candidate.get("content", {})
                                        parts = content_data.get("parts", [])

                                        for part in parts:
                                            if "text" in part:
                                                text = part["text"]
                                                if not text:  # 跳过空文本
                                                    continue

                                                total_tokens += len(text.split())
                                                has_content = True

                                                # 检查是否为思考部分
                                                is_thought = part.get("thought", False)

                                                # 如果是思考部分且配置不包含思考，跳过
                                                if is_thought and not (openai_request.thinking_config and
                                                                       openai_request.thinking_config.include_thoughts):
                                                    continue

                                                # 思考功能的处理逻辑
                                                if is_thought and not thinking_sent:
                                                    thinking_header = {
                                                        "id": stream_id,
                                                        "object": "chat.completion.chunk",
                                                        "created": created,
                                                        "model": openai_request.model,
                                                        "choices": [{
                                                            "index": 0,
                                                            "delta": {"content": "**Thinking:**\n"},
                                                            "finish_reason": None
                                                        }]
                                                    }
                                                    yield f"data: {json.dumps(thinking_header)}\n\n"
                                                    thinking_sent = True
                                                    logger.debug("🧠 Sent thinking header")
                                                elif not is_thought and thinking_sent:
                                                    response_header = {
                                                        "id": stream_id,
                                                        "object": "chat.completion.chunk",
                                                        "created": created,
                                                        "model": openai_request.model,
                                                        "choices": [{
                                                            "index": 0,
                                                            "delta": {"content": "\n\n**Response:**\n"},
                                                            "finish_reason": None
                                                        }]
                                                    }
                                                    yield f"data: {json.dumps(response_header)}\n\n"
                                                    thinking_sent = False
                                                    logger.debug("💬 Sent response header")

                                                # 发送文本内容
                                                chunk_data = {
                                                    "id": stream_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": created,
                                                    "model": openai_request.model,
                                                    "choices": [{
                                                        "index": 0,
                                                        "delta": {"content": text},
                                                        "finish_reason": None
                                                    }]
                                                }
                                                yield f"data: {json.dumps(chunk_data)}\n\n"

                                        # 检查是否结束
                                        finish_reason = candidate.get("finishReason")
                                        if finish_reason:
                                            finish_chunk = {
                                                "id": stream_id,
                                                "object": "chat.completion.chunk",
                                                "created": created,
                                                "model": openai_request.model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {},
                                                    "finish_reason": map_finish_reason(finish_reason)
                                                }]
                                            }
                                            yield f"data: {json.dumps(finish_chunk)}\n\n"
                                            yield "data: [DONE]\n\n"

                                            logger.info(
                                                f"✅ Stream completed with finish_reason: {finish_reason}, tokens: {total_tokens}")

                                            # 记录使用量
                                            await rate_limiter.add_usage(model_name, 1, total_tokens)
                                            return

                                except json.JSONDecodeError as e:
                                    logger.warning(f"⚠️ JSON decode error: {e}, line: {json_str[:200]}...")
                                    continue

                            # 处理其他SSE事件类型（如果需要）
                            elif line.startswith("event: "):
                                # 可以处理特定事件类型
                                continue
                            elif line.startswith("id: ") or line.startswith("retry: "):
                                # SSE元数据，忽略
                                continue

                        # 如果流正常结束但没有显式的结束信号
                        if has_content:
                            finish_chunk = {
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": openai_request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            yield f"data: {json.dumps(finish_chunk)}\n\n"
                            yield "data: [DONE]\n\n"

                            logger.info(
                                f"✅ Stream ended naturally, processed {processed_lines} lines, tokens: {total_tokens}")

                        # 如果确实没有收到任何内容，才回退
                        if not has_content:
                            logger.warning(
                                f"⚠️ Stream response had no content after processing {processed_lines} lines, falling back to non-stream")
                            try:
                                fallback_response = await make_gemini_request_with_retry(
                                    gemini_key, gemini_request, model_name, 1
                                )

                                thoughts, content = extract_thoughts_and_content(fallback_response)

                                if thoughts and openai_request.thinking_config and openai_request.thinking_config.include_thoughts:
                                    full_content = f"**Thinking:**\n{thoughts}\n\n**Response:**\n{content}"
                                else:
                                    full_content = content

                                if full_content:
                                    chunk_data = {
                                        "id": stream_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": openai_request.model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": full_content},
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(chunk_data)}\n\n"

                                    finish_chunk = {
                                        "id": stream_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": openai_request.model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": "stop"
                                        }]
                                    }
                                    yield f"data: {json.dumps(finish_chunk)}\n\n"
                                    total_tokens = len(full_content.split())

                                    logger.info(f"✅ Fallback completed, tokens: {total_tokens}")

                            except Exception as e:
                                logger.error(f"❌ Fallback request failed: {e}")
                                yield f"data: {json.dumps({'error': {'message': 'Failed to get response', 'type': 'server_error'}})}\n\n"

                        # 记录使用量
                        await rate_limiter.add_usage(model_name, 1, total_tokens)
                        yield "data: [DONE]\n\n"
                        return  # 成功完成

                    except (httpx.ReadError, httpx.RemoteProtocolError) as e:
                        logger.warning(f"🔌 Stream connection error (attempt {attempt + 1}): {str(e)}")
                        if attempt < max_retries - 1:
                            yield f"data: {json.dumps({'error': {'message': 'Connection interrupted, retrying...', 'type': 'connection_error'}})}\n\n"
                            await asyncio.sleep(1)
                            continue
                        else:
                            yield f"data: {json.dumps({'error': {'message': 'Stream connection failed after retries', 'type': 'connection_error'}})}\n\n"
                            yield "data: [DONE]\n\n"
                            return

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning(f"⏰ Connection error (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                yield f"data: {json.dumps({'error': {'message': f'Connection error, retrying... (attempt {attempt + 1})', 'type': 'connection_error'}})}\n\n"
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                yield f"data: {json.dumps({'error': {'message': 'Connection failed after all retries', 'type': 'connection_error'}})}\n\n"
                yield "data: [DONE]\n\n"
                return
        except Exception as e:
            logger.error(f"💥 Unexpected error in stream (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            else:
                yield f"data: {json.dumps({'error': {'message': 'Unexpected error occurred', 'type': 'server_error'}})}\n\n"
                yield "data: [DONE]\n\n"
                return


# API端点
@app.get("/")
async def root():
    """根端点"""
    return {
        "service": "Gemini API Proxy",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    available_keys = len(db.get_available_gemini_keys())
    uptime = time.time() - start_time

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "available_keys": available_keys,
        "environment": "render" if os.getenv('RENDER_EXTERNAL_URL') else "local",
        "uptime_seconds": int(uptime),
        "request_count": request_count,
        "version": "1.0.0"
    }


@app.get("/wake")
async def wake_up():
    """快速唤醒端点"""
    return {
        "status": "awake",
        "timestamp": datetime.now().isoformat(),
        "message": "Service is active"
    }


@app.get("/status")
async def get_status():
    """获取详细服务状态"""
    import psutil
    import sys

    process = psutil.Process(os.getpid())

    return {
        "service": "Gemini API Proxy",
        "status": "running",
        "version": "1.0.0",
        "render_url": os.getenv('RENDER_EXTERNAL_URL'),
        "python_version": sys.version,
        "models": db.get_supported_models(),
        "active_keys": len(db.get_available_gemini_keys()),
        "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "uptime_seconds": int(time.time() - start_time),
        "total_requests": request_count,
        "thinking_enabled": db.get_thinking_config()['enabled'],
        "keep_alive_active": scheduler is not None and scheduler.running
    }


@app.get("/metrics")
async def get_metrics():
    """获取服务指标"""
    import psutil

    process = psutil.Process(os.getpid())

    return {
        "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "active_connections": len(db.get_available_gemini_keys()),
        "uptime_seconds": int(time.time() - start_time),
        "requests_count": request_count,
        "database_size_mb": os.path.getsize(db.db_path) / 1024 / 1024 if os.path.exists(db.db_path) else 0
    }


@app.post("/v1/chat/completions")
async def chat_completions(
        request: ChatCompletionRequest,
        authorization: str = Header(None)
):
    try:
        # 验证API Key
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")

        api_key = authorization.replace("Bearer ", "")
        user_key = db.validate_user_key(api_key)

        if not user_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # 基础请求验证
        if not request.messages or len(request.messages) == 0:
            raise HTTPException(status_code=422, detail="Messages cannot be empty")

        # 验证消息格式
        for msg in request.messages:
            if not hasattr(msg, 'role') or not hasattr(msg, 'content'):
                raise HTTPException(status_code=422, detail="Invalid message format")
            if msg.role not in ['system', 'user', 'assistant']:
                raise HTTPException(status_code=422, detail=f"Invalid role: {msg.role}")

            # 确保content已经被标准化为字符串
            if not isinstance(msg.content, str):
                raise HTTPException(status_code=422, detail="Content validation failed")

        # 获取实际使用的模型名称
        actual_model_name = get_actual_model_name(request.model)

        # 注入prompt（如果启用）
        request.messages = inject_prompt_to_messages(request.messages)

        # 选择可用的Gemini Key并检查限制
        selection_result = await select_gemini_key_and_check_limits(actual_model_name)

        if not selection_result:
            raise HTTPException(
                status_code=429,
                detail=f"No available API keys or model {actual_model_name} has reached rate limits."
            )

        gemini_key_info = selection_result['key_info']
        model_config = selection_result['model_config']

        # 转换请求格式
        gemini_request = openai_to_gemini(request)

        # 记录请求
        await rate_limiter.add_usage(actual_model_name, 1, 0)
        db.log_usage(gemini_key_info['id'], user_key['id'], actual_model_name, 1, 0)

        if request.stream:
            # 流式响应
            return StreamingResponse(
                stream_gemini_response(
                    gemini_key_info['key'],
                    gemini_request,
                    request,
                    gemini_key_info,
                    actual_model_name
                ),
                media_type="text/event-stream"
            )
        else:
            # 非流式响应
            gemini_response = await make_gemini_request_with_retry(
                gemini_key_info['key'],
                gemini_request,
                actual_model_name,
                int(db.get_config('max_retries', '3'))
            )

            # 估算token使用量
            total_tokens = 0
            for candidate in gemini_response.get("candidates", []):
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        total_tokens += len(part["text"].split())

            # 记录token使用
            await rate_limiter.add_usage(actual_model_name, 0, total_tokens)
            db.log_usage(gemini_key_info['id'], user_key['id'], actual_model_name, 0, total_tokens)

            # 转换响应格式
            usage_info = {
                "prompt_tokens": len(str(request.messages).split()),
                "completion_tokens": total_tokens,
                "total_tokens": len(str(request.messages).split()) + total_tokens
            }

            openai_response = gemini_to_openai(gemini_response, request, usage_info)

            return JSONResponse(content=openai_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """列出可用的模型"""
    models = db.get_supported_models()

    model_list = []
    for model in models:
        model_list.append({
            "id": model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "google"
        })

    return {"object": "list", "data": model_list}


# 🔥 新增：管理端点（修复404错误）
@app.get("/admin/models/{model_name}")
async def get_model_config(model_name: str):
    """获取指定模型的配置"""
    try:
        model_config = db.get_model_config(model_name)
        if not model_config:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        return {
            "success": True,
            "model_name": model_name,
            **model_config
        }
    except Exception as e:
        logger.error(f"❌ Failed to get model config for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/models/{model_name}")
async def update_model_config(model_name: str, request: dict):
    """更新指定模型的配置"""
    try:
        # 验证模型是否存在
        if model_name not in db.get_supported_models():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not supported")

        # 提取允许更新的字段
        allowed_fields = ['single_api_rpm_limit', 'single_api_tpm_limit', 'single_api_rpd_limit', 'status']
        update_data = {}

        for field in allowed_fields:
            if field in request:
                update_data[field] = request[field]

        if not update_data:
            raise HTTPException(status_code=422, detail="No valid fields to update")

        # 更新模型配置
        success = db.update_model_config(model_name, **update_data)

        if success:
            logger.info(f"✅ Updated model config for {model_name}: {update_data}")
            return {
                "success": True,
                "message": f"Model {model_name} configuration updated successfully",
                "updated_fields": update_data
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update model configuration")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update model config for {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/models")
async def list_model_configs():
    """获取所有模型的配置"""
    try:
        model_configs = db.get_all_model_configs()
        return {
            "success": True,
            "models": model_configs
        }
    except Exception as e:
        logger.error(f"❌ Failed to get model configs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/config/gemini-key")
async def add_gemini_key(request: dict):
    """通过API添加Gemini密钥"""
    key = request.get("key")
    if key and db.add_gemini_key(key):
        logger.info(f"✅ Added new Gemini API key")
        return {"success": True, "message": "Key added successfully"}
    return {"success": False, "message": "Failed to add key"}


@app.post("/admin/config/user-key")
async def generate_user_key(request: dict):
    """生成用户密钥"""
    name = request.get("name", "API User")
    key = db.generate_user_key(name)
    logger.info(f"🔑 Generated new user key for: {name}")
    return {"success": True, "key": key, "name": name}


@app.post("/admin/config/thinking")
async def update_thinking_config(request: dict):
    """更新思考模式配置"""
    try:
        enabled = request.get('enabled')
        budget = request.get('budget')
        include_thoughts = request.get('include_thoughts')

        success = db.set_thinking_config(
            enabled=enabled,
            budget=budget,
            include_thoughts=include_thoughts
        )

        if success:
            logger.info(f"✅ Updated thinking config: {request}")
            return {
                "success": True,
                "message": "Thinking configuration updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update thinking configuration")

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Failed to update thinking config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/config/inject-prompt")
async def update_inject_prompt_config(request: dict):
    """更新提示词注入配置"""
    try:
        enabled = request.get('enabled')
        content = request.get('content')
        position = request.get('position')

        success = db.set_inject_prompt_config(
            enabled=enabled,
            content=content,
            position=position
        )

        if success:
            logger.info(f"✅ Updated inject prompt config: enabled={enabled}, position={position}")
            return {
                "success": True,
                "message": "Inject prompt configuration updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update inject prompt configuration")

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Failed to update inject prompt config: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/config")
async def get_all_config():
    """获取所有系统配置"""
    try:
        configs = db.get_all_configs()
        thinking_config = db.get_thinking_config()
        inject_config = db.get_inject_prompt_config()

        return {
            "success": True,
            "system_configs": configs,
            "thinking_config": thinking_config,
            "inject_config": inject_config
        }
    except Exception as e:
        logger.error(f"❌ Failed to get configs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/stats")
async def get_admin_stats():
    """获取管理统计"""
    return {
        "gemini_keys": len(db.get_all_gemini_keys()),
        "active_gemini_keys": len(db.get_available_gemini_keys()),
        "user_keys": len(db.get_all_user_keys()),
        "active_user_keys": len([k for k in db.get_all_user_keys() if k['status'] == 1]),
        "supported_models": db.get_supported_models(),
        "usage_stats": db.get_all_usage_stats(),
        "thinking_config": db.get_thinking_config(),
        "inject_config": db.get_inject_prompt_config()
    }


# 运行服务器的函数
def run_api_server(port: int = 8000):
    """运行API服务器"""
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Gemini API Proxy on port {port}")
    run_api_server(port)