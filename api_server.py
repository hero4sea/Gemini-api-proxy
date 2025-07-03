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
    def __init__(self, max_entries: int = 10000):
        self.cache: Dict[str, Dict[str, List[tuple]]] = {}
        self.max_entries = max_entries
        self.lock = asyncio.Lock()

    async def cleanup_expired(self, window_seconds: int = 60):
        """定期清理过期缓存"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        async with self.lock:
            for model_name in list(self.cache.keys()):
                if model_name in self.cache:
                    self.cache[model_name]['requests'] = [
                        (t, v) for t, v in self.cache[model_name]['requests']
                        if t > cutoff_time
                    ]
                    self.cache[model_name]['tokens'] = [
                        (t, v) for t, v in self.cache[model_name]['tokens']
                        if t > cutoff_time
                    ]

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


# 健康检测功能
async def check_gemini_key_health(api_key: str, timeout: int = 10) -> Dict[str, Any]:
    """检测单个Gemini Key的健康状态"""
    test_request = {
        "contents": [{"role": "user", "parts": [{"text": "Test"}]}],
        "generationConfig": {"maxOutputTokens": 1}
    }

    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                json=test_request,
                headers={"x-goog-api-key": api_key}
            )

        response_time = time.time() - start_time

        if response.status_code == 200:
            return {
                "healthy": True,
                "response_time": response_time,
                "status_code": response.status_code,
                "error": None
            }
        else:
            return {
                "healthy": False,
                "response_time": response_time,
                "status_code": response.status_code,
                "error": f"HTTP {response.status_code}"
            }

    except asyncio.TimeoutError:
        return {
            "healthy": False,
            "response_time": timeout,
            "status_code": None,
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "healthy": False,
            "response_time": time.time() - start_time,
            "status_code": None,
            "error": str(e)
        }


# 全局变量
db = Database()
rate_limiter = RateLimitCache()
scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler
    # 启动时的操作
    logger.info("Starting Gemini API Proxy...")
    logger.info(f"Available API keys: {len(db.get_available_gemini_keys())}")
    logger.info(f"Environment: {'Render' if os.getenv('RENDER_EXTERNAL_URL') else 'Local'}")

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
        logger.info("Keep-alive scheduler started (14min interval)")

    yield

    # 关闭时的操作
    if scheduler:
        scheduler.shutdown()
        logger.info("Scheduler shutdown")
    logger.info("API Server shutting down...")


app = FastAPI(
    title="Gemini API Proxy",
    description="A high-performance proxy for Gemini API with OpenAI compatibility",
    version="1.1.0",
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
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")

    return response


# 全局异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误"""
    logger.warning(f"Request validation error: {exc}")

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
    logger.warning(f"Pydantic validation error: {exc}")

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
    logger.error(f"Global exception: {str(exc)}")
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
        logger.info(f"Using requested model: {request_model}")
        return request_model

    # 否则使用默认模型
    default_model = db.get_config('default_model_name', 'gemini-2.5-flash')
    logger.info(f"Unsupported model: {request_model}, using default: {default_model}")
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


async def select_gemini_key_and_check_limits(model_name: str, excluded_keys: set = None) -> Optional[Dict]:
    """自适应选择可用的Gemini Key并检查模型限制（支持排除特定key）"""
    if excluded_keys is None:
        excluded_keys = set()

    available_keys = db.get_available_gemini_keys()

    # 过滤排除的key
    available_keys = [k for k in available_keys if k['id'] not in excluded_keys]

    if not available_keys:
        logger.warning("No available Gemini keys found after exclusions")
        return None

    # 获取模型配置（已经包含计算的总限制）
    model_config = db.get_model_config(model_name)
    if not model_config:
        logger.error(f"Model config not found for: {model_name}")
        return None

    # 记录限制信息
    logger.info(
        f"Model {model_name} limits: RPM={model_config['total_rpm_limit']}, TPM={model_config['total_tpm_limit']}, RPD={model_config['total_rpd_limit']}")
    logger.info(f"Available API keys: {len(available_keys)}")

    # 检查模型级别的限制（使用总限制）
    current_usage = await rate_limiter.get_current_usage(model_name)

    if (current_usage['requests'] >= model_config['total_rpm_limit'] or
            current_usage['tokens'] >= model_config['total_tpm_limit']):
        logger.warning(
            f"Model {model_name} has reached rate limits: requests={current_usage['requests']}/{model_config['total_rpm_limit']}, tokens={current_usage['tokens']}/{model_config['total_tpm_limit']}")
        return None

    # 检查RPD限制（使用总限制）
    day_usage = db.get_usage_stats(model_name, 'day')
    if day_usage['requests'] >= model_config['total_rpd_limit']:
        logger.warning(
            f"Model {model_name} has reached daily request limit: {day_usage['requests']}/{model_config['total_rpd_limit']}")
        return None

    # 自适应策略选择Key
    strategy = db.get_config('load_balance_strategy', 'adaptive')

    if strategy == 'round_robin':
        # 简单轮询，选择第一个可用的
        selected_key = available_keys[0]
    elif strategy == 'least_used':
        # 选择使用量最少的Key（基于Key ID的使用分布）
        selected_key = available_keys[0]
    else:  # adaptive strategy
        # 自适应策略：综合考虑成功率、响应时间、使用率
        best_key = None
        best_score = -1

        for key_info in available_keys:
            # 计算综合得分
            success_rate = key_info.get('success_rate', 1.0)
            avg_response_time = key_info.get('avg_response_time', 0.0)

            # 响应时间得分（越低越好，转换为0-1分数）
            time_score = max(0, 1.0 - (avg_response_time / 10.0))  # 假设10秒为最差响应时间

            # 综合得分：成功率权重0.7，响应时间权重0.3
            score = success_rate * 0.7 + time_score * 0.3

            if score > best_score:
                best_score = score
                best_key = key_info

        selected_key = best_key if best_key else available_keys[0]

    logger.info(f"Selected API key #{selected_key['id']} for model {model_name} (strategy: {strategy})")

    # 返回选中的Key和模型配置
    return {
        'key_info': selected_key,
        'model_config': model_config
    }


async def make_gemini_request_with_retry(
        gemini_key: str,
        key_id: int,
        gemini_request: Dict,
        model_name: str,
        max_retries: int = 3
) -> Dict:
    """带重试的Gemini API请求，记录性能指标"""
    timeout = float(db.get_config('request_timeout', '60'))

    for attempt in range(max_retries):
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

                response = await client.post(
                    gemini_url,
                    json=gemini_request,
                    headers={"x-goog-api-key": gemini_key}
                )

                response_time = time.time() - start_time

                if response.status_code == 200:
                    # 记录成功的性能指标
                    db.update_key_performance(key_id, True, response_time)
                    return response.json()
                else:
                    # 记录失败的性能指标
                    db.update_key_performance(key_id, False, response_time)
                    error_detail = response.json() if response.content else {"error": {"message": "Unknown error"}}
                    if attempt == max_retries - 1:  # 最后一次尝试
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=error_detail.get("error", {}).get("message", "Unknown error")
                        )
                    else:
                        logger.warning(f"Request failed (attempt {attempt + 1}), retrying...")
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                        continue

        except httpx.TimeoutException as e:
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt == max_retries - 1:
                raise HTTPException(status_code=504, detail="Request timeout")
            else:
                logger.warning(f"Request timeout (attempt {attempt + 1}), retrying...")
                await asyncio.sleep(2 ** attempt)
                continue
        except Exception as e:
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))
            else:
                logger.warning(f"Request failed (attempt {attempt + 1}): {str(e)}, retrying...")
                await asyncio.sleep(2 ** attempt)
                continue

    raise HTTPException(status_code=500, detail="Max retries exceeded")


# 请求处理函数
async def make_request_with_failover(
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        max_key_attempts: int = None,
        excluded_keys: set = None
) -> Dict:
    """
    请求处理

    Args:
        gemini_request: 转换后的Gemini请求
        openai_request: 原始OpenAI请求
        model_name: 模型名称
        max_key_attempts: 最大尝试key数量，默认为所有可用key
        excluded_keys: 排除的key ID集合

    Returns:
        成功的Gemini响应

    Raises:
        HTTPException: 所有key都失败时抛出
    """
    if excluded_keys is None:
        excluded_keys = set()

    # 获取所有可用的key
    available_keys = db.get_available_gemini_keys()

    # 过滤排除的key
    available_keys = [k for k in available_keys if k['id'] not in excluded_keys]

    if not available_keys:
        logger.error("No available keys for failover")
        raise HTTPException(
            status_code=503,
            detail="No available API keys"
        )

    # 如果没有指定最大尝试次数，就尝试所有可用key
    if max_key_attempts is None:
        max_key_attempts = len(available_keys)
    else:
        max_key_attempts = min(max_key_attempts, len(available_keys))

    logger.info(f"Starting failover with {max_key_attempts} key attempts for model {model_name}")

    last_error = None
    failed_keys = []

    for attempt in range(max_key_attempts):
        try:
            # 重新选择可用的key（排除已失败的）
            selection_result = await select_gemini_key_and_check_limits(
                model_name,
                excluded_keys=excluded_keys.union(set(failed_keys))
            )

            if not selection_result:
                logger.warning(f"No more available keys after {attempt} attempts")
                break

            key_info = selection_result['key_info']
            model_config = selection_result['model_config']

            logger.info(f"Attempt {attempt + 1}: Using key #{key_info['id']} for {model_name}")

            try:
                # 尝试使用当前key发送请求（包含单key重试）
                response = await make_gemini_request_with_retry(
                    key_info['key'],
                    key_info['id'],
                    gemini_request,
                    model_name,
                    max_retries=2  # 每个key内部重试2次
                )

                # 成功！记录使用统计
                logger.info(f"✅ Request successful with key #{key_info['id']} on attempt {attempt + 1}")

                # 估算token使用量并记录
                total_tokens = 0
                for candidate in response.get("candidates", []):
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    for part in parts:
                        if "text" in part:
                            total_tokens += len(part["text"].split())

                # 记录成功的使用统计
                await rate_limiter.add_usage(model_name, 1, total_tokens)

                return response

            except HTTPException as e:
                # 记录失败的key
                failed_keys.append(key_info['id'])
                last_error = e

                # 更新key性能统计（失败）
                db.update_key_performance(key_info['id'], False, 0.0)

                # 记录失败统计
                await rate_limiter.add_usage(model_name, 1, 0)

                logger.warning(f"❌ Key #{key_info['id']} failed with {e.status_code}: {e.detail}")

                # 如果是客户端错误（4xx），可能是请求问题，不再尝试其他key
                if e.status_code < 500:
                    logger.warning(f"Client error {e.status_code}, stopping failover")
                    raise e

                # 服务器错误或超时，继续尝试下一个key
                continue

        except Exception as e:
            logger.error(f"Unexpected error during failover attempt {attempt + 1}: {str(e)}")
            last_error = HTTPException(status_code=500, detail=str(e))
            continue

    # 所有key都尝试失败了
    failed_count = len(failed_keys)
    logger.error(f"❌ All {failed_count} keys failed for {model_name}")

    if last_error:
        raise last_error
    else:
        raise HTTPException(
            status_code=503,
            detail=f"All {failed_count} available API keys failed"
        )


# 流式响应处理函数
async def stream_with_failover(
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        model_name: str,
        max_key_attempts: int = None,
        excluded_keys: set = None
) -> AsyncGenerator[bytes, None]:
    """
    流式响应处理
    """
    if excluded_keys is None:
        excluded_keys = set()

    available_keys = db.get_available_gemini_keys()
    available_keys = [k for k in available_keys if k['id'] not in excluded_keys]

    if not available_keys:
        error_data = {
            'error': {
                'message': 'No available API keys',
                'type': 'service_unavailable',
                'code': 503
            }
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
        yield "data: [DONE]\n\n".encode('utf-8')
        return

    if max_key_attempts is None:
        max_key_attempts = len(available_keys)
    else:
        max_key_attempts = min(max_key_attempts, len(available_keys))

    logger.info(f"Starting stream failover with {max_key_attempts} key attempts for {model_name}")

    failed_keys = []

    for attempt in range(max_key_attempts):
        try:
            selection_result = await select_gemini_key_and_check_limits(
                model_name,
                excluded_keys=excluded_keys.union(set(failed_keys))
            )

            if not selection_result:
                break

            key_info = selection_result['key_info']
            logger.info(f"Stream attempt {attempt + 1}: Using key #{key_info['id']}")

            # 尝试流式响应
            success = False
            try:
                async for chunk in stream_gemini_response(
                        key_info['key'],
                        key_info['id'],
                        gemini_request,
                        openai_request,
                        key_info,
                        model_name
                ):
                    yield chunk
                    success = True

                # 如果成功开始流式传输，就不再尝试其他key
                if success:
                    return

            except Exception as e:
                failed_keys.append(key_info['id'])
                logger.warning(f"Stream key #{key_info['id']} failed: {str(e)}")

                # 更新失败统计
                db.update_key_performance(key_info['id'], False, 0.0)

                # 如果还有其他key可以尝试，继续
                if attempt < max_key_attempts - 1:
                    # 发送重试提示（可选）
                    retry_msg = {
                        'error': {
                            'message': f'Key #{key_info["id"]} failed, trying next key...',
                            'type': 'retry_info',
                            'retry_attempt': attempt + 1
                        }
                    }
                    yield f"data: {json.dumps(retry_msg, ensure_ascii=False)}\n\n".encode('utf-8')
                    continue
                else:
                    # 最后一次尝试失败
                    break

        except Exception as e:
            logger.error(f"Stream failover error on attempt {attempt + 1}: {str(e)}")
            continue

    # 所有key都失败了
    error_data = {
        'error': {
            'message': f'All {len(failed_keys)} available API keys failed',
            'type': 'all_keys_failed',
            'code': 503,
            'failed_keys': failed_keys
        }
    }
    yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n".encode('utf-8')
    yield "data: [DONE]\n\n".encode('utf-8')


async def stream_gemini_response(
        gemini_key: str,
        key_id: int,
        gemini_request: Dict,
        openai_request: ChatCompletionRequest,
        key_info: Dict,
        model_name: str
) -> AsyncGenerator[bytes, None]:
    """处理Gemini的流式响应，记录性能指标"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
    timeout = float(db.get_config('request_timeout', '60'))
    max_retries = int(db.get_config('max_retries', '3'))

    logger.info(f"Starting stream request to: {url}")

    start_time = time.time()

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
                        # 记录失败
                        response_time = time.time() - start_time
                        db.update_key_performance(key_id, False, response_time)

                        error_text = await response.aread()
                        error_msg = error_text.decode() if error_text else "Unknown error"
                        logger.error(f"Stream request failed with status {response.status_code}: {error_msg}")
                        yield f"data: {json.dumps({'error': {'message': error_msg, 'type': 'api_error', 'code': response.status_code}}, ensure_ascii=False)}\n\n".encode(
                            'utf-8')
                        yield "data: [DONE]\n\n".encode('utf-8')
                        return

                    stream_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                    created = int(time.time())
                    total_tokens = 0
                    thinking_sent = False
                    has_content = False
                    processed_lines = 0

                    logger.info(f"Stream response started, status: {response.status_code}")

                    try:
                        # 按照官方文档的SSE格式处理
                        async for line in response.aiter_lines():
                            processed_lines += 1

                            if not line:
                                continue

                            # 记录原始行以便调试
                            if processed_lines <= 5:  # 只记录前几行
                                logger.debug(f"Stream line {processed_lines}: {line[:100]}...")

                            if line.startswith("data: "):
                                json_str = line[6:]

                                # 检查结束标志
                                if json_str.strip() == "[DONE]":
                                    logger.info("Received [DONE] signal from stream")
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
                                                            "delta": {"content": "**Thinking Process:**\n"},
                                                            "finish_reason": None
                                                        }]
                                                    }
                                                    yield f"data: {json.dumps(thinking_header, ensure_ascii=False)}\n\n".encode(
                                                        'utf-8')
                                                    thinking_sent = True
                                                    logger.debug("Sent thinking header")
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
                                                    yield f"data: {json.dumps(response_header, ensure_ascii=False)}\n\n".encode(
                                                        'utf-8')
                                                    thinking_sent = False
                                                    logger.debug("Sent response header")

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
                                                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode(
                                                    'utf-8')

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
                                            yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode(
                                                'utf-8')
                                            yield "data: [DONE]\n\n".encode('utf-8')

                                            logger.info(
                                                f"Stream completed with finish_reason: {finish_reason}, tokens: {total_tokens}")

                                            # 记录成功的使用量和性能指标
                                            response_time = time.time() - start_time
                                            db.update_key_performance(key_id, True, response_time)
                                            await rate_limiter.add_usage(model_name, 1, total_tokens)
                                            return

                                except json.JSONDecodeError as e:
                                    logger.warning(f"JSON decode error: {e}, line: {json_str[:200]}...")
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
                            yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode('utf-8')
                            yield "data: [DONE]\n\n".encode('utf-8')

                            logger.info(
                                f"Stream ended naturally, processed {processed_lines} lines, tokens: {total_tokens}")

                            # 记录成功的性能指标
                            response_time = time.time() - start_time
                            db.update_key_performance(key_id, True, response_time)

                        # 如果确实没有收到任何内容，才回退
                        if not has_content:
                            logger.warning(
                                f"Stream response had no content after processing {processed_lines} lines, falling back to non-stream")
                            try:
                                fallback_response = await make_gemini_request_with_retry(
                                    gemini_key, key_id, gemini_request, model_name, 1
                                )

                                thoughts, content = extract_thoughts_and_content(fallback_response)

                                if thoughts and openai_request.thinking_config and openai_request.thinking_config.include_thoughts:
                                    full_content = f"**Thinking Process:**\n{thoughts}\n\n**Response:**\n{content}"
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
                                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n".encode('utf-8')

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
                                    yield f"data: {json.dumps(finish_chunk, ensure_ascii=False)}\n\n".encode('utf-8')
                                    total_tokens = len(full_content.split())

                                    logger.info(f"Fallback completed, tokens: {total_tokens}")

                            except Exception as e:
                                logger.error(f"Fallback request failed: {e}")
                                response_time = time.time() - start_time
                                db.update_key_performance(key_id, False, response_time)
                                yield f"data: {json.dumps({'error': {'message': 'Failed to get response', 'type': 'server_error'}}, ensure_ascii=False)}\n\n".encode(
                                    'utf-8')

                        # 记录使用量
                        await rate_limiter.add_usage(model_name, 1, total_tokens)
                        yield "data: [DONE]\n\n".encode('utf-8')
                        return  # 成功完成

                    except (httpx.ReadError, httpx.RemoteProtocolError) as e:
                        logger.warning(f"Stream connection error (attempt {attempt + 1}): {str(e)}")
                        response_time = time.time() - start_time
                        db.update_key_performance(key_id, False, response_time)
                        if attempt < max_retries - 1:
                            yield f"data: {json.dumps({'error': {'message': 'Connection interrupted, retrying...', 'type': 'connection_error'}}, ensure_ascii=False)}\n\n".encode(
                                'utf-8')
                            await asyncio.sleep(1)
                            continue
                        else:
                            yield f"data: {json.dumps({'error': {'message': 'Stream connection failed after retries', 'type': 'connection_error'}}, ensure_ascii=False)}\n\n".encode(
                                'utf-8')
                            yield "data: [DONE]\n\n".encode('utf-8')
                            return

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning(f"Connection error (attempt {attempt + 1}): {str(e)}")
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt < max_retries - 1:
                yield f"data: {json.dumps({'error': {'message': f'Connection error, retrying... (attempt {attempt + 1})', 'type': 'connection_error'}}, ensure_ascii=False)}\n\n".encode(
                    'utf-8')
                await asyncio.sleep(2 ** attempt)
                continue
            else:
                yield f"data: {json.dumps({'error': {'message': 'Connection failed after all retries', 'type': 'connection_error'}}, ensure_ascii=False)}\n\n".encode(
                    'utf-8')
                yield "data: [DONE]\n\n".encode('utf-8')
                return
        except Exception as e:
            logger.error(f"Unexpected error in stream (attempt {attempt + 1}): {str(e)}")
            response_time = time.time() - start_time
            db.update_key_performance(key_id, False, response_time)
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            else:
                yield f"data: {json.dumps({'error': {'message': 'Unexpected error occurred', 'type': 'server_error'}}, ensure_ascii=False)}\n\n".encode(
                    'utf-8')
                yield "data: [DONE]\n\n".encode('utf-8')
                return


# API端点
@app.get("/")
async def root():
    """根端点"""
    return {
        "service": "Gemini API Proxy",
        "status": "running",
        "version": "1.1.0",
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
        "version": "1.1.0"
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
        "version": "1.1.0",
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


@app.get("/v1")
async def api_v1_info():
    """v1 API 信息端点 - 提供 API 版本信息和可用端点"""
    available_keys = len(db.get_available_gemini_keys())
    supported_models = db.get_supported_models()
    thinking_config = db.get_thinking_config()

    # 获取当前服务的基础URL
    render_url = os.getenv('RENDER_EXTERNAL_URL')
    base_url = render_url if render_url else 'https://your-service.onrender.com'

    return {
        "service": "Gemini API Proxy",
        "version": "1.1.0",
        "api_version": "v1",
        "compatibility": "OpenAI API v1",
        "description": "A high-performance proxy for Gemini API with OpenAI compatibility and multi-key polling",
        "status": "operational",
        "base_url": base_url,
        "features": [
            "Multi-key polling & load balancing",
            "OpenAI API compatibility",
            "Rate limiting & usage analytics",
            "Thinking mode support",
            "Streaming responses",
            "Automatic failover",
            "Real-time monitoring",
            "Health checking",
            "Adaptive load balancing"
        ],
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "api_info": "/v1",
            "health": "/health",
            "status": "/status",
            "admin": "/admin/*",
            "docs": "/docs"
        },
        "supported_models": supported_models,
        "service_status": {
            "active_gemini_keys": available_keys,
            "thinking_enabled": thinking_config.get('enabled', False),
            "thinking_budget": thinking_config.get('budget', -1),
            "uptime_seconds": int(time.time() - start_time),
            "total_requests": request_count
        },
        "documentation": {
            "openapi_json": "/openapi.json",
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "github": "https://github.com/arain119/gemini-api-proxy"
        },
        "example_usage": {
            "curl": f"curl -X POST '{base_url}/v1/chat/completions' \\\n  -H 'Authorization: Bearer YOUR_API_KEY' \\\n  -H 'Content-Type: application/json' \\\n  -d '{{\n    \"model\": \"gemini-2.5-flash\",\n    \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]\n  }}'",
            "python": f"import openai\n\nclient = openai.OpenAI(\n    api_key='YOUR_API_KEY',\n    base_url='{base_url}/v1'\n)\n\nresponse = client.chat.completions.create(\n    model='gemini-2.5-flash',\n    messages=[{{'role': 'user', 'content': 'Hello!'}}]\n)",
            "javascript": f"import OpenAI from 'openai';\n\nconst openai = new OpenAI({{\n  apiKey: 'YOUR_API_KEY',\n  baseURL: '{base_url}/v1'\n}});\n\nconst response = await openai.chat.completions.create({{\n  model: 'gemini-2.5-flash',\n  messages: [{{ role: 'user', content: 'Hello!' }}]\n}});"
        },
        "rate_limits": {
            "info": "Limits scale with number of healthy Gemini API keys",
            "check_admin": "/admin/models for current limits"
        },
        "timestamp": datetime.now().isoformat()
    }


# chat_completions端点
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

        # 转换请求格式
        gemini_request = openai_to_gemini(request)

        # 故障转移机制
        if request.stream:
            # 流式响应使用故障转移
            return StreamingResponse(
                stream_with_failover(
                    gemini_request,
                    request,
                    actual_model_name,
                    max_key_attempts=5  # 最多尝试5个key
                ),
                media_type="text/event-stream; charset=utf-8"
            )
        else:
            # 非流式响应使用故障转移
            gemini_response = await make_request_with_failover(
                gemini_request,
                request,
                actual_model_name,
                max_key_attempts=5  # 最多尝试5个key
            )

            # 计算token使用量
            total_tokens = 0
            for candidate in gemini_response.get("candidates", []):
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        total_tokens += len(part["text"].split())

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
        logger.error(f"Unexpected error: {str(e)}")
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


# 健康检测相关端点
@app.post("/admin/health/check-all")
async def check_all_keys_health():
    """一键检测所有Gemini Keys的健康状态"""
    try:
        all_keys = db.get_all_gemini_keys()
        active_keys = [key for key in all_keys if key['status'] == 1]

        if not active_keys:
            return {
                "success": True,
                "message": "No active keys to check",
                "results": []
            }

        results = []
        healthy_count = 0

        # 并发检测所有keys
        tasks = []
        for key_info in active_keys:
            task = check_gemini_key_health(key_info['key'])
            tasks.append((key_info['id'], task))

        # 等待所有检测完成
        for key_id, task in tasks:
            health_result = await task

            # 更新数据库中的健康状态
            db.update_key_performance(
                key_id,
                health_result['healthy'],
                health_result['response_time']
            )

            if health_result['healthy']:
                healthy_count += 1

            results.append({
                "key_id": key_id,
                "healthy": health_result['healthy'],
                "response_time": health_result['response_time'],
                "error": health_result['error']
            })

        return {
            "success": True,
            "message": f"Health check completed: {healthy_count}/{len(active_keys)} keys healthy",
            "total_checked": len(active_keys),
            "healthy_count": healthy_count,
            "unhealthy_count": len(active_keys) - healthy_count,
            "results": results
        }

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/health/summary")
async def get_health_summary():
    """获取健康状态汇总"""
    try:
        summary = db.get_keys_health_summary()
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Failed to get health summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 密钥管理端点
@app.get("/admin/keys/gemini")
async def get_gemini_keys():
    """获取所有Gemini密钥列表"""
    try:
        keys = db.get_all_gemini_keys()
        return {
            "success": True,
            "keys": keys
        }
    except Exception as e:
        logger.error(f"Failed to get Gemini keys: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/keys/user")
async def get_user_keys():
    """获取所有用户密钥列表"""
    try:
        keys = db.get_all_user_keys()
        return {
            "success": True,
            "keys": keys
        }
    except Exception as e:
        logger.error(f"Failed to get user keys: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/keys/gemini/{key_id}")
async def delete_gemini_key(key_id: int):
    """删除指定的Gemini密钥"""
    try:
        success = db.delete_gemini_key(key_id)
        if success:
            logger.info(f"Deleted Gemini key #{key_id}")
            return {
                "success": True,
                "message": f"Gemini key #{key_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Key not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete Gemini key #{key_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/keys/user/{key_id}")
async def delete_user_key(key_id: int):
    """删除指定的用户密钥"""
    try:
        success = db.delete_user_key(key_id)
        if success:
            logger.info(f"Deleted user key #{key_id}")
            return {
                "success": True,
                "message": f"User key #{key_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Key not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user key #{key_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/keys/gemini/{key_id}/toggle")
async def toggle_gemini_key_status(key_id: int):
    """切换Gemini密钥状态"""
    try:
        success = db.toggle_gemini_key_status(key_id)
        if success:
            logger.info(f"Toggled Gemini key #{key_id} status")
            return {
                "success": True,
                "message": f"Gemini key #{key_id} status toggled successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Key not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle Gemini key #{key_id} status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/keys/user/{key_id}/toggle")
async def toggle_user_key_status(key_id: int):
    """切换用户密钥状态"""
    try:
        success = db.toggle_user_key_status(key_id)
        if success:
            logger.info(f"Toggled user key #{key_id} status")
            return {
                "success": True,
                "message": f"User key #{key_id} status toggled successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Key not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle user key #{key_id} status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# 管理端点
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
        logger.error(f"Failed to get model config for {model_name}: {str(e)}")
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
            logger.info(f"Updated model config for {model_name}: {update_data}")
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
        logger.error(f"Failed to update model config for {model_name}: {str(e)}")
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
        logger.error(f"Failed to get model configs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/config/gemini-key")
async def add_gemini_key(request: dict):
    """通过API添加Gemini密钥，支持批量添加"""
    input_keys = request.get("key", "").strip()

    if not input_keys:
        return {"success": False, "message": "请提供API密钥"}

    # 检测是否包含分隔符，支持批量添加
    separators = [',', ';', '\n', '\r\n', '\r', '\t']  # 逗号、分号、换行符、制表符
    has_separator = any(sep in input_keys for sep in separators)

    # 如果包含分隔符或多个空格，进行分割
    if has_separator or '  ' in input_keys:  # 两个或更多空格也视为分隔符
        # 首先按换行符分割
        lines = input_keys.replace('\r\n', '\n').replace('\r', '\n').split('\n')

        keys_to_add = []
        for line in lines:
            # 再按其他分隔符分割每一行
            line_keys = []
            for sep in [',', ';', '\t']:
                if sep in line:
                    line_keys.extend([k.strip() for k in line.split(sep)])
                    break
            else:
                # 如果没有找到分隔符，检查是否有多个空格
                if '  ' in line:  # 多个空格
                    line_keys.extend([k.strip() for k in line.split()])
                else:
                    line_keys.append(line.strip())

            keys_to_add.extend(line_keys)

        # 清理空字符串
        keys_to_add = [key for key in keys_to_add if key]

        logger.info(f"检测到批量添加模式，将添加 {len(keys_to_add)} 个密钥")

    else:
        # 单个密钥
        keys_to_add = [input_keys]

    # 验证和添加密钥
    results = {
        "success": True,
        "total_processed": len(keys_to_add),
        "successful_adds": 0,
        "failed_adds": 0,
        "details": [],
        "invalid_keys": [],
        "duplicate_keys": []
    }

    for i, key in enumerate(keys_to_add, 1):
        key = key.strip()

        # 验证密钥格式
        if not key:
            continue

        if not key.startswith('AIzaSy'):
            results["invalid_keys"].append(f"#{i}: {key[:20]}... (不是有效的Gemini API密钥格式)")
            results["failed_adds"] += 1
            continue

        if len(key) < 30 or len(key) > 50:  # Gemini API Key 长度通常在35-40字符
            results["invalid_keys"].append(f"#{i}: {key[:20]}... (密钥长度异常)")
            results["failed_adds"] += 1
            continue

        # 尝试添加到数据库
        try:
            if db.add_gemini_key(key):
                results["successful_adds"] += 1
                results["details"].append(f"✅ #{i}: {key[:10]}...{key[-4:]} 添加成功")
                logger.info(f"成功添加Gemini密钥 #{i}")
            else:
                results["duplicate_keys"].append(f"#{i}: {key[:10]}...{key[-4:]} (密钥已存在)")
                results["failed_adds"] += 1
        except Exception as e:
            results["failed_adds"] += 1
            results["details"].append(f"❌ #{i}: {key[:10]}...{key[-4:]} 添加失败 - {str(e)}")
            logger.error(f"添加Gemini密钥 #{i} 失败: {str(e)}")

    # 生成返回消息
    if results["successful_adds"] > 0:
        message_parts = [f"成功添加 {results['successful_adds']} 个密钥"]

        if results["failed_adds"] > 0:
            message_parts.append(f"失败 {results['failed_adds']} 个")

        results["message"] = "、".join(message_parts)

        # 如果有部分成功，整体仍视为成功
        results["success"] = True
    else:
        results["success"] = False
        results["message"] = f"所有 {results['total_processed']} 个密钥添加失败"

    # 详细日志
    logger.info(
        f"批量添加结果: 处理{results['total_processed']}个，成功{results['successful_adds']}个，失败{results['failed_adds']}个")

    return results


@app.post("/admin/config/user-key")
async def generate_user_key(request: dict):
    """生成用户密钥"""
    name = request.get("name", "API User")
    key = db.generate_user_key(name)
    logger.info(f"Generated new user key for: {name}")
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
            logger.info(f"Updated thinking config: {request}")
            return {
                "success": True,
                "message": "Thinking configuration updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update thinking configuration")

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update thinking config: {str(e)}")
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
            logger.info(f"Updated inject prompt config: enabled={enabled}, position={position}")
            return {
                "success": True,
                "message": "Inject prompt configuration updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update inject prompt configuration")

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update inject prompt config: {str(e)}")
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
        logger.error(f"Failed to get configs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/stats")
async def get_admin_stats():
    """获取管理统计"""
    health_summary = db.get_keys_health_summary()

    return {
        "gemini_keys": len(db.get_all_gemini_keys()),
        "active_gemini_keys": len(db.get_available_gemini_keys()),
        "healthy_gemini_keys": health_summary['healthy'],
        "user_keys": len(db.get_all_user_keys()),
        "active_user_keys": len([k for k in db.get_all_user_keys() if k['status'] == 1]),
        "supported_models": db.get_supported_models(),
        "usage_stats": db.get_all_usage_stats(),
        "thinking_config": db.get_thinking_config(),
        "inject_config": db.get_inject_prompt_config(),
        "health_summary": health_summary
    }


# 运行服务器的函数
def run_api_server(port: int = 8000):
    """运行API服务器"""
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting Gemini API Proxy on port {port}")
    run_api_server(port)