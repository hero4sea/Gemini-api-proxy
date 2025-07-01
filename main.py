import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import os
import time
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import schedule

# --- 页面配置 ---
st.set_page_config(
    page_title="Gemini API Proxy",
    page_icon="🌠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- API配置 ---
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

if 'streamlit.io' in os.getenv('STREAMLIT_SERVER_HEADLESS', ''):
    API_BASE_URL = os.getenv('API_BASE_URL', 'https://your-app.onrender.com')


# --- 保活机制 ---
class KeepAliveManager:
    def __init__(self):
        self.scheduler_thread = None
        self.is_running = False
        self.render_url = os.getenv('RENDER_EXTERNAL_URL')
        self.backend_url = API_BASE_URL

    def keep_alive_backend(self):
        """保活后端API服务"""
        try:
            response = requests.get(f"{self.backend_url}/wake", timeout=10)
            if response.status_code == 200:
                logger.info("Backend keep-alive ping sent successfully")
                return True
        except Exception as e:
            logger.warning(f"Backend keep-alive ping failed: {e}")
            return False

    def keep_alive_frontend(self):
        """保活前端服务（如果在Render环境）"""
        if not self.render_url:
            return True

        try:
            # 向自己发送请求保活
            response = requests.get(f"{self.render_url}/", timeout=10)
            if response.status_code == 200:
                logger.info("Frontend keep-alive ping sent successfully")
                return True
        except Exception as e:
            logger.warning(f"Frontend keep-alive ping failed: {e}")
            return False

    def combined_keep_alive_task(self):
        """组合保活任务"""
        logger.info("Executing keep-alive tasks...")

        # 保活后端
        backend_success = self.keep_alive_backend()

        # 保活前端（仅在Render环境）
        frontend_success = True
        if self.render_url:
            frontend_success = self.keep_alive_frontend()

        # 记录结果
        if backend_success and frontend_success:
            logger.info("Keep-alive tasks completed successfully")
        else:
            logger.warning(f"Keep-alive partial failure - Backend: {backend_success}, Frontend: {frontend_success}")

    def run_scheduler_loop(self):
        """调度器循环（运行在后台线程）"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(60)

    def start_keep_alive_scheduler(self):
        """启动保活调度器"""
        # 检测是否为Render环境或需要保活的环境
        need_keepalive = (
                self.render_url or  # Render环境
                'streamlit.io' in os.getenv('STREAMLIT_SERVER_HEADLESS', '') or  # Streamlit Cloud
                os.getenv('ENABLE_KEEPALIVE', '').lower() == 'true'  # 手动启用
        )

        if not need_keepalive:
            logger.info("Keep-alive not needed in current environment")
            return False

        if self.is_running:
            logger.warning("Keep-alive scheduler already running")
            return False

        try:
            # 设置每14分钟执行一次（在15分钟睡眠前保持唤醒）
            schedule.every(14).minutes.do(self.combined_keep_alive_task)

            # 立即执行一次
            self.combined_keep_alive_task()

            # 启动后台线程
            self.is_running = True
            self.scheduler_thread = threading.Thread(
                target=self.run_scheduler_loop,
                daemon=True,
                name="KeepAliveScheduler"
            )
            self.scheduler_thread.start()

            logger.info("Keep-alive scheduler started (14min interval)")

            # 记录环境信息
            if self.render_url:
                logger.info(f"Render URL detected: {self.render_url}")
            logger.info(f"Backend URL: {self.backend_url}")

            return True

        except Exception as e:
            logger.error(f"Failed to start keep-alive scheduler: {e}")
            self.is_running = False
            return False

    def stop_scheduler(self):
        """停止调度器"""
        if self.is_running:
            self.is_running = False
            schedule.clear()  # 清除所有定时任务
            logger.info("Keep-alive scheduler stopped")

    def get_status(self):
        """获取保活状态"""
        return {
            'running': self.is_running,
            'render_url': self.render_url,
            'backend_url': self.backend_url,
            'thread_alive': self.scheduler_thread.is_alive() if self.scheduler_thread else False,
            'scheduled_jobs': len(schedule.jobs)
        }


# 全局保活管理器
if 'keep_alive_manager' not in st.session_state:
    st.session_state.keep_alive_manager = KeepAliveManager()

# 启动保活机制（只启动一次）
if 'keep_alive_started' not in st.session_state:
    st.session_state.keep_alive_started = True
    success = st.session_state.keep_alive_manager.start_keep_alive_scheduler()
    if success:
        logger.info("Keep-alive system initialized")
    else:
        logger.info("Keep-alive system not started (not needed or failed)")


# --- API调用函数 ---
def call_api(endpoint: str, method: str = 'GET', data: Any = None, timeout: int = 30) -> Optional[Dict]:
    """统一API调用函数"""
    url = f"{API_BASE_URL}{endpoint}"

    try:
        spinner_message = "加载中..." if method == 'GET' else "保存中..."
        with st.spinner(spinner_message):
            if method == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, timeout=timeout)
            else:
                raise ValueError(f"不支持的方法: {method}")

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API错误: {response.status_code}")
                return None

    except requests.exceptions.Timeout:
        st.error("请求超时，请重试。")
        return None
    except requests.exceptions.ConnectionError:
        st.error("无法连接到API服务。")
        return None
    except Exception as e:
        st.error(f"API错误: {str(e)}")
        return None


def wake_up_service():
    """唤醒服务"""
    try:
        response = requests.get(f"{API_BASE_URL}/wake", timeout=10)
        if response.status_code == 200:
            st.success("服务已激活")
            return True
    except:
        pass
    return False


def check_service_health():
    """检查服务健康状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


# --- 健康检测函数 ---
def check_all_keys_health():
    """一键检测所有Key健康状态"""
    result = call_api('/admin/health/check-all', 'POST', timeout=60)
    return result


def get_health_summary():
    """获取健康状态汇总"""
    result = call_api('/admin/health/summary')
    return result


# --- 缓存函数 ---
@st.cache_data(ttl=30)
def get_cached_stats():
    """获取缓存的统计数据"""
    return call_api('/admin/stats')


@st.cache_data(ttl=60)
def get_cached_status():
    """获取缓存的服务状态"""
    return call_api('/status')


@st.cache_data(ttl=30)
def get_cached_model_config(model_name: str):
    """获取缓存的模型配置"""
    return call_api(f'/admin/models/{model_name}')


@st.cache_data(ttl=30)
def get_cached_gemini_keys():
    """获取缓存的Gemini密钥列表"""
    return call_api('/admin/keys/gemini')


@st.cache_data(ttl=30)
def get_cached_user_keys():
    """获取缓存的用户密钥列表"""
    return call_api('/admin/keys/user')


@st.cache_data(ttl=30)
def get_cached_health_summary():
    """获取缓存的健康状态汇总"""
    return get_health_summary()


# --- 密钥管理函数 ---
def mask_key(key: str, show_full: bool = False) -> str:
    """密钥掩码处理"""
    if show_full:
        return key

    if key.startswith('sk-'):
        # 用户密钥格式: sk-xxxxxxxx...
        if len(key) > 10:
            return f"{key[:6]}{'•' * (len(key) - 10)}{key[-4:]}"
        return key
    elif key.startswith('AIzaSy'):
        # Gemini密钥格式: AIzaSyxxxxxxx...
        if len(key) > 12:
            return f"{key[:8]}{'•' * (len(key) - 12)}{key[-4:]}"
        return key
    else:
        # 其他格式
        if len(key) > 8:
            return f"{key[:4]}{'•' * (len(key) - 8)}{key[-4:]}"
        return key


def delete_key(key_type: str, key_id: int) -> bool:
    """删除密钥"""
    endpoint = f'/admin/keys/{key_type}/{key_id}'
    result = call_api(endpoint, 'DELETE')
    return result and result.get('success', False)


def toggle_key_status(key_type: str, key_id: int) -> bool:
    """切换密钥状态"""
    endpoint = f'/admin/keys/{key_type}/{key_id}/toggle'
    result = call_api(endpoint, 'POST')
    return result and result.get('success', False)


def get_health_status_color(health_status: str) -> str:
    """获取健康状态颜色"""
    status_colors = {
        'healthy': '#10b981',  # 绿色
        'unhealthy': '#ef4444',  # 红色
        'unknown': '#f59e0b'  # 黄色
    }
    return status_colors.get(health_status, '#6b7280')  # 默认灰色


def format_health_status(health_status: str) -> str:
    """格式化健康状态显示"""
    status_map = {
        'healthy': '正常',
        'unhealthy': '异常',
        'unknown': '未知'
    }
    return status_map.get(health_status, health_status)


# --- 自定义CSS样式 ---
st.markdown("""
<style>
    /* 全局字体 */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro SC", "SF Pro Display", "Helvetica Neue", "PingFang SC", "Microsoft YaHei UI", sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* 整体布局 */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }

    /* 度量卡片 */
    [data-testid="metric-container"] {
        background: #ffffff;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }

    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-color: #d1d5db;
    }

    /* 度量值样式 */
    [data-testid="metric-container"] > div:nth-child(1) {
        font-size: 0.75rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }

    [data-testid="metric-container"] > div:nth-child(2) {
        font-size: 1.875rem;
        font-weight: 600;
        color: #111827;
        line-height: 1.2;
    }

    [data-testid="metric-container"] > div:nth-child(3) {
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }

    /* 按钮样式 */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.15s ease;
        border: 1px solid transparent;
        font-size: 0.875rem;
        padding: 0.5rem 1rem;
        letter-spacing: 0.01em;
        background: #111827;
        color: white;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    .stButton > button:hover {
        background: #1f2937;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    /* Primary按钮 */
    .stButton > button[type="primary"] {
        background: #6366f1;
        color: white;
    }

    .stButton > button[type="primary"]:hover {
        background: #4f46e5;
    }

    /* Secondary按钮 */
    .stButton > button[type="secondary"] {
        background: #ffffff;
        color: #374151;
        border: 1px solid #e5e7eb;
    }

    .stButton > button[type="secondary"]:hover {
        background: #f9fafb;
        border-color: #d1d5db;
    }

    /* 健康状态标签 */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.625rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
        line-height: 1;
    }

    .status-healthy {
        background: #d1fae5;
        color: #065f46;
    }

    .status-unhealthy {
        background: #fee2e2;
        color: #991b1b;
    }

    .status-unknown {
        background: #fef3c7;
        color: #92400e;
    }

    .status-active {
        background: #dbeafe;
        color: #1e40af;
    }

    .status-inactive {
        background: #f3f4f6;
        color: #6b7280;
    }

    /* 输入框样式 */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        font-size: 0.875rem;
        padding: 0.625rem 0.875rem;
        background-color: #ffffff;
        transition: all 0.15s ease;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        outline: none;
    }

    /* 标签页样式 */
    .stTabs [data-testid="stTabBar"] {
        gap: 2rem;
        border-bottom: 1px solid #e5e7eb;
        padding: 0;
        margin-bottom: 2rem;
    }

    .stTabs [data-testid="stTabBar"] button {
        font-weight: 500;
        color: #6b7280;
        padding-bottom: 1rem;
        border-bottom: 2px solid transparent;
        font-size: 0.875rem;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
    }

    .stTabs [data-testid="stTabBar"] button[aria-selected="true"] {
        color: #111827;
        border-bottom-color: #6366f1;
    }

    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background: #fafbfc;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] > div:nth-child(1) > div:nth-child(2) {
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #374151;
    }

    /* 成功/错误消息样式 */
    .stAlert {
        border-radius: 6px;
        font-size: 0.875rem;
        padding: 0.75rem 1rem;
        border: 1px solid;
    }

    /* 图表容器 */
    .js-plotly-plot .plotly {
        border-radius: 8px;
        overflow: hidden;
    }

    /* 表格样式 */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #e5e7eb;
    }

    /* 分隔线样式 */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e5e7eb;
    }

    /* 标题样式 */
    h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem;
    }

    h2 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #111827;
        letter-spacing: -0.02em;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    h3 {
        font-size: 1.125rem;
        font-weight: 600;
        color: #374151;
        letter-spacing: -0.01em;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    /* 密钥容器样式 */
    .key-container {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        transition: all 0.2s ease;
    }

    .key-container:hover {
        border-color: #d1d5db;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    /* 信息容器 */
    .info-container {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* 性能指标文本 */
    .performance-metric {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }

    /* 代码块样式 */
    .stCodeBlock {
        border-radius: 6px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
    }

    /* Expander样式 */
    .streamlit-expanderHeader {
        font-size: 0.875rem;
        font-weight: 500;
        color: #374151;
        background: #f9fafb;
        border-radius: 6px;
    }

    .streamlit-expanderHeader:hover {
        background: #f3f4f6;
    }

    /* 页面标题副标题 */
    .page-subtitle {
        font-size: 0.875rem;
        color: #6b7280;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* 操作按钮组 */
    .action-buttons {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }

    /* 小按钮样式 */
    .small-button {
        font-size: 0.75rem;
        padding: 0.375rem 0.75rem;
    }

    /* 刷新按钮特殊样式 */
    .refresh-button {
        background: transparent;
        border: 1px solid #e5e7eb;
        color: #6b7280;
        padding: 0.375rem 0.625rem;
        font-size: 1.125rem;
        line-height: 1;
        border-radius: 6px;
    }

    .refresh-button:hover {
        background: #f9fafb;
        border-color: #d1d5db;
        color: #374151;
    }

    /* ==================== 修复 Streamlit 消息组件边框问题 ==================== */

    /* 修复所有 Streamlit Alert 组件的边框 */
    [data-testid="stAlert"] {
        border: none !important;
        box-shadow: none !important;
    }

    /* 修复具体的消息类型 */
    [data-testid="stAlert"][kind="info"] {
        border: none !important;
        background: #dbeafe !important;
        color: #1e40af !important;
    }

    [data-testid="stAlert"][kind="success"] {
        border: none !important;
        background: #d1fae5 !important;
        color: #065f46 !important;
    }

    [data-testid="stAlert"][kind="warning"] {
        border: none !important;
        background: #fef3c7 !important;
        color: #92400e !important;
    }

    [data-testid="stAlert"][kind="error"] {
        border: none !important;
        background: #fee2e2 !important;
        color: #991b1b !important;
    }

    /* 修复 Alert 内部元素 */
    [data-testid="stAlert"] > div {
        border: none !important;
        box-shadow: none !important;
    }

    /* 修复可能的子元素 */
    [data-testid="stAlert"] * {
        border: none !important;
    }

    /* 如果还有其他消息框组件 */
    [data-testid="stSuccess"],
    [data-testid="stInfo"], 
    [data-testid="stWarning"],
    [data-testid="stError"] {
        border: none !important;
        box-shadow: none !important;
    }

    /* 修复任何可能的边框样式 */
    .stAlert, 
    .st-alert,
    div[role="alert"] {
        border: none !important;
        box-shadow: none !important;
    }

    /* 强制移除所有可能的边框 */
    .stAlert *,
    .st-alert *,
    [data-testid="stAlert"] *,
    div[role="alert"] * {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* 修复气泡框边框（针对内联样式） */
    div[style*="background: #dbeafe"],
    div[style*="background: #d1fae5"],
    div[style*="background: #fee2e2"],
    div[style*="background: #fef3c7"],
    div[style*="background: #f3f4f6"] {
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏 ---
with st.sidebar:
    st.markdown("## Gemini 轮询")
    st.markdown('<hr style="margin: 1rem 0;">', unsafe_allow_html=True)

    page = st.radio(
        "导航",
        ["控制台", "模型配置", "密钥管理", "系统设置"],
        label_visibility="collapsed"
    )

    st.markdown('<hr style="margin: 1rem 0;">', unsafe_allow_html=True)

    # 服务状态
    st.markdown("##### 服务状态")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("刷新", use_container_width=True, key="refresh_sidebar"):
            st.cache_data.clear()
    with col2:
        if st.button("唤醒", use_container_width=True, key="wake_sidebar"):
            wake_up_service()

    # 检查服务健康状态
    health = check_service_health()
    if health:
        st.markdown('<div class="status-badge status-healthy">服务正常</div>', unsafe_allow_html=True)
        with st.expander("详细信息", expanded=False):
            st.text(f"地址: {API_BASE_URL}")
            st.text(f"状态: {health.get('status', 'unknown')}")
            st.text(f"运行: {health.get('uptime_seconds', 0) // 3600}小时")
    else:
        st.markdown('<div class="status-badge status-unhealthy">服务离线</div>', unsafe_allow_html=True)
        st.caption("点击'唤醒'按钮激活服务")

    # 系统概览
    st.markdown('<hr style="margin: 1rem 0;">', unsafe_allow_html=True)
    st.markdown("##### 系统概览")

    status_data = get_cached_status()
    if status_data:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("可用密钥", status_data.get('active_keys', 0))
        with col2:
            thinking_enabled = status_data.get('thinking_enabled', False)
            st.metric("思考模式", "开启" if thinking_enabled else "关闭")

    # 健康状态
    health_summary = get_cached_health_summary()
    if health_summary and health_summary.get('success'):
        summary = health_summary['summary']
        if summary.get('unhealthy', 0) > 0:
            st.warning(f"发现 {summary.get('unhealthy', 0)} 个异常密钥")

# --- 主页面内容 ---
if page == "控制台":
    st.title("控制台")
    st.markdown('<p class="page-subtitle">实时监控服务运行状态和使用情况</p>', unsafe_allow_html=True)

    # 刷新按钮
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("⟳", help="刷新数据", key="refresh_dashboard"):
            st.cache_data.clear()
            st.rerun()

    # 获取统计数据
    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("无法获取服务数据，请检查服务连接")
        st.stop()

    # 健康状态提示
    health_summary = stats_data.get('health_summary', {})
    if health_summary:
        col1, col2 = st.columns([4, 1])
        with col1:
            total_active = health_summary.get('total_active', 0)
            healthy_count = health_summary.get('healthy', 0)
            unhealthy_count = health_summary.get('unhealthy', 0)

            if unhealthy_count > 0:
                st.error(f"发现 {unhealthy_count} 个异常密钥，共 {total_active} 个激活密钥")
            elif healthy_count > 0:
                st.success(f"所有 {healthy_count} 个密钥运行正常")
            else:
                st.info("暂无激活的密钥")

        with col2:
            if st.button("健康检测", help="检测所有密钥健康状态", use_container_width=True):
                with st.spinner("正在检测..."):
                    result = check_all_keys_health()
                    if result and result.get('success'):
                        st.success(result['message'])
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()

    # 核心指标
    st.markdown("### 核心指标")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gemini_keys = stats_data.get('active_gemini_keys', 0)
        healthy_gemini = stats_data.get('healthy_gemini_keys', 0)
        st.metric(
            "GEMINI密钥",
            gemini_keys,
            delta=f"{healthy_gemini} 正常"
        )

    with col2:
        user_keys = stats_data.get('active_user_keys', 0)
        total_user = stats_data.get('user_keys', 0)
        st.metric(
            "用户密钥",
            user_keys,
            delta=f"共 {total_user} 个"
        )

    with col3:
        models = stats_data.get('supported_models', [])
        st.metric("支持模型", len(models))

    with col4:
        thinking_status = "启用" if status_data.get('thinking_enabled', False) else "禁用"
        st.metric("思考功能", thinking_status)

    # 使用率分析
    st.markdown("### 使用率分析")

    usage_stats = stats_data.get('usage_stats', {})
    if usage_stats and models:
        # 准备数据
        model_data = []
        for model in models:
            stats = usage_stats.get(model, {'minute': {'requests': 0}, 'day': {'requests': 0}})

            model_config_data = get_cached_model_config(model)
            if not model_config_data:
                rpm_limit = 10 if 'flash' in model else 5
                rpd_limit = 250 if 'flash' in model else 100
            else:
                rpm_limit = model_config_data.get('total_rpm_limit', 10)
                rpd_limit = model_config_data.get('total_rpd_limit', 250)

            rpm_used = stats['minute']['requests']
            rpm_percent = (rpm_used / rpm_limit * 100) if rpm_limit > 0 else 0

            rpd_used = stats['day']['requests']
            rpd_percent = (rpd_used / rpd_limit * 100) if rpd_limit > 0 else 0

            model_data.append({
                'Model': model,
                'RPM Used': rpm_used,
                'RPM Limit': rpm_limit,
                'RPM %': rpm_percent,
                'RPD Used': rpd_used,
                'RPD Limit': rpd_limit,
                'RPD %': rpd_percent
            })

        if model_data:
            df = pd.DataFrame(model_data)

            # 创建图表
            col1, col2 = st.columns(2)

            with col1:
                fig_rpm = go.Figure()
                fig_rpm.add_trace(go.Bar(
                    x=df['Model'],
                    y=df['RPM %'],
                    text=[f"{x:.1f}%" for x in df['RPM %']],
                    textposition='outside',
                    marker_color='#6366f1',
                    hovertemplate='<b>%{x}</b><br>使用率: %{y:.1f}%<br>当前: %{customdata[0]:,}<br>限制: %{customdata[1]:,}<extra></extra>',
                    customdata=df[['RPM Used', 'RPM Limit']].values
                ))
                fig_rpm.update_layout(
                    title="每分钟请求数 (RPM)",
                    title_font=dict(size=14, color='#374151'),
                    yaxis_title="使用率 (%)",
                    yaxis_range=[0, max(100, df['RPM %'].max() * 1.2) if len(df) > 0 else 100],
                    height=320,
                    showlegend=False,
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font=dict(family='-apple-system, BlinkMacSystemFont', color='#6b7280', size=12),
                    yaxis=dict(gridcolor='#f3f4f6', zerolinecolor='#e5e7eb'),
                    xaxis=dict(linecolor='#e5e7eb'),
                    bargap=0.3,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_rpm, use_container_width=True)

            with col2:
                fig_rpd = go.Figure()
                fig_rpd.add_trace(go.Bar(
                    x=df['Model'],
                    y=df['RPD %'],
                    text=[f"{x:.1f}%" for x in df['RPD %']],
                    textposition='outside',
                    marker_color='#10b981',
                    hovertemplate='<b>%{x}</b><br>使用率: %{y:.1f}%<br>当前: %{customdata[0]:,}<br>限制: %{customdata[1]:,}<extra></extra>',
                    customdata=df[['RPD Used', 'RPD Limit']].values
                ))
                fig_rpd.update_layout(
                    title="每日请求数 (RPD)",
                    title_font=dict(size=14, color='#374151'),
                    yaxis_title="使用率 (%)",
                    yaxis_range=[0, max(100, df['RPD %'].max() * 1.2) if len(df) > 0 else 100],
                    height=320,
                    showlegend=False,
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font=dict(family='-apple-system, BlinkMacSystemFont', color='#6b7280', size=12),
                    yaxis=dict(gridcolor='#f3f4f6', zerolinecolor='#e5e7eb'),
                    xaxis=dict(linecolor='#e5e7eb'),
                    bargap=0.3,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig_rpd, use_container_width=True)

            # 详细数据表
            with st.expander("查看详细数据"):
                display_df = df[['Model', 'RPM Used', 'RPM Limit', 'RPM %', 'RPD Used', 'RPD Limit', 'RPD %']].copy()
                display_df.columns = ['模型', '分钟请求', '分钟限制', '分钟使用率', '日请求', '日限制', '日使用率']
                display_df['分钟使用率'] = display_df['分钟使用率'].apply(lambda x: f"{x:.1f}%")
                display_df['日使用率'] = display_df['日使用率'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无使用数据")

elif page == "密钥管理":
    st.title("密钥管理")
    st.markdown('<p class="page-subtitle">管理 Gemini API 密钥和用户访问令牌</p>', unsafe_allow_html=True)

    # 刷新按钮
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("⟳", help="刷新数据", key="refresh_keys"):
            st.cache_data.clear()
            st.rerun()

    tab1, tab2 = st.tabs(["Gemini 密钥", "用户密钥"])

    with tab1:
        st.markdown("#### 添加新密钥")

        with st.form("add_gemini_key"):
            new_key = st.text_input(
                "Gemini API 密钥",
                type="password",
                placeholder="AIzaSy...",
                help="从 Google AI Studio 获取"
            )
            submitted = st.form_submit_button("添加密钥", type="primary")

            if submitted and new_key:
                result = call_api('/admin/config/gemini-key', 'POST', {'key': new_key})
                if result and result.get('success'):
                    st.success("密钥添加成功")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("添加失败，密钥可能已存在")

        st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)

        # 现有密钥
        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.markdown("#### 现有密钥")
        with col2:
            if st.button("健康检测", help="检测所有密钥状态", key="health_check_gemini"):
                with st.spinner("检测中..."):
                    result = check_all_keys_health()
                    if result and result.get('success'):
                        st.success(result['message'])
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
        with col3:
            show_full_keys = st.checkbox("显示完整", key="show_gemini_full")

        # 获取密钥列表
        gemini_keys_data = get_cached_gemini_keys()
        if gemini_keys_data and gemini_keys_data.get('success'):
            gemini_keys = gemini_keys_data.get('keys', [])

            if gemini_keys:
                # 统计信息
                active_count = len([k for k in gemini_keys if k['status'] == 1])
                healthy_count = len(
                    [k for k in gemini_keys if k['status'] == 1 and k.get('health_status') == 'healthy'])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div style="color: #1e40af; font-weight: 500;">共 {len(gemini_keys)} 个密钥</div>',
                                unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div style="color: #1e40af; font-weight: 500;">激活 {active_count} 个</div>',
                                unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div style="color: #10b981; font-weight: 500;">正常 {healthy_count} 个</div>',
                                unsafe_allow_html=True)

                # 密钥列表
                for key_info in gemini_keys:
                    with st.container():
                        st.markdown('<div class="key-container">', unsafe_allow_html=True)

                        col1, col2, col3, col4 = st.columns([1, 4, 1.5, 1])

                        with col1:
                            st.markdown(f"**#{key_info['id']}**")

                        with col2:
                            masked_key = mask_key(key_info['key'], show_full_keys)
                            st.code(masked_key, language=None)

                            # 性能指标
                            if key_info.get('total_requests', 0) > 0:
                                success_rate = key_info.get('success_rate', 1.0)
                                avg_response = key_info.get('avg_response_time', 0.0)
                                st.caption(
                                    f"成功率 {success_rate * 100:.1f}% · 响应时间 {avg_response:.2f}s · 请求数 {key_info['total_requests']}")

                        with col3:
                            # 健康状态
                            health_status = key_info.get('health_status', 'unknown')
                            status_text = format_health_status(health_status)
                            status_class = f"status-{health_status}"
                            st.markdown(f'<div class="status-badge {status_class}">{status_text}</div>',
                                        unsafe_allow_html=True)

                            # 激活状态
                            if key_info['status'] == 1:
                                st.markdown('<div class="status-badge status-active">激活</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="status-badge status-inactive">禁用</div>',
                                            unsafe_allow_html=True)

                        with col4:
                            # 操作按钮
                            toggle_text = "禁用" if key_info['status'] == 1 else "激活"
                            if st.button(toggle_text, key=f"toggle_g_{key_info['id']}", use_container_width=True):
                                if toggle_key_status('gemini', key_info['id']):
                                    st.success("状态已更新")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()

                            if st.button("删除", key=f"del_g_{key_info['id']}", use_container_width=True):
                                if delete_key('gemini', key_info['id']):
                                    st.success("删除成功")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()

                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("暂无密钥，请添加第一个 Gemini API 密钥")
        else:
            st.error("无法获取密钥列表")

    with tab2:
        st.markdown("#### 生成访问密钥")

        with st.form("generate_user_key"):
            key_name = st.text_input("密钥名称", placeholder="例如：生产环境、测试环境")
            submitted = st.form_submit_button("生成新密钥", type="primary")

            if submitted:
                name = key_name if key_name else '未命名'
                result = call_api('/admin/config/user-key', 'POST', {'name': name})
                if result and result.get('success'):
                    new_key = result.get('key')
                    st.success("密钥生成成功")
                    st.warning("请立即保存此密钥，它不会再次显示")
                    st.code(new_key, language=None)

                    with st.expander("使用示例"):
                        st.code(f"""
import openai

client = openai.OpenAI(
    api_key="{new_key}",
    base_url="{API_BASE_URL}/v1"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{{"role": "user", "content": "Hello"}}]
)
                        """, language="python")

                    st.cache_data.clear()

        st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)

        # 现有密钥
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("#### 现有密钥")
        with col2:
            show_full_user_keys = st.checkbox("显示完整", key="show_user_full")

        # 获取用户密钥
        user_keys_data = get_cached_user_keys()
        if user_keys_data and user_keys_data.get('success'):
            user_keys = user_keys_data.get('keys', [])

            if user_keys:
                active_count = len([k for k in user_keys if k['status'] == 1])
                st.markdown(
                    f'<div style="color: #6b7280; font-weight: 500; margin-bottom: 1rem;">共 {len(user_keys)} 个密钥，{active_count} 个激活</div>',
                    unsafe_allow_html=True)

                for key_info in user_keys:
                    with st.container():
                        st.markdown('<div class="key-container">', unsafe_allow_html=True)

                        col1, col2, col3, col4 = st.columns([1, 3, 2, 1])

                        with col1:
                            st.markdown(f"**#{key_info['id']}**")

                        with col2:
                            masked_key = mask_key(key_info['key'], show_full_user_keys)
                            st.code(masked_key, language=None)
                            if key_info.get('name'):
                                st.caption(f"名称: {key_info['name']}")

                        with col3:
                            # 使用情况
                            if key_info.get('last_used'):
                                last_used = key_info['last_used'][:16]
                                st.caption(f"最后使用: {last_used}")
                            else:
                                st.caption("从未使用")

                            # 状态
                            if key_info['status'] == 1:
                                st.markdown('<div class="status-badge status-active">激活</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="status-badge status-inactive">停用</div>',
                                            unsafe_allow_html=True)

                        with col4:
                            toggle_text = "停用" if key_info['status'] == 1 else "激活"
                            if st.button(toggle_text, key=f"toggle_u_{key_info['id']}", use_container_width=True):
                                if toggle_key_status('user', key_info['id']):
                                    st.success("状态已更新")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()

                            if st.button("删除", key=f"del_u_{key_info['id']}", use_container_width=True):
                                if delete_key('user', key_info['id']):
                                    st.success("删除成功")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()

                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("暂无用户密钥")

elif page == "模型配置":
    st.title("模型配置")
    st.markdown('<p class="page-subtitle">调整模型参数和使用限制</p>', unsafe_allow_html=True)

    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("无法获取数据")
        st.stop()

    models = status_data.get('models', [])
    if not models:
        st.warning("暂无可用模型")
        st.stop()

    # 使用内联样式移除黑色边框
    st.markdown(
        '<div style="background: #dbeafe; color: #1e40af; padding: 0.75rem 1rem; border-radius: 6px; font-size: 0.875rem; margin-bottom: 1rem;">'
        '显示的限制针对单个 API Key，总限制会根据健康密钥数量自动倍增'
        '</div>',
        unsafe_allow_html=True
    )

    for model in models:
        st.markdown(f"### {model}")

        current_config = get_cached_model_config(model)
        if not current_config or not current_config.get('success'):
            st.warning(f"无法加载模型配置")
            continue

        with st.form(f"model_config_{model}"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                rpm = st.number_input(
                    "RPM (每分钟请求)",
                    min_value=1,
                    value=current_config.get('single_api_rpm_limit', 10 if 'flash' in model else 5),
                    key=f"rpm_{model}"
                )

            with col2:
                rpd = st.number_input(
                    "RPD (每日请求)",
                    min_value=1,
                    value=current_config.get('single_api_rpd_limit', 250 if 'flash' in model else 100),
                    key=f"rpd_{model}"
                )

            with col3:
                tpm = st.number_input(
                    "TPM (每分钟令牌)",
                    min_value=1000,
                    value=current_config.get('single_api_tpm_limit', 250000),
                    key=f"tpm_{model}"
                )

            with col4:
                status_options = {1: "激活", 0: "禁用"}
                current_status = current_config.get('status', 1)
                new_status = st.selectbox(
                    "状态",
                    options=list(status_options.values()),
                    index=0 if current_status == 1 else 1,
                    key=f"status_{model}"
                )

            if st.form_submit_button("保存配置", type="primary", use_container_width=True):
                update_data = {
                    "single_api_rpm_limit": rpm,
                    "single_api_rpd_limit": rpd,
                    "single_api_tpm_limit": tpm,
                    "status": 1 if new_status == "激活" else 0
                }

                result = call_api(f'/admin/models/{model}', 'POST', data=update_data)
                if result and result.get('success'):
                    st.success("配置已保存")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("保存失败")

elif page == "系统设置":
    st.title("系统设置")
    st.markdown('<p class="page-subtitle">配置高级功能和系统参数</p>', unsafe_allow_html=True)

    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("无法获取配置数据")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["思考模式", "提示词注入", "负载均衡", "保活管理", "系统信息"])

    with tab1:
        st.markdown("#### 思考模式配置")
        st.markdown("启用推理功能以提高复杂查询的响应质量")

        thinking_config = stats_data.get('thinking_config', {})

        with st.form("thinking_config_form"):
            col1, col2 = st.columns(2)

            with col1:
                thinking_enabled = st.checkbox(
                    "启用思考模式",
                    value=thinking_config.get('enabled', False)
                )

                include_thoughts = st.checkbox(
                    "在响应中包含思考过程",
                    value=thinking_config.get('include_thoughts', False)
                )

            with col2:
                budget_options = {
                    "自动": -1,
                    "禁用": 0,
                    "低 (4k)": 4096,
                    "中 (8k)": 8192,
                    "flash最大思考预算 (24k)": 24576,
                    "pro最大思考预算 (32k)": 32768
                }

                current_budget = thinking_config.get('budget', -1)
                selected_option = next((k for k, v in budget_options.items() if v == current_budget), "自动")

                budget_option = st.selectbox(
                    "思考预算",
                    options=list(budget_options.keys()),
                    index=list(budget_options.keys()).index(selected_option)
                )

            if st.form_submit_button("保存配置", type="primary", use_container_width=True):
                update_data = {
                    "enabled": thinking_enabled,
                    "budget": budget_options[budget_option],
                    "include_thoughts": include_thoughts
                }

                result = call_api('/admin/config/thinking', 'POST', data=update_data)
                if result and result.get('success'):
                    st.success("配置已保存")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()

    with tab2:
        st.markdown("#### 提示词注入")
        st.markdown("为所有请求自动添加自定义指令")

        inject_config = stats_data.get('inject_config', {})

        with st.form("inject_prompt_form"):
            inject_enabled = st.checkbox(
                "启用提示词注入",
                value=inject_config.get('enabled', False)
            )

            position_options = {
                'system': '系统消息',
                'user_prefix': '用户消息前',
                'user_suffix': '用户消息后'
            }

            position = st.selectbox(
                "注入位置",
                options=list(position_options.keys()),
                format_func=lambda x: position_options[x],
                index=list(position_options.keys()).index(inject_config.get('position', 'system'))
            )

            content = st.text_area(
                "提示词内容",
                value=inject_config.get('content', ''),
                height=150,
                placeholder="输入自定义提示词..."
            )

            if st.form_submit_button("保存配置", type="primary", use_container_width=True):
                update_data = {
                    "enabled": inject_enabled,
                    "content": content,
                    "position": position
                }

                result = call_api('/admin/config/inject-prompt', 'POST', data=update_data)
                if result and result.get('success'):
                    st.success("配置已保存")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()

    with tab3:
        st.markdown("#### 负载均衡策略")
        st.markdown("优化 API Key 选择策略")

        # 获取当前策略
        all_configs = call_api('/admin/config')
        current_strategy = 'adaptive'

        if all_configs and all_configs.get('success'):
            system_configs = all_configs.get('system_configs', [])
            for config in system_configs:
                if config['key'] == 'load_balance_strategy':
                    current_strategy = config['value']
                    break

        with st.form("load_balance_form"):
            strategy_options = {
                'adaptive': '自适应策略',
                'least_used': '最少使用',
                'round_robin': '轮询'
            }

            strategy_descriptions = {
                'adaptive': '根据成功率和响应时间智能选择',
                'least_used': '优先使用请求最少的密钥',
                'round_robin': '按顺序轮流使用'
            }

            strategy = st.selectbox(
                "选择策略",
                options=list(strategy_options.keys()),
                format_func=lambda x: strategy_options[x],
                index=list(strategy_options.keys()).index(current_strategy)
            )

            st.info(strategy_descriptions[strategy])

            if st.form_submit_button("保存策略", type="primary", use_container_width=True):
                st.success(f"策略已更新为: {strategy_options[strategy]}")

    with tab4:
        st.markdown("#### 保活管理")
        st.markdown("防止服务休眠")

        keep_alive_status = st.session_state.keep_alive_manager.get_status()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("状态", "运行中" if keep_alive_status['running'] else "已停止")
        with col2:
            st.metric("线程", "活跃" if keep_alive_status['thread_alive'] else "停止")
        with col3:
            st.metric("任务数", keep_alive_status['scheduled_jobs'])

        with st.expander("详细信息"):
            if keep_alive_status['render_url']:
                st.text(f"Render URL: {keep_alive_status['render_url']}")
            st.text(f"后端地址: {keep_alive_status['backend_url']}")

        col1, col2 = st.columns(2)
        with col1:
            if not keep_alive_status['running']:
                if st.button("启动保活", type="primary", use_container_width=True):
                    if st.session_state.keep_alive_manager.start_keep_alive_scheduler():
                        st.success("保活服务已启动")
                        time.sleep(1)
                        st.rerun()
        with col2:
            if keep_alive_status['running']:
                if st.button("停止保活", use_container_width=True):
                    st.session_state.keep_alive_manager.stop_scheduler()
                    st.success("保活服务已停止")
                    time.sleep(1)
                    st.rerun()

    with tab5:
        st.markdown("#### 系统信息")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 服务信息")
            st.text(f"Python: {status_data.get('python_version', 'Unknown').split()[0]}")
            st.text(f"版本: {status_data.get('version', '1.0.0')}")
            st.text(f"模型: {', '.join(status_data.get('models', []))}")

        with col2:
            st.markdown("##### 资源使用")
            st.text(f"内存: {status_data.get('memory_usage_mb', 0):.1f} MB")
            st.text(f"CPU: {status_data.get('cpu_percent', 0):.1f}%")
            st.text(f"运行: {status_data.get('uptime_seconds', 0) // 3600} 小时")

# --- 页脚 ---
st.markdown(
    f"""
    <div style='text-align: center; color: #9ca3af; font-size: 0.75rem; margin-top: 4rem; padding: 2rem 0; border-top: 1px solid #e5e7eb;'>
        <a href='{API_BASE_URL}/health' target='_blank' style='color: #6b7280; text-decoration: none;'>健康检查</a> · 
        <span style='color: #9ca3af;'>{API_BASE_URL}</span> ·
        <span style='color: #9ca3af;'>v1.1</span>
    </div>
    """,
    unsafe_allow_html=True
)