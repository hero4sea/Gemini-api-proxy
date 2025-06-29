import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import schedule

# --- 页面配置 ---
st.set_page_config(
    page_title="Gemini API 代理服务",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API配置 ---
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

if 'streamlit.io' in os.getenv('STREAMLIT_SERVER_HEADLESS', ''):
    API_BASE_URL = os.getenv('API_BASE_URL', 'https://your-app.onrender.com')


# --- 保活机制 ---
def keep_alive_task():
    """保活任务，每14分钟执行一次"""
    try:
        response = requests.get(f"{API_BASE_URL}/wake", timeout=10)
        if response.status_code == 200:
            print(f"[{datetime.now()}] Keep-alive ping sent successfully")
    except Exception as e:
        print(f"[{datetime.now()}] Keep-alive ping failed: {e}")


def start_keep_alive_scheduler():
    """启动保活调度器（仅在Streamlit Cloud环境）"""
    # 只在Streamlit Cloud环境启用保活
    if 'streamlit.io' in os.getenv('STREAMLIT_SERVER_HEADLESS', ''):
        # 设置每14分钟执行一次
        schedule.every(14).minutes.do(keep_alive_task)

        # 立即执行一次
        keep_alive_task()

        # 在后台线程中运行调度器
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次

        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        print(f"[{datetime.now()}] Keep-alive scheduler started")


# 初始化保活机制
if 'keep_alive_started' not in st.session_state:
    st.session_state.keep_alive_started = True
    start_keep_alive_scheduler()


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
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }

    /* 度量卡片 */
    [data-testid="metric-container"] {
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        transition: all 0.2s ease;
    }

    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        transform: translateY(-2px);
        border-color: #d1d5db;
    }

    /* 按钮样式 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s ease;
        border: 1px solid transparent;
        font-size: 0.875rem;
        padding: 0.5rem 1rem;
        letter-spacing: 0.01em;
        background: #1f2937;
        color: white;
    }

    .stButton > button:hover {
        background: #374151;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }

    /* 输入框样式 */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
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
        gap: 3rem;
        border-bottom: 1px solid #e5e7eb;
        padding: 0;
        margin-bottom: 1.5rem;
    }

    .stTabs [data-testid="stTabBar"] button {
        font-weight: 500;
        color: #6b7280;
        padding-bottom: 0.875rem;
        border-bottom: 2px solid transparent;
        font-size: 0.875rem;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
    }

    .stTabs [data-testid="stTabBar"] button[aria-selected="true"] {
        color: #1f2937;
        border-bottom-color: #6366f1;
    }

    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f9fafb 0%, #f3f4f6 100%);
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #4b5563;
    }

    /* 成功/错误消息样式 */
    .stAlert {
        border-radius: 8px;
        font-size: 0.875rem;
        padding: 0.875rem 1rem;
    }

    /* 图表 */
    .js-plotly-plot .plotly {
        border-radius: 8px;
        overflow: hidden;
    }

    /* 表格样式 */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    /* 分隔线样式 */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid #e5e7eb;
    }

    /* 标题样式 */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    .css-1d391kg {
        padding-top: 1rem;
    }

    /* 密钥容器样式 */
    .key-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }

    .key-container:hover {
        background: #f1f5f9;
        border-color: #cbd5e1;
    }

    /* 状态指示器 */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    .status-active {
        background: #dcfce7;
        color: #166534;
    }

    .status-inactive {
        background: #fee2e2;
        color: #991b1b;
    }

    /* 垂直布局容器 */
    .vertical-layout {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    /* 状态操作区域 */
    .status-action-area {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        padding: 0.5rem 0;
    }

    /* 状态操作区域中的按钮 */
    .status-action-area + div [data-testid="stButton"] > button {
        margin-top: 0.5rem;
        font-size: 0.8rem;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    /* 密钥容器改进 */
    .key-item-container {
        background: #fafbfc;
        border: 1px solid #e1e5e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.75rem 0;
        transition: all 0.2s ease;
    }

    .key-item-container:hover {
        background: #f6f8fa;
        border-color: #d0d7de;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏 ---
with st.sidebar:
    st.markdown("### Gemini API 代理服务")
    st.markdown("---")

    page = st.radio(
        "导航",
        ["控制台", "模型配置", "密钥管理", "系统设置"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # 服务状态检查
    st.markdown("#### 服务状态")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("刷新", use_container_width=True):
            st.cache_data.clear()
    with col2:
        if st.button("唤醒", use_container_width=True):
            wake_up_service()

    # 检查服务健康状态
    health = check_service_health()
    if health:
        st.success("服务在线")
        with st.expander("详细信息"):
            st.text(f"地址: {API_BASE_URL}")
            st.text(f"状态: {health.get('status', 'unknown')}")
            st.text(f"运行时间: {health.get('uptime_seconds', 0)}秒")
    else:
        st.error("服务离线")
        st.info("点击'唤醒'按钮激活服务")

    st.markdown("---")

    # 快速统计
    st.markdown("#### 系统概览")
    status_data = get_cached_status()
    if status_data:
        st.metric("可用密钥", status_data.get('active_keys', 0))
        thinking_enabled = status_data.get('thinking_enabled', False)
        st.metric("思考模式", "开启" if thinking_enabled else "关闭")

        memory_mb = status_data.get('memory_usage_mb', 0)
        if memory_mb > 0:
            st.metric("内存使用", f"{memory_mb:.1f}MB")

# --- 主页面内容 ---
if page == "控制台":
    st.title("服务控制台")
    st.markdown("监控 API 代理服务使用指标")

    # 刷新按钮
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("↻", help="刷新数据", key="refresh_dashboard"):
            st.cache_data.clear()
            st.rerun()

    # 获取统计数据
    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("无法获取服务数据")
        st.info("请尝试点击侧边栏的'唤醒'按钮")
        st.stop()

    # 核心指标
    st.markdown("## 核心指标")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gemini_keys = stats_data.get('active_gemini_keys', 0)
        total_gemini = stats_data.get('gemini_keys', 0)
        st.metric(
            "Gemini密钥",
            gemini_keys,
            delta=f"共{total_gemini}个"
        )

    with col2:
        user_keys = stats_data.get('active_user_keys', 0)
        total_user = stats_data.get('user_keys', 0)
        st.metric(
            "用户密钥",
            user_keys,
            delta=f"共{total_user}个"
        )

    with col3:
        models = stats_data.get('supported_models', [])
        st.metric("支持模型", len(models))

    with col4:
        thinking_status = "已启用" if status_data.get('thinking_enabled', False) else "已禁用"
        st.metric("思考功能", thinking_status)

    # 使用率分析
    st.markdown("## 使用率分析")

    usage_stats = stats_data.get('usage_stats', {})
    if usage_stats and models:
        # 准备数据
        model_data = []
        for model in models:
            stats = usage_stats.get(model, {'minute': {'requests': 0}, 'day': {'requests': 0}})

            model_config_data = get_cached_model_config(model)
            if not model_config_data:
                rpm_limit = 1000 if 'flash' in model else 100
                rpd_limit = 50000 if 'flash' in model else 10000
            else:
                rpm_limit = model_config_data.get('total_rpm_limit', 1000)
                rpd_limit = model_config_data.get('total_rpd_limit', 50000)

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
                    title={
                        'text': "每分钟请求数 (RPM)",
                        'font': {'size': 14, 'color': '#1f2937', 'family': '-apple-system, BlinkMacSystemFont'}
                    },
                    yaxis_title="使用率 (%)",
                    yaxis_range=[0, max(100, df['RPM %'].max() * 1.2) if len(df) > 0 else 100],
                    height=320,
                    showlegend=False,
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font={'family': '-apple-system, BlinkMacSystemFont', 'color': '#4b5563', 'size': 12},
                    yaxis={'gridcolor': '#e5e7eb', 'zerolinecolor': '#e5e7eb'},
                    xaxis={'linecolor': '#e5e7eb'},
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
                    title={
                        'text': "每日请求数 (RPD)",
                        'font': {'size': 14, 'color': '#1f2937', 'family': '-apple-system, BlinkMacSystemFont'}
                    },
                    yaxis_title="使用率 (%)",
                    yaxis_range=[0, max(100, df['RPD %'].max() * 1.2) if len(df) > 0 else 100],
                    height=320,
                    showlegend=False,
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    font={'family': '-apple-system, BlinkMacSystemFont', 'color': '#4b5563', 'size': 12},
                    yaxis={'gridcolor': '#e5e7eb', 'zerolinecolor': '#e5e7eb'},
                    xaxis={'linecolor': '#e5e7eb'},
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
        st.info("暂无使用数据。请先配置API密钥并发送请求。")

elif page == "密钥管理":
    st.title("密钥管理")
    st.markdown("管理 Gemini API 密钥和用户访问令牌")

    # 全局刷新按钮
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("刷新", help="刷新数据", key="refresh_keys"):
            st.cache_data.clear()
            st.rerun()

    tab1, tab2 = st.tabs(["Gemini 密钥", "用户密钥"])

    with tab1:
        st.markdown("### 添加新密钥")

        with st.form("add_gemini_key"):
            new_key = st.text_input(
                "Gemini API 密钥",
                type="password",
                placeholder="输入你的 Gemini API 密钥...",
                help="从 Google AI Studio 获取你的 Gemini API 密钥"
            )
            submitted = st.form_submit_button("添加密钥", type="primary")

            if submitted and new_key:
                result = call_api('/admin/config/gemini-key', 'POST', {'key': new_key})
                if result and result.get('success'):
                    st.success("密钥添加成功！")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("添加失败，密钥可能已存在。")

        st.divider()

        # 显示控制选项
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 现有密钥")
        with col2:
            show_full_keys = st.checkbox("显示完整密钥", help="注意信息安全", key="show_gemini_full")

        # 获取真实的Gemini密钥
        gemini_keys_data = get_cached_gemini_keys()
        if gemini_keys_data and gemini_keys_data.get('success'):
            gemini_keys = gemini_keys_data.get('keys', [])

            if gemini_keys:
                active_count = len([k for k in gemini_keys if k['status'] == 1])
                st.info(f"共有 {len(gemini_keys)} 个密钥，其中 {active_count} 个处于激活状态")

                # 显示密钥列表
                for i, key_info in enumerate(gemini_keys):
                    with st.container():
                        # 添加密钥项容器
                        st.markdown('<div class="key-item-container">', unsafe_allow_html=True)

                        col1, col2, col3, col4 = st.columns([1, 4, 1.5, 1])

                        with col1:
                            st.markdown(f"**#{key_info['id']}**")

                        with col2:
                            masked_key = mask_key(key_info['key'], show_full_keys)
                            st.code(masked_key, language=None)

                            # 显示创建时间
                            if 'created_at' in key_info:
                                created_date = key_info['created_at'][:10] if len(key_info['created_at']) > 10 else \
                                key_info['created_at']
                                st.caption(f"创建于: {created_date}")

                        with col3:
                            # 状态操作区域，使用垂直布局
                            st.markdown('<div class="status-action-area">', unsafe_allow_html=True)

                            # 状态显示
                            if key_info['status'] == 1:
                                st.markdown('<div class="status-indicator status-active">激活</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="status-indicator status-inactive">禁用</div>',
                                            unsafe_allow_html=True)

                            st.markdown('</div>', unsafe_allow_html=True)

                            # 状态切换按钮
                            toggle_text = "禁用" if key_info['status'] == 1 else "激活"
                            if st.button(toggle_text, key=f"toggle_gemini_{key_info['id']}", use_container_width=True):
                                if toggle_key_status('gemini', key_info['id']):
                                    st.success(f"状态已更新！")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("状态更新失败")

                        with col4:
                            # 删除按钮，使用确认对话框
                            delete_key_id = f"delete_gemini_{key_info['id']}"
                            if delete_key_id not in st.session_state:
                                st.session_state[delete_key_id] = False

                            if st.session_state[delete_key_id]:
                                # 显示确认界面
                                st.error("确认删除？")
                                col_yes, col_no = st.columns(2)
                                with col_yes:
                                    if st.button("确认", key=f"confirm_del_gemini_{key_info['id']}",
                                                 use_container_width=True, type="primary"):
                                        if delete_key('gemini', key_info['id']):
                                            st.success("删除成功！")
                                            st.cache_data.clear()
                                            st.session_state[delete_key_id] = False
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.error("删除失败")
                                            st.session_state[delete_key_id] = False
                                with col_no:
                                    if st.button("取消", key=f"cancel_del_gemini_{key_info['id']}",
                                                 use_container_width=True):
                                        st.session_state[delete_key_id] = False
                                        st.rerun()
                            else:
                                if st.button("删除", key=f"del_gemini_{key_info['id']}", use_container_width=True,
                                             help="删除密钥"):
                                    st.session_state[delete_key_id] = True
                                    st.rerun()

                        # 关闭密钥项容器
                        st.markdown('</div>', unsafe_allow_html=True)

                # 统计信息
                with st.expander("统计信息", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总密钥数", len(gemini_keys))
                    with col2:
                        st.metric("激活密钥", active_count)
                    with col3:
                        usage_rate = (active_count / len(gemini_keys) * 100) if len(gemini_keys) > 0 else 0
                        st.metric("激活率", f"{usage_rate:.1f}%")
            else:
                st.info("暂无 Gemini 密钥。请在上方添加你的第一个密钥。")
        else:
            st.error("无法获取 Gemini 密钥列表。请检查API连接。")

    with tab2:
        st.markdown("### 生成访问密钥")

        with st.form("generate_user_key"):
            key_name = st.text_input("密钥名称", placeholder="例如：移动应用、网站后端等")
            submitted = st.form_submit_button("生成新密钥", type="primary")

            if submitted:
                name = key_name if key_name else 'API密钥'
                result = call_api('/admin/config/user-key', 'POST', {'name': name})
                if result and result.get('success'):
                    new_key = result.get('key')
                    st.success("用户密钥生成成功！")
                    st.warning("请立即保存此密钥，它不会再次显示。")

                    # 可复制的密钥显示
                    st.text_area("新生成的密钥", new_key, height=80, help="选中文本并复制")

                    # 使用说明
                    with st.expander("使用示例", expanded=True):
                        st.code(f"""
import openai

client = openai.OpenAI(
    api_key="{new_key}",
    base_url="{API_BASE_URL}/v1"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{{"role": "user", "content": "你好！"}}]
)

print(response.choices[0].message.content)
                        """, language="python")

                    st.cache_data.clear()
                else:
                    st.error("生成失败，请重试。")

        st.divider()

        # 显示控制选项
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 现有密钥")
        with col2:
            show_full_user_keys = st.checkbox("显示完整密钥", key="show_user_keys", help="注意信息安全")

        # 获取真实的用户密钥
        user_keys_data = get_cached_user_keys()
        if user_keys_data and user_keys_data.get('success'):
            user_keys = user_keys_data.get('keys', [])

            if user_keys:
                active_count = len([k for k in user_keys if k['status'] == 1])
                st.info(f"共有 {len(user_keys)} 个用户密钥，其中 {active_count} 个处于激活状态")

                # 显示用户密钥列表
                for i, key_info in enumerate(user_keys):
                    with st.container():
                        # 添加密钥项容器
                        st.markdown('<div class="key-item-container">', unsafe_allow_html=True)

                        col1, col2, col3, col4 = st.columns([1, 3, 2, 1])

                        with col1:
                            st.markdown(f"**#{key_info['id']}**")

                        with col2:
                            masked_key = mask_key(key_info['key'], show_full_user_keys)
                            st.code(masked_key, language=None)

                            # 显示密钥名称
                            if key_info.get('name'):
                                st.caption(f"名称: {key_info['name']}")

                            # 显示创建时间
                            if 'created_at' in key_info:
                                created_date = key_info['created_at'][:10] if len(key_info['created_at']) > 10 else \
                                key_info['created_at']
                                st.caption(f"创建于: {created_date}")

                        with col3:
                            # 状态操作区域，使用垂直布局
                            st.markdown('<div class="status-action-area">', unsafe_allow_html=True)

                            # 显示最后使用时间
                            if key_info.get('last_used'):
                                last_used = key_info['last_used'][:16] if len(key_info['last_used']) > 16 else key_info[
                                    'last_used']
                                st.caption(f"最后使用: {last_used}")
                            else:
                                st.caption("从未使用")

                            # 状态显示
                            if key_info['status'] == 1:
                                st.markdown('<div class="status-indicator status-active">激活</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="status-indicator status-inactive">停用</div>',
                                            unsafe_allow_html=True)

                            st.markdown('</div>', unsafe_allow_html=True)

                            # 在状态下方放置切换按钮
                            toggle_text = "停用" if key_info['status'] == 1 else "激活"
                            if st.button(toggle_text, key=f"toggle_user_{key_info['id']}", use_container_width=True):
                                if toggle_key_status('user', key_info['id']):
                                    st.success(f"状态已更新！")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("状态更新失败")

                        with col4:
                            # 删除按钮，使用确认对话框
                            delete_key_id = f"delete_user_{key_info['id']}"
                            if delete_key_id not in st.session_state:
                                st.session_state[delete_key_id] = False

                            if st.session_state[delete_key_id]:
                                # 显示确认界面
                                st.error("确认删除？")
                                col_yes, col_no = st.columns(2)
                                with col_yes:
                                    if st.button("确认", key=f"confirm_del_user_{key_info['id']}",
                                                 use_container_width=True, type="primary"):
                                        if delete_key('user', key_info['id']):
                                            st.success("删除成功！")
                                            st.cache_data.clear()
                                            st.session_state[delete_key_id] = False
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.error("删除失败")
                                            st.session_state[delete_key_id] = False
                                with col_no:
                                    if st.button("取消", key=f"cancel_del_user_{key_info['id']}",
                                                 use_container_width=True):
                                        st.session_state[delete_key_id] = False
                                        st.rerun()
                            else:
                                if st.button("删除", key=f"del_user_{key_info['id']}", use_container_width=True,
                                             help="删除密钥"):
                                    st.session_state[delete_key_id] = True
                                    st.rerun()

                        # 关闭密钥项容器
                        st.markdown('</div>', unsafe_allow_html=True)

                # 统计信息
                with st.expander("统计信息", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总密钥数", len(user_keys))
                    with col2:
                        st.metric("激活密钥", active_count)
                    with col3:
                        usage_rate = (active_count / len(user_keys) * 100) if len(user_keys) > 0 else 0
                        st.metric("激活率", f"{usage_rate:.1f}%")
            else:
                st.info("暂无用户密钥。请在上方生成你的第一个访问密钥。")
        else:
            st.error("无法获取用户密钥列表。请检查API连接。")

elif page == "模型配置":
    st.title("模型配置")
    st.markdown("查看并调整模型状态和使用限制")

    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("无法获取模型数据")
        st.stop()

    models = status_data.get('models', [])
    if not models:
        st.warning("暂无可用模型")
        st.stop()

    st.info("显示的限制是针对单个 Gemini API Key 的，总限制会根据激活的密钥数量自动倍增。")

    for model in models:
        st.markdown(f"---")
        st.markdown(f"### {model}")

        # 获取当前模型的配置
        current_config = get_cached_model_config(model)
        if not current_config or not current_config.get('success'):
            st.warning(f"无法加载模型 {model} 的配置。")
            continue

        with st.form(f"model_config_form_{model}"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("#### 单Key限制")
                rpm = st.number_input(
                    "RPM",
                    min_value=1,
                    value=current_config.get('single_api_rpm_limit', 1000),
                    key=f"rpm_{model}",
                    help="每分钟请求数"
                )

            with col2:
                st.markdown("#### &nbsp;")
                rpd = st.number_input(
                    "RPD",
                    min_value=1,
                    value=current_config.get('single_api_rpd_limit', 50000),
                    key=f"rpd_{model}",
                    help="每日请求数"
                )

            with col3:
                st.markdown("#### &nbsp;")
                tpm = st.number_input(
                    "TPM",
                    min_value=1000,
                    value=current_config.get('single_api_tpm_limit', 2000000),
                    key=f"tpm_{model}",
                    help="每分钟令牌数"
                )

            with col4:
                st.markdown("#### 状态")
                status_options = {1: "激活", 0: "禁用"}
                current_status_label = status_options.get(current_config.get('status', 1), "激活")
                new_status_label = st.selectbox(
                    "状态",
                    options=list(status_options.values()),
                    index=list(status_options.values()).index(current_status_label),
                    key=f"status_{model}"
                )

            submitted = st.form_submit_button(f"保存 {model} 配置", type="primary", use_container_width=True)

            if submitted:
                new_status = 1 if new_status_label == "激活" else 0
                update_data = {
                    "single_api_rpm_limit": rpm,
                    "single_api_rpd_limit": rpd,
                    "single_api_tpm_limit": tpm,
                    "status": new_status
                }

                result = call_api(f'/admin/models/{model}', 'POST', data=update_data)
                if result and result.get('success'):
                    st.success(f"{model} 配置已成功保存！")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"更新模型 {model} 失败！")

elif page == "系统设置":
    st.title("系统设置")
    st.markdown("配置高级功能和系统行为")

    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("无法获取配置数据")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["思考模式", "提示词注入", "系统信息"])

    with tab1:
        st.markdown("### 思考模式配置")
        st.markdown("启用内部推理以提高复杂查询的响应质量。")

        thinking_config = stats_data.get('thinking_config', {})

        thinking_enabled = thinking_config.get('enabled', False)
        thinking_budget = thinking_config.get('budget', -1)
        include_thoughts = thinking_config.get('include_thoughts', False)

        with st.form("thinking_config_form"):
            st.markdown("#### 配置选项")

            new_thinking_enabled = st.checkbox(
                "启用思考模式",
                value=thinking_enabled,
                help="模型将在生成响应前进行内部推理"
            )

            new_include_thoughts = st.checkbox(
                "在API响应中包含思考过程",
                value=include_thoughts,
                help="API响应将包含模型的推理过程"
            )

            budget_options = {
                "自动": -1,
                "禁用": 0,
                "低 (4k)": 4096,
                "中 (8k)": 8192,
                "高 (24k)": 24576,
                "最高 (32k)": 32768,
                "自定义": "custom"
            }

            current_option = next((k for k, v in budget_options.items() if v == thinking_budget), "自定义")

            selected_option = st.selectbox(
                "思考预算",
                options=list(budget_options.keys()),
                index=list(budget_options.keys()).index(current_option),
                help="控制思考过程的深度"
            )

            if selected_option == "自定义":
                new_budget = st.number_input(
                    "自定义令牌数",
                    min_value=-1,
                    max_value=32768,
                    value=thinking_budget if thinking_budget > 0 else 4096
                )
            else:
                new_budget = budget_options[selected_option]

            if st.form_submit_button("保存配置", type="primary", use_container_width=True):
                update_data = {
                    "enabled": new_thinking_enabled,
                    "budget": new_budget,
                    "include_thoughts": new_include_thoughts
                }

                result = call_api('/admin/config/thinking', 'POST', data=update_data)
                if result and result.get('success'):
                    st.success("思考模式配置已保存！")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("保存失败，请重试")

        with st.expander("当前配置"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("模式", "启用" if thinking_enabled else "禁用")
                st.metric("预算", f"{thinking_budget} tokens" if thinking_budget >= 0 else "自动")
            with col2:
                st.metric("显示思考过程", "是" if include_thoughts else "否")

    with tab2:
        st.markdown("### 提示词注入")
        st.markdown("自动为所有API请求添加自定义指令。")

        inject_config = stats_data.get('inject_config', {})

        inject_enabled = inject_config.get('enabled', False)
        inject_content = inject_config.get('content', '')
        inject_position = inject_config.get('position', 'system')

        with st.form("inject_prompt_form"):
            st.markdown("#### 配置选项")

            new_inject_enabled = st.checkbox(
                "启用提示词注入",
                value=inject_enabled,
                help="所有请求都会包含你的自定义提示词"
            )

            position_options = {
                'system': '作为系统消息',
                'user_prefix': '用户消息之前',
                'user_suffix': '用户消息之后'
            }

            new_position = st.selectbox(
                "注入位置",
                options=list(position_options.keys()),
                format_func=lambda x: position_options[x],
                index=list(position_options.keys()).index(inject_position)
            )

            new_content = st.text_area(
                "自定义提示词内容",
                value=inject_content,
                height=150,
                placeholder="你是一个专业的AI助手...",
                help="这里输入的内容会自动添加到所有API请求中"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("保存配置", type="primary", use_container_width=True):
                    update_data = {
                        "enabled": new_inject_enabled,
                        "content": new_content,
                        "position": new_position
                    }

                    result = call_api('/admin/config/inject-prompt', 'POST', data=update_data)
                    if result and result.get('success'):
                        st.success("提示词注入配置已保存！")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("保存失败，请重试")

            with col2:
                if st.form_submit_button("清除内容", type="secondary", use_container_width=True):
                    clear_data = {
                        "enabled": False,
                        "content": "",
                        "position": "system"
                    }

                    result = call_api('/admin/config/inject-prompt', 'POST', data=clear_data)
                    if result and result.get('success'):
                        st.success("提示词内容已清除！")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("清除失败，请重试")

        with st.expander("当前配置"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("状态", "启用" if inject_enabled else "禁用")
                st.metric("位置", position_options.get(inject_position, inject_position))
            with col2:
                content_preview = inject_content[:50] + "..." if len(inject_content) > 50 else inject_content
                st.metric("内容预览", content_preview if content_preview else "无")

    with tab3:
        st.markdown("### 系统信息")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 服务信息")
            st.metric("Python版本", status_data.get('python_version', 'Unknown').split()[0])
            st.metric("服务版本", status_data.get('version', '1.0.0'))
            st.metric("保持唤醒", "激活" if status_data.get('keep_alive_active', False) else "未激活")

        with col2:
            st.markdown("#### 支持的模型")
            models = status_data.get('models', [])
            for model in models:
                st.markdown(f"• {model}")

        st.markdown("### 系统指标")

        col1, col2, col3 = st.columns(3)

        with col1:
            memory_mb = status_data.get('memory_usage_mb', 0)
            st.metric("内存使用", f"{memory_mb:.1f} MB")

        with col2:
            cpu_percent = status_data.get('cpu_percent', 0)
            st.metric("CPU使用率", f"{cpu_percent:.1f}%")

        with col3:
            uptime = status_data.get('uptime_seconds', 0)
            uptime_hours = uptime / 3600
            st.metric("运行时间", f"{uptime_hours:.1f} 小时")

# --- 页脚 ---
st.markdown(
    f"""
    <div style='text-align: center; color: #9ca3af; font-size: 0.75rem; margin-top: 4rem; padding: 2rem 0; border-top: 1px solid #e5e7eb;'>
        Gemini API 代理服务 | 
        <a href='{API_BASE_URL}/health' target='_blank' style='color: #9ca3af;'>健康检查</a> | 
        <span style='color: #9ca3af;'>端点: {API_BASE_URL}</span>
    </div>
    """,
    unsafe_allow_html=True
)