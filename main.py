import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

# --- 页面配置 ---
st.set_page_config(
    page_title="Gemini API Gateway",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API配置 ---
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

if 'streamlit.io' in os.getenv('STREAMLIT_SERVER_HEADLESS', ''):
    API_BASE_URL = os.getenv('API_BASE_URL', 'https://your-app.onrender.com')

# --- API调用函数 ---
def call_api(endpoint: str, method: str = 'GET', data: Any = None, timeout: int = 30) -> Optional[Dict]:
    """统一API调用函数"""
    url = f"{API_BASE_URL}{endpoint}"

    try:
        spinner_message = "Loading..." if method == 'GET' else "Saving..."
        with st.spinner(spinner_message):
            if method == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                return None

    except requests.exceptions.Timeout:
        st.error("Request timeout. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API service.")
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def wake_up_service():
    """唤醒服务"""
    try:
        response = requests.get(f"{API_BASE_URL}/wake", timeout=10)
        if response.status_code == 200:
            st.success("Service activated")
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


# --- 自定义CSS样式 - 极简高级设计 ---
st.markdown("""
<style>
    /* 全局字体优化 */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro SC", "SF Pro Display", "Helvetica Neue", "PingFang SC", "Microsoft YaHei UI", sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* 优化整体布局 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }

    /* 度量卡片 - 极简风格 */
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

    /* 按钮样式 - 更精致 */
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

    /* 输入框样式 - 更精致 */
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

    /* 标签页样式 - 更现代 */
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

    /* 侧边栏样式 - 更精致 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f9fafb 0%, #f3f4f6 100%);
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #4b5563;
    }

    /* 成功/错误消息样式 - 更精致 */
    .stAlert {
        border-radius: 8px;
        font-size: 0.875rem;
        padding: 0.875rem 1rem;
    }

    /* 图表优化 */
    .js-plotly-plot .plotly {
        border-radius: 8px;
        overflow: hidden;
    }

    /* 表格样式优化 */
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

    /* 标题样式优化 */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    /* 移除多余的padding */
    .css-1d391kg {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏 ---
with st.sidebar:
    st.markdown("### Gemini API Gateway")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Models", "API Keys", "Settings"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # 服务状态检查
    st.markdown("#### Service Status")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
    with col2:
        if st.button("Wake Up", use_container_width=True):
            wake_up_service()

    # 检查服务健康状态
    health = check_service_health()
    if health:
        st.success("Service Online")
        with st.expander("Details"):
            st.text(f"URL: {API_BASE_URL}")
            st.text(f"Status: {health.get('status', 'unknown')}")
            st.text(f"Uptime: {health.get('uptime_seconds', 0)}s")
    else:
        st.error("Service Offline")
        st.info("Click 'Wake Up' to activate")

    st.markdown("---")

    # 快速统计
    st.markdown("#### System Overview")
    status_data = get_cached_status()
    if status_data:
        st.metric("Active Keys", status_data.get('active_keys', 0))
        thinking_enabled = status_data.get('thinking_enabled', False)
        st.metric("Thinking Mode", "ON" if thinking_enabled else "OFF")

        memory_mb = status_data.get('memory_usage_mb', 0)
        if memory_mb > 0:
            st.metric("Memory", f"{memory_mb:.1f}MB")

# --- 主页面内容 ---
if page == "Dashboard":
    st.title("Service Dashboard")
    st.markdown("Monitor API gateway performance and usage metrics")

    # 刷新按钮
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("↻", help="Refresh data", key="refresh_dashboard"):
            st.cache_data.clear()
            st.rerun()

    # 获取统计数据
    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("Unable to retrieve service data")
        st.info("Try clicking 'Wake Up' in the sidebar")
        st.stop()

    # 核心指标
    st.markdown("## Core Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gemini_keys = stats_data.get('active_gemini_keys', 0)
        total_gemini = stats_data.get('gemini_keys', 0)
        st.metric(
            "Gemini Keys",
            gemini_keys,
            delta=f"of {total_gemini}"
        )

    with col2:
        user_keys = stats_data.get('active_user_keys', 0)
        total_user = stats_data.get('user_keys', 0)
        st.metric(
            "User Keys",
            user_keys,
            delta=f"of {total_user}"
        )

    with col3:
        models = stats_data.get('supported_models', [])
        st.metric("Models", len(models))

    with col4:
        thinking_status = "Enabled" if status_data.get('thinking_enabled', False) else "Disabled"
        st.metric("Thinking", thinking_status)

    # 系统状态
    st.markdown("## System Status")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        uptime = status_data.get('uptime_seconds', 0)
        uptime_hours = uptime / 3600
        st.metric("Uptime", f"{uptime_hours:.1f}h")

    with col2:
        memory_mb = status_data.get('memory_usage_mb', 0)
        st.metric("Memory", f"{memory_mb:.1f}MB")

    with col3:
        cpu_percent = status_data.get('cpu_percent', 0)
        st.metric("CPU", f"{cpu_percent:.1f}%")

    with col4:
        total_requests = status_data.get('total_requests', 0)
        st.metric("Requests", f"{total_requests:,}")

    # 使用率分析
    st.markdown("## Usage Analysis")

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
                    hovertemplate='<b>%{x}</b><br>Usage: %{y:.1f}%<br>Current: %{customdata[0]:,}<br>Limit: %{customdata[1]:,}<extra></extra>',
                    customdata=df[['RPM Used', 'RPM Limit']].values
                ))
                fig_rpm.update_layout(
                    title={
                        'text': "Requests per Minute (RPM)",
                        'font': {'size': 14, 'color': '#1f2937', 'family': '-apple-system, BlinkMacSystemFont'}
                    },
                    yaxis_title="Usage (%)",
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
                    hovertemplate='<b>%{x}</b><br>Usage: %{y:.1f}%<br>Current: %{customdata[0]:,}<br>Limit: %{customdata[1]:,}<extra></extra>',
                    customdata=df[['RPD Used', 'RPD Limit']].values
                ))
                fig_rpd.update_layout(
                    title={
                        'text': "Requests per Day (RPD)",
                        'font': {'size': 14, 'color': '#1f2937', 'family': '-apple-system, BlinkMacSystemFont'}
                    },
                    yaxis_title="Usage (%)",
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
            with st.expander("View detailed data"):
                display_df = df[['Model', 'RPM Used', 'RPM Limit', 'RPM %', 'RPD Used', 'RPD Limit', 'RPD %']].copy()
                display_df['RPM %'] = display_df['RPM %'].apply(lambda x: f"{x:.1f}%")
                display_df['RPD %'] = display_df['RPD %'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No usage data available. Please configure API keys and send requests.")

elif page == "API Keys":
    st.title("API Key Management")
    st.markdown("Manage Gemini API keys and user access tokens")

    tab1, tab2 = st.tabs(["Gemini Keys", "User Keys"])

    with tab1:
        st.markdown("### Add New Key")

        with st.form("add_gemini_key"):
            new_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="Enter your Gemini API key...",
                help="Get your Gemini API key from Google AI Studio"
            )
            submitted = st.form_submit_button("Add Key", type="primary")

            if submitted and new_key:
                result = call_api('/admin/config/gemini-key', 'POST', {'key': new_key})
                if result and result.get('success'):
                    st.success("Key added successfully!")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to add key. It may already exist.")

        st.divider()

        # 显示现有密钥
        st.markdown("### Active Keys")
        stats_data = get_cached_stats()
        if stats_data:
            total_keys = stats_data.get('gemini_keys', 0)
            active_keys = stats_data.get('active_gemini_keys', 0)

            if total_keys > 0:
                st.info(f"Total: {total_keys} keys, Active: {active_keys}")

                # 显示密钥列表
                for i in range(min(total_keys, 5)):
                    with st.container():
                        col1, col2, col3 = st.columns([1, 4, 1])
                        with col1:
                            st.markdown(f"**#{i + 1}**")
                        with col2:
                            masked_key = f"AIzaSy{'•' * 30}abc{i + 1:02d}"
                            st.code(masked_key, language=None)
                        with col3:
                            status = "Active" if i < active_keys else "Inactive"
                            st.markdown(f"**{status}**")

                        if i < total_keys - 1:
                            st.markdown("---")
            else:
                st.info("No Gemini keys configured. Add your first key above.")

    with tab2:
        st.markdown("### Generate Access Key")

        with st.form("generate_user_key"):
            submitted = st.form_submit_button("Generate New Key", type="primary")

            if submitted:
                result = call_api('/admin/config/user-key', 'POST', {'name': 'API Key'})
                if result and result.get('success'):
                    new_key = result.get('key')
                    st.success("User key generated successfully!")
                    st.warning("Save this key immediately. It won't be shown again.")
                    st.code(new_key, language=None)

                    # 使用说明
                    st.markdown("### Usage Example")
                    st.code(f"""
import openai

client = openai.OpenAI(
    api_key="{new_key}",
    base_url="{API_BASE_URL}/v1"
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{{"role": "user", "content": "Hello!"}}]
)
                    """, language="python")

                    st.cache_data.clear()
                else:
                    st.error("Failed to generate key. Please try again.")

        st.divider()

        # 显示现有用户密钥
        st.markdown("### Active Keys")
        stats_data = get_cached_stats()
        if stats_data:
            total_user_keys = stats_data.get('user_keys', 0)
            active_user_keys = stats_data.get('active_user_keys', 0)

            if total_user_keys > 0:
                st.info(f"Total: {total_user_keys} keys, Active: {active_user_keys}")

                # 用户密钥列表
                data = []
                for i in range(min(total_user_keys, 10)):
                    data.append({
                        'ID': i + 1,
                        'Key Preview': f"sk-{'•' * 15}...",
                        'Status': 'Active' if i < active_user_keys else 'Inactive'
                    })

                df = pd.DataFrame(data)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'ID': st.column_config.NumberColumn(width='small'),
                        'Status': st.column_config.TextColumn(width='small')
                    }
                )
            else:
                st.info("No user keys generated. Generate your first access key above.")

elif page == "Models":
    st.title("Model Configuration")
    st.markdown("View and adjust model status and usage limits")

    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("Unable to retrieve model data")
        st.stop()

    models = status_data.get('models', [])
    if not models:
        st.warning("No models available")
        st.stop()

    st.info("Limits shown are per API key. Total limits scale with the number of active keys.")

    for model in models:
        st.markdown(f"---")
        st.markdown(f"### {model}")

        # 获取当前模型的配置
        current_config = get_cached_model_config(model)
        if not current_config or not current_config.get('success'):
            st.warning(f"Unable to load configuration for {model}")
            continue

        with st.form(f"model_config_form_{model}"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("#### Per-Key Limits")
                rpm = st.number_input(
                    "RPM",
                    min_value=1,
                    value=current_config.get('single_api_rpm_limit', 1000),
                    key=f"rpm_{model}",
                    help="Requests per minute"
                )

            with col2:
                st.markdown("#### &nbsp;")
                rpd = st.number_input(
                    "RPD",
                    min_value=1,
                    value=current_config.get('single_api_rpd_limit', 50000),
                    key=f"rpd_{model}",
                    help="Requests per day"
                )

            with col3:
                st.markdown("#### &nbsp;")
                tpm = st.number_input(
                    "TPM",
                    min_value=1000,
                    value=current_config.get('single_api_tpm_limit', 2000000),
                    key=f"tpm_{model}",
                    help="Tokens per minute"
                )

            with col4:
                st.markdown("#### Status")
                status_options = {1: "Active", 0: "Disabled"}
                current_status_label = status_options.get(current_config.get('status', 1), "Active")
                new_status_label = st.selectbox(
                    "Status",
                    options=list(status_options.values()),
                    index=list(status_options.values()).index(current_status_label),
                    key=f"status_{model}"
                )

            submitted = st.form_submit_button(f"Save {model} Configuration", type="primary", use_container_width=True)

            if submitted:
                new_status = 1 if new_status_label == "Active" else 0
                update_data = {
                    "single_api_rpm_limit": rpm,
                    "single_api_rpd_limit": rpd,
                    "single_api_tpm_limit": tpm,
                    "status": new_status
                }

                result = call_api(f'/admin/models/{model}', 'POST', data=update_data)
                if result and result.get('success'):
                    st.success(f"{model} configuration saved successfully!")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Failed to update {model}")

elif page == "Settings":
    st.title("Settings")
    st.markdown("Configure advanced features and system behavior")

    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("Unable to retrieve configuration data")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Thinking Mode", "Prompt Injection", "System"])

    with tab1:
        st.markdown("### Thinking Mode Configuration")
        st.markdown("Enable internal reasoning to improve response quality for complex queries.")

        thinking_config = stats_data.get('thinking_config', {})

        thinking_enabled = thinking_config.get('enabled', False)
        thinking_budget = thinking_config.get('budget', -1)
        include_thoughts = thinking_config.get('include_thoughts', False)

        with st.form("thinking_config_form"):
            st.markdown("#### Configuration Options")

            new_thinking_enabled = st.checkbox(
                "Enable Thinking Mode",
                value=thinking_enabled,
                help="Model will perform internal reasoning before generating responses"
            )

            new_include_thoughts = st.checkbox(
                "Include Thoughts in API Response",
                value=include_thoughts,
                help="API responses will include the model's reasoning process"
            )

            budget_options = {
                "Automatic": -1,
                "Disabled": 0,
                "Low (4k)": 4096,
                "Medium (8k)": 8192,
                "High (24k)": 24576,
                "Maximum (32k)": 32768,
                "Custom": "custom"
            }

            current_option = next((k for k, v in budget_options.items() if v == thinking_budget), "Custom")

            selected_option = st.selectbox(
                "Thinking Budget",
                options=list(budget_options.keys()),
                index=list(budget_options.keys()).index(current_option),
                help="Controls the depth of thinking process"
            )

            if selected_option == "Custom":
                new_budget = st.number_input(
                    "Custom Token Count",
                    min_value=-1,
                    max_value=32768,
                    value=thinking_budget if thinking_budget > 0 else 4096
                )
            else:
                new_budget = budget_options[selected_option]

            if st.form_submit_button("Save Configuration", type="primary", use_container_width=True):
                update_data = {
                    "enabled": new_thinking_enabled,
                    "budget": new_budget,
                    "include_thoughts": new_include_thoughts
                }

                result = call_api('/admin/config/thinking', 'POST', data=update_data)
                if result and result.get('success'):
                    st.success("Thinking mode configuration saved!")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to save configuration")

        with st.expander("Current Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mode", "Enabled" if thinking_enabled else "Disabled")
                st.metric("Budget", f"{thinking_budget} tokens" if thinking_budget >= 0 else "Automatic")
            with col2:
                st.metric("Include Thoughts", "Yes" if include_thoughts else "No")

    with tab2:
        st.markdown("### Prompt Injection")
        st.markdown("Automatically add custom instructions to all API requests.")

        inject_config = stats_data.get('inject_config', {})

        inject_enabled = inject_config.get('enabled', False)
        inject_content = inject_config.get('content', '')
        inject_position = inject_config.get('position', 'system')

        with st.form("inject_prompt_form"):
            st.markdown("#### Configuration Options")

            new_inject_enabled = st.checkbox(
                "Enable Prompt Injection",
                value=inject_enabled,
                help="All requests will include your custom prompt"
            )

            position_options = {
                'system': 'As System Message',
                'user_prefix': 'Before User Message',
                'user_suffix': 'After User Message'
            }

            new_position = st.selectbox(
                "Injection Position",
                options=list(position_options.keys()),
                format_func=lambda x: position_options[x],
                index=list(position_options.keys()).index(inject_position)
            )

            new_content = st.text_area(
                "Custom Prompt Content",
                value=inject_content,
                height=150,
                placeholder="You are a professional AI assistant...",
                help="This content will be added to all API requests"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Save Configuration", type="primary", use_container_width=True):
                    update_data = {
                        "enabled": new_inject_enabled,
                        "content": new_content,
                        "position": new_position
                    }

                    result = call_api('/admin/config/inject-prompt', 'POST', data=update_data)
                    if result and result.get('success'):
                        st.success("Prompt injection configuration saved!")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to save configuration")

            with col2:
                if st.form_submit_button("Clear Content", type="secondary", use_container_width=True):
                    clear_data = {
                        "enabled": False,
                        "content": "",
                        "position": "system"
                    }

                    result = call_api('/admin/config/inject-prompt', 'POST', data=clear_data)
                    if result and result.get('success'):
                        st.success("Prompt content cleared!")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to clear content")

        with st.expander("Current Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Status", "Enabled" if inject_enabled else "Disabled")
                st.metric("Position", inject_position)
            with col2:
                content_preview = inject_content[:50] + "..." if len(inject_content) > 50 else inject_content
                st.metric("Content Preview", content_preview if content_preview else "None")

    with tab3:
        st.markdown("### System Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Service Information")
            st.metric("Python Version", status_data.get('python_version', 'Unknown').split()[0])
            st.metric("Service Version", status_data.get('version', '1.0.0'))
            st.metric("Keep Alive", "Active" if status_data.get('keep_alive_active', False) else "Inactive")

        with col2:
            st.markdown("#### Supported Models")
            models = status_data.get('models', [])
            for model in models:
                st.markdown(f"• {model}")

        st.markdown("### System Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            memory_mb = status_data.get('memory_usage_mb', 0)
            st.metric("Memory Usage", f"{memory_mb:.1f} MB")

        with col2:
            cpu_percent = status_data.get('cpu_percent', 0)
            st.metric("CPU Usage", f"{cpu_percent:.1f}%")

        with col3:
            uptime = status_data.get('uptime_seconds', 0)
            uptime_hours = uptime / 3600
            st.metric("Uptime", f"{uptime_hours:.1f} hours")

# --- 页脚 ---
st.markdown(
    f"""
    <div style='text-align: center; color: #9ca3af; font-size: 0.75rem; margin-top: 4rem; padding: 2rem 0; border-top: 1px solid #e5e7eb;'>
        Gemini API Gateway | 
        <a href='{API_BASE_URL}/docs' target='_blank' style='color: #9ca3af;'>API Docs</a> | 
        <a href='{API_BASE_URL}/health' target='_blank' style='color: #9ca3af;'>Health Check</a> | 
        <span style='color: #9ca3af;'>API Endpoint: {API_BASE_URL}</span>
    </div>
    """,
    unsafe_allow_html=True
)