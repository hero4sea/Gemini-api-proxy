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
    page_title="Gemini API 轮询",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API配置 ---
# 支持本地和远程API
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

# 如果是Streamlit Cloud环境，需要配置远程API地址
if 'streamlit.io' in os.getenv('STREAMLIT_SERVER_HEADLESS', ''):
    # Streamlit Cloud环境，使用环境变量中的API地址
    API_BASE_URL = os.getenv('API_BASE_URL', 'https://your-app.onrender.com')

st.markdown(f"**🌐 API 地址**: {API_BASE_URL}")


# --- API调用函数 ---
def call_api(endpoint: str, method: str = 'GET', data: Any = None, timeout: int = 30) -> Optional[Dict]:
    """统一API调用函数"""
    url = f"{API_BASE_URL}{endpoint}"

    try:
        spinner_message = "正在请求数据..." if method == 'GET' else "正在保存更改..."
        with st.spinner(spinner_message):
            if method == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API调用失败: {response.status_code} - {response.text}")
                return None

    except requests.exceptions.Timeout:
        st.error("⏰ API调用超时，服务可能正在唤醒中，请稍后重试...")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 无法连接到API服务，请检查服务状态")
        return None
    except Exception as e:
        st.error(f"❌ API调用异常: {str(e)}")
        return None


def wake_up_service():
    """唤醒服务"""
    try:
        response = requests.get(f"{API_BASE_URL}/wake", timeout=10)
        if response.status_code == 200:
            st.success("✅ 服务已唤醒")
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
@st.cache_data(ttl=30)  # 缓存30秒
def get_cached_stats():
    """获取缓存的统计数据"""
    return call_api('/admin/stats')


@st.cache_data(ttl=60)  # 缓存60秒
def get_cached_status():
    """获取缓存的服务状态"""
    return call_api('/status')


@st.cache_data(ttl=30)  # 缓存30秒
def get_cached_model_config(model_name: str):
    """获取缓存的模型配置"""
    return call_api(f'/admin/models/{model_name}')


@st.cache_data(ttl=30)  # 缓存30秒
def get_cached_gemini_keys():
    """获取缓存的Gemini密钥"""
    return call_api('/admin/gemini-keys')


@st.cache_data(ttl=30)  # 缓存30秒
def get_cached_user_keys():
    """获取缓存的用户密钥"""
    return call_api('/admin/user-keys')


# --- 自定义CSS样式 ---
st.markdown("""
<style>
    /* 全局字体优化 */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro SC", "SF Pro Display", "Helvetica Neue", "PingFang SC", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* 优化整体布局 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    /* 度量卡片 */
    [data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 1.25rem;
        border-radius: 16px;
        border: 1px solid rgba(0, 0, 0, 0.04);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    [data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        transform: translateY(-1px);
    }

    /* 按钮样式 */
    .stButton > button {
        border-radius: 12px;
        font-weight: 500;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        font-size: 0.9375rem;
        padding: 0.625rem 1.25rem;
        letter-spacing: -0.01em;
        box-shadow: none;
    }

    /* Primary按钮 */
    .stButton > button[kind="primary"] {
        background-color: #000;
        color: #fff;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #333;
    }

    /* 输入框样式 */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextArea > div > div > textarea {
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 10px;
        font-size: 0.9375rem;
        padding: 0.75rem 1rem;
        background-color: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* 标签页样式 */
    .stTabs [data-testid="stTabBar"] {
        gap: 7rem;
        border-bottom: 1px solid rgba(0, 0, 0, 0.06);
        padding: 0;
        margin-bottom: 1rem;
    }

    .stTabs [data-testid="stTabBar"] button {
        font-weight: 500;
        color: #86868b;
        padding-bottom: 1rem;
        border-bottom: 2px solid transparent;
        font-size: 0.9375rem;
        letter-spacing: -0.01em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .stTabs [data-testid="stTabBar"] button[aria-selected="true"] {
        color: #000;
        border-bottom-color: #000;
    }

    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background-color: rgba(246, 246, 246, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0, 0, 0, 0.06);
    }

    /* 成功/错误消息样式 */
    .stAlert[data-baseweb="notification"][aria-label*="success"] {
        background-color: rgba(52, 199, 89, 0.08);
        color: #34c759;
        border-radius: 10px;
    }

    .stAlert[data-baseweb="notification"][aria-label*="error"] {
        background-color: rgba(255, 59, 48, 0.08);
        color: #ff3b30;
        border-radius: 10px;
    }

    .stAlert[data-baseweb="notification"][aria-label*="warning"] {
        background-color: rgba(255, 149, 0, 0.08);
        color: #ff9500;
        border-radius: 10px;
    }

    .stAlert[data-baseweb="notification"][aria-label*="info"] {
        background-color: rgba(0, 122, 255, 0.08);
        color: #0066cc;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏 ---
with st.sidebar:
    st.markdown("### Gemini API 轮询")
    st.markdown("---")

    page = st.radio(
        "导航",
        ["概览", "模型", "密钥", "设置"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # 服务状态检查
    st.markdown("#### 服务状态")

    # 添加唤醒按钮
    if st.button("🔄 刷新状态"):
        # 清除缓存
        st.cache_data.clear()

    if st.button("☕ 唤醒服务"):
        wake_up_service()

    # 检查服务健康状态
    health = check_service_health()
    if health:
        st.success("✅ API 服务正常")
        with st.expander("服务详情"):
            st.code(f"地址: {API_BASE_URL}")
            st.code(f"状态: {health.get('status', 'unknown')}")
            st.code(f"运行时间: {health.get('uptime_seconds', 0)}秒")
            if 'request_count' in health:
                st.code(f"总请求: {health['request_count']}")
    else:
        st.error("❌ API 服务离线")
        st.info("💡 点击'唤醒服务'按钮尝试启动")

    st.markdown("---")

    # 快速统计
    st.markdown("#### 系统状态")
    status_data = get_cached_status()
    if status_data:
        st.metric("可用密钥", status_data.get('active_keys', 0))
        thinking_enabled = status_data.get('thinking_enabled', False)
        st.metric("思考模式", "开启" if thinking_enabled else "关闭")

        # 显示内存使用
        memory_mb = status_data.get('memory_usage_mb', 0)
        if memory_mb > 0:
            st.metric("内存使用", f"{memory_mb:.1f}MB")

# --- 主页面内容 ---
if page == "概览":
    st.title("📊 服务概览")
    st.markdown("监控 API 网关性能和使用指标")

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
        st.error("❌ 无法获取服务数据，请检查API服务是否正常运行")
        st.info("💡 尝试点击侧边栏的'唤醒服务'按钮")
        st.stop()

    # 核心指标
    st.markdown("## 核心指标")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gemini_keys = stats_data.get('active_gemini_keys', 0)
        total_gemini = stats_data.get('gemini_keys', 0)
        st.metric(
            "Gemini 密钥",
            gemini_keys,
            delta=f"共 {total_gemini} 个"
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
        thinking_status = "已启用" if status_data.get('thinking_enabled', False) else "已禁用"
        st.metric("思考功能", thinking_status)

    # 系统状态
    st.markdown("## 系统状态")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        uptime = status_data.get('uptime_seconds', 0)
        uptime_hours = uptime / 3600
        st.metric("运行时间", f"{uptime_hours:.1f}小时")

    with col2:
        memory_mb = status_data.get('memory_usage_mb', 0)
        st.metric("内存使用", f"{memory_mb:.1f}MB")

    with col3:
        cpu_percent = status_data.get('cpu_percent', 0)
        st.metric("CPU使用", f"{cpu_percent:.1f}%")

    with col4:
        total_requests = status_data.get('total_requests', 0)
        st.metric("总请求数", f"{total_requests:,}")

    # 使用率分析
    st.markdown("## 使用率分析")

    usage_stats = stats_data.get('usage_stats', {})
    if usage_stats and models:
        # 准备数据
        model_data = []
        for model in models:
            stats = usage_stats.get(model, {'minute': {'requests': 0}, 'day': {'requests': 0}})

            # 从API获取模型配置
            model_config_data = get_cached_model_config(model)
            if not model_config_data:
                # 使用默认值
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
                    marker_color='#007aff',
                    hovertemplate='<b>%{x}</b><br>使用率: %{y:.1f}%<br>当前: %{customdata[0]:,}<br>限制: %{customdata[1]:,}<extra></extra>',
                    customdata=df[['RPM Used', 'RPM Limit']].values
                ))
                fig_rpm.update_layout(
                    title={
                        'text': "每分钟请求数 (RPM)",
                        'font': {'size': 16, 'color': '#000', 'family': '-apple-system, BlinkMacSystemFont'}
                    },
                    yaxis_title="使用率 (%)",
                    yaxis_range=[0, max(100, df['RPM %'].max() * 1.2) if len(df) > 0 else 100],
                    height=350,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': '-apple-system, BlinkMacSystemFont', 'color': '#000'},
                    yaxis={'gridcolor': 'rgba(0,0,0,0.06)', 'zerolinecolor': 'rgba(0,0,0,0.06)'},
                    xaxis={'linecolor': 'rgba(0,0,0,0.06)'},
                    bargap=0.3
                )
                st.plotly_chart(fig_rpm, use_container_width=True)

            with col2:
                fig_rpd = go.Figure()
                fig_rpd.add_trace(go.Bar(
                    x=df['Model'],
                    y=df['RPD %'],
                    text=[f"{x:.1f}%" for x in df['RPD %']],
                    textposition='outside',
                    marker_color='#34c759',
                    hovertemplate='<b>%{x}</b><br>使用率: %{y:.1f}%<br>当前: %{customdata[0]:,}<br>限制: %{customdata[1]:,}<extra></extra>',
                    customdata=df[['RPD Used', 'RPD Limit']].values
                ))
                fig_rpd.update_layout(
                    title={
                        'text': "每日请求数 (RPD)",
                        'font': {'size': 16, 'color': '#000', 'family': '-apple-system, BlinkMacSystemFont'}
                    },
                    yaxis_title="使用率 (%)",
                    yaxis_range=[0, max(100, df['RPD %'].max() * 1.2) if len(df) > 0 else 100],
                    height=350,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': '-apple-system, BlinkMacSystemFont', 'color': '#000'},
                    yaxis={'gridcolor': 'rgba(0,0,0,0.06)', 'zerolinecolor': 'rgba(0,0,0,0.06)'},
                    xaxis={'linecolor': 'rgba(0,0,0,0.06)'},
                    bargap=0.3
                )
                st.plotly_chart(fig_rpd, use_container_width=True)

            # 详细数据表
            with st.expander("查看详细数据"):
                display_df = df[['Model', 'RPM Used', 'RPM Limit', 'RPM %', 'RPD Used', 'RPD Limit', 'RPD %']].copy()
                display_df.columns = ['模型', '分钟请求', '分钟限制', '分钟使用率', '日请求', '日限制', '日使用率']
                display_df['分钟使用率'] = display_df['分钟使用率'].apply(lambda x: f"{x:.1f}%")
                display_df['日使用率'] = display_df['日使用率'].apply(lambda x: f"{x:.1f}%")
                display_df['分钟请求'] = display_df['分钟请求'].apply(lambda x: f"{x:,}")
                display_df['分钟限制'] = display_df['分钟限制'].apply(lambda x: f"{x:,}")
                display_df['日请求'] = display_df['日请求'].apply(lambda x: f"{x:,}")
                display_df['日限制'] = display_df['日限制'].apply(lambda x: f"{x:,}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("📊 暂无使用数据。请先配置API密钥并发送请求。")

elif page == "密钥":
    st.title("🔑 密钥管理")
    st.markdown("管理 Gemini API 密钥和用户访问令牌")

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
                    st.success("✅ 密钥添加成功！")
                    st.cache_data.clear()  # 清除缓存
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ 添加失败，密钥可能已存在或无效")

        st.divider()

        # 🔥 修复：显示真实Gemini密钥数据
        st.markdown("### 现有密钥")

        # 获取真实的Gemini密钥数据
        gemini_keys_data = get_cached_gemini_keys()

        if gemini_keys_data and gemini_keys_data.get('success'):
            keys = gemini_keys_data.get('keys', [])

            if keys:
                st.info(f"📊 共有 {len(keys)} 个密钥")

                for idx, key in enumerate(keys):
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 1, 1])

                        with col1:
                            st.markdown(f"**#{key['id']}**")

                        with col2:
                            # 显示真实的掩码密钥
                            st.code(key['masked_key'], language=None)

                        with col3:
                            # 显示真实的创建时间
                            created_date = key['created_at'][:10] if key['created_at'] else '未知'
                            st.caption(f"添加于 {created_date}")

                        with col4:
                            # 真实的状态显示和切换
                            is_enabled = key['status'] == 1
                            status_label = "🟢 激活" if is_enabled else "🔴 停用"
                            st.caption(status_label)

                            if st.button(
                                    "停用" if is_enabled else "激活",
                                    key=f"toggle_gemini_{key['id']}",
                                    type="secondary"
                            ):
                                toggle_result = call_api(f'/admin/gemini-keys/{key["id"]}/toggle', 'POST')
                                if toggle_result and toggle_result.get('success'):
                                    st.success("✅ 状态已更新")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("❌ 状态更新失败")

                        with col5:
                            # 删除确认机制
                            confirm_key = f"confirm_delete_gemini_{key['id']}"
                            if st.button("删除", key=f"delete_gemini_{key['id']}", type="secondary"):
                                if st.session_state.get(confirm_key, False):
                                    delete_result = call_api(f'/admin/gemini-keys/{key["id"]}', 'DELETE')
                                    if delete_result and delete_result.get('success'):
                                        st.success("✅ 密钥已删除")
                                        st.cache_data.clear()
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("❌ 删除失败")
                                    # 重置确认状态
                                    st.session_state[confirm_key] = False
                                else:
                                    # 第一次点击，设置确认状态
                                    st.session_state[confirm_key] = True
                                    st.warning("⚠️ 再次点击确认删除")

                        if idx < len(keys) - 1:
                            st.markdown("---")
            else:
                st.info("暂无配置的 Gemini 密钥。请在上方添加你的第一个密钥。")
        else:
            st.error("❌ 无法获取Gemini密钥数据")

    with tab2:
        st.markdown("### 生成访问密钥")

        with st.form("generate_user_key"):
            key_name = st.text_input(
                "密钥描述",
                placeholder="例如：生产环境密钥",
                help="为这个密钥添加一个描述，便于管理"
            )
            submitted = st.form_submit_button("生成密钥", type="primary")

            if submitted:
                result = call_api('/admin/config/user-key', 'POST', {'name': key_name or '未命名密钥'})
                if result and result.get('success'):
                    new_key = result.get('key')
                    st.success("✅ 用户密钥生成成功！")
                    st.warning("⚠️ 请立即保存此密钥，它不会再次显示。")
                    st.code(new_key, language=None)

                    # 使用说明
                    st.markdown("### 使用说明")
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

                    st.cache_data.clear()  # 清除缓存
                else:
                    st.error("❌ 生成失败，请重试")

        st.divider()

        # 🔥 修复：显示真实用户密钥数据
        st.markdown("### 现有密钥")

        # 获取真实的用户密钥数据
        user_keys_data = get_cached_user_keys()

        if user_keys_data and user_keys_data.get('success'):
            keys = user_keys_data.get('keys', [])

            if keys:
                st.info(f"📊 共有 {len(keys)} 个用户密钥")

                # 创建真实数据表
                data = []
                for key in keys:
                    # 格式化时间显示
                    created_date = key['created_at'][:10] if key['created_at'] else '未知'
                    last_used = '从未使用'
                    if key['last_used']:
                        try:
                            # 处理时间格式
                            last_used_date = key['last_used'][:16] if len(key['last_used']) > 16 else key['last_used']
                            last_used = last_used_date.replace('T', ' ')
                        except:
                            last_used = '解析错误'

                    data.append({
                        'ID': key['id'],
                        '描述': key['name'] or '未命名',
                        '密钥预览': key['masked_key'],
                        '状态': '🟢 激活' if key['status'] == 1 else '🔴 停用',
                        '创建时间': created_date,
                        '最后使用': last_used
                    })

                df = pd.DataFrame(data)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'ID': st.column_config.NumberColumn(width='small'),
                        '状态': st.column_config.TextColumn(width='small'),
                        '描述': st.column_config.TextColumn(width='medium'),
                        '密钥预览': st.column_config.TextColumn(width='medium'),
                        '创建时间': st.column_config.TextColumn(width='small'),
                        '最后使用': st.column_config.TextColumn(width='medium')
                    }
                )

                # 真实的密钥操作区
                if keys:  # 只有当有密钥时才显示操作区
                    st.markdown("### 密钥操作")
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        selected_key = st.selectbox(
                            "选择密钥",
                            options=keys,
                            format_func=lambda x: f"密钥 #{x['id']} - {x['name'] or '未命名'}",
                            key="selected_user_key"
                        )

                    with col2:
                        if st.button("切换状态", use_container_width=True):
                            if selected_key:
                                toggle_result = call_api(f'/admin/user-keys/{selected_key["id"]}/toggle', 'POST')
                                if toggle_result and toggle_result.get('success'):
                                    st.success(f"✅ 密钥 #{selected_key['id']} 状态已更新")
                                    st.cache_data.clear()
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("❌ 状态更新失败")

                    with col3:
                        if st.button("删除", type="secondary", use_container_width=True):
                            if selected_key:
                                # 确认删除机制
                                confirm_key = f"confirm_delete_user_{selected_key['id']}"
                                if st.session_state.get(confirm_key, False):
                                    delete_result = call_api(f'/admin/user-keys/{selected_key["id"]}', 'DELETE')
                                    if delete_result and delete_result.get('success'):
                                        st.success(f"✅ 密钥 #{selected_key['id']} 已删除")
                                        st.cache_data.clear()
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("❌ 删除失败")
                                    # 重置确认状态
                                    st.session_state[confirm_key] = False
                                else:
                                    # 第一次点击，设置确认状态
                                    st.session_state[confirm_key] = True
                                    st.warning("⚠️ 再次点击确认删除")
            else:
                st.info("暂无用户密钥。请在上方生成你的第一个访问密钥。")
        else:
            st.error("❌ 无法获取用户密钥数据")

elif page == "模型":
    st.title("🤖 模型配置")
    st.markdown("查看并调整模型状态和使用限制")

    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("❌ 无法获取模型数据")
        st.stop()

    models = status_data.get('models', [])
    if not models:
        st.warning("暂无可用模型")
        st.stop()

    st.info(
        f"当前支持 {len(models)} 个模型。请注意，这里的限制是针对**单个 Gemini API Key** 的，总限制会根据您激活的密钥数量自动倍增。")

    # 🔥 关键改进：为每个模型单独处理配置
    for model in models:
        st.markdown(f"---")
        st.markdown(f"### {model}")

        # 获取当前模型的配置
        current_config = get_cached_model_config(model)
        if not current_config or not current_config.get('success'):
            st.warning(f"无法加载模型 {model} 的配置。")
            continue

        # 创建表单 - 每个模型独立的表单
        with st.form(f"model_config_form_{model}"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("#### 单 Key 限制")
                rpm = st.number_input(
                    "每分钟请求 (RPM)",
                    min_value=1,
                    value=current_config.get('single_api_rpm_limit', 1000),
                    key=f"rpm_{model}"
                )

            with col2:
                st.markdown("#### &nbsp;")  # 占位
                rpd = st.number_input(
                    "每日请求 (RPD)",
                    min_value=1,
                    value=current_config.get('single_api_rpd_limit', 50000),
                    key=f"rpd_{model}"
                )

            with col3:
                st.markdown("#### &nbsp;")  # 占位
                tpm = st.number_input(
                    "每分钟令牌 (TPM)",
                    min_value=1000,
                    value=current_config.get('single_api_tpm_limit', 2000000),
                    key=f"tpm_{model}"
                )

            with col4:
                st.markdown("#### 模型状态")
                status_options = {1: "激活", 0: "禁用"}
                current_status_label = status_options.get(current_config.get('status', 1), "激活")
                new_status_label = st.selectbox(
                    "状态",
                    options=list(status_options.values()),
                    index=list(status_options.values()).index(current_status_label),
                    key=f"status_{model}"
                )

            # 每个模型的独立提交按钮
            submitted = st.form_submit_button(f"💾 保存 {model} 配置", type="primary", use_container_width=True)

            if submitted:
                # 构造要发送的数据
                new_status = 1 if new_status_label == "激活" else 0
                update_data = {
                    "single_api_rpm_limit": rpm,
                    "single_api_rpd_limit": rpd,
                    "single_api_tpm_limit": tpm,
                    "status": new_status
                }

                # 调用API更新
                result = call_api(f'/admin/models/{model}', 'POST', data=update_data)
                if result and result.get('success'):
                    st.success(f"✅ {model} 配置已成功保存！")
                    st.cache_data.clear()  # 清除缓存以便刷新后看到新数据
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ 更新模型 {model} 失败！")

elif page == "设置":
    st.title("⚙️ 设置")
    st.markdown("配置高级功能和系统行为")

    # 🔥 修复：使用真实API数据而不是缓存的stats
    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("❌ 无法获取配置数据")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["思考模式", "提示词注入", "系统"])

    with tab1:
        st.markdown("### 思考模式配置")
        st.markdown("启用内部推理以提高复杂查询的响应质量。")

        thinking_config = stats_data.get('thinking_config', {})

        # 显示当前状态
        thinking_enabled = thinking_config.get('enabled', False)
        thinking_budget = thinking_config.get('budget', -1)
        include_thoughts = thinking_config.get('include_thoughts', False)

        # 🔥 关键改进：添加真正的配置功能
        with st.form("thinking_config_form"):
            st.markdown("#### 配置选项")

            new_thinking_enabled = st.checkbox(
                "启用思考模式",
                value=thinking_enabled,
                help="启用后，模型将在生成响应前进行内部推理"
            )

            new_include_thoughts = st.checkbox(
                "在 API 响应中包含思考过程",
                value=include_thoughts,
                help="启用后，API 响应将包含模型的推理过程"
            )

            # 思考预算选择
            budget_options = {
                "自动": -1,
                "禁用": 0,
                "低 (4k)": 4096,
                "中 (8k)": 8192,
                "flash最高 (24k)": 24576,
                "pro最高 (32k)": 32768,
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

            # 提交按钮
            if st.form_submit_button("💾 保存思考模式配置", type="primary", use_container_width=True):
                update_data = {
                    "enabled": new_thinking_enabled,
                    "budget": new_budget,
                    "include_thoughts": new_include_thoughts
                }

                result = call_api('/admin/config/thinking', 'POST', data=update_data)
                if result and result.get('success'):
                    st.success("✅ 思考模式配置已保存！")
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ 保存失败，请重试")

        # 显示当前状态
        with st.expander("当前配置状态"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("思考模式", "启用" if thinking_enabled else "禁用")
                st.metric("思考预算", f"{thinking_budget} tokens" if thinking_budget >= 0 else "自动")
            with col2:
                st.metric("显示思考过程", "是" if include_thoughts else "否")

    with tab2:
        st.markdown("### 提示词注入")
        st.markdown("自动为所有 API 请求添加自定义指令。")

        inject_config = stats_data.get('inject_config', {})

        # 显示当前状态
        inject_enabled = inject_config.get('enabled', False)
        inject_content = inject_config.get('content', '')
        inject_position = inject_config.get('position', 'system')

        # 🔥 关键改进：添加真正的配置功能
        with st.form("inject_prompt_form"):
            st.markdown("#### 配置选项")

            new_inject_enabled = st.checkbox(
                "启用提示词注入",
                value=inject_enabled,
                help="启用后，所有请求都会包含你的自定义提示词"
            )

            # 注入位置
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

            # Prompt 内容
            new_content = st.text_area(
                "自定义提示词内容",
                value=inject_content,
                height=150,
                placeholder="你是一个专业的 AI 助手...",
                help="这里输入的内容会自动添加到所有API请求中"
            )

            # 提交按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("💾 保存配置", type="primary", use_container_width=True):
                    update_data = {
                        "enabled": new_inject_enabled,
                        "content": new_content,
                        "position": new_position
                    }

                    result = call_api('/admin/config/inject-prompt', 'POST', data=update_data)
                    if result and result.get('success'):
                        st.success("✅ 提示词注入配置已保存！")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ 保存失败，请重试")

            with col2:
                if st.form_submit_button("🗑️ 清除内容", type="secondary", use_container_width=True):
                    clear_data = {
                        "enabled": False,
                        "content": "",
                        "position": "system"
                    }

                    result = call_api('/admin/config/inject-prompt', 'POST', data=clear_data)
                    if result and result.get('success'):
                        st.success("✅ 提示词内容已清除！")
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ 清除失败，请重试")

        # 显示当前状态
        with st.expander("当前配置状态"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("注入状态", "启用" if inject_enabled else "禁用")
                st.metric("注入位置", inject_position)
            with col2:
                content_preview = inject_content[:50] + "..." if len(inject_content) > 50 else inject_content
                st.metric("内容预览", content_preview if content_preview else "无")

    with tab3:
        st.markdown("### 系统配置")

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

        # 系统指标
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
    """
    <div style='text-align: center; color: #86868b; font-size: 0.8125rem; margin-top: 4rem; padding: 2rem 0; border-top: 1px solid rgba(0, 0, 0, 0.06); letter-spacing: -0.01em;'>
        Gemini API 轮询服务 | 
        <a href='{api_url}/docs' target='_blank' style='color: #86868b; text-decoration: none;'>API文档</a> | 
        <a href='{api_url}/health' target='_blank' style='color: #86868b; text-decoration: none;'>健康检查</a>
    </div>
    """.format(api_url=API_BASE_URL),
    unsafe_allow_html=True
)