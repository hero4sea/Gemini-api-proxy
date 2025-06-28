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

st.write(f"🌐 API地址: {API_BASE_URL}")


# --- API调用函数 ---
def call_api(endpoint: str, method: str = 'GET', data: Any = None, timeout: int = 30) -> Optional[Dict]:
    """统一API调用函数"""
    url = f"{API_BASE_URL}{endpoint}"

    try:
        with st.spinner(f"调用 {endpoint}..."):
            if method == 'GET':
                response = requests.get(url, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, timeout=timeout)
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

            # 这里需要从API获取模型配置
            model_config_data = call_api(f'/admin/models/{model}')  # 假设有这个端点
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

        # 显示现有密钥（模拟显示）
        st.markdown("### 现有密钥")
        stats_data = get_cached_stats()
        if stats_data:
            total_keys = stats_data.get('gemini_keys', 0)
            active_keys = stats_data.get('active_gemini_keys', 0)

            if total_keys > 0:
                st.info(f"📊 共有 {total_keys} 个密钥，其中 {active_keys} 个处于激活状态")

                # 创建模拟的密钥列表显示
                for i in range(min(total_keys, 5)):  # 最多显示5个
                    with st.container():
                        col1, col2, col3 = st.columns([1, 4, 1])
                        with col1:
                            st.markdown(f"**#{i + 1}**")
                        with col2:
                            # 模拟显示掩码密钥
                            masked_key = f"AIzaSy{'•' * 30}abc{i + 1:02d}"
                            st.code(masked_key, language=None)
                        with col3:
                            status = "🟢 激活" if i < active_keys else "🔴 禁用"
                            st.markdown(status)

                        if i < total_keys - 1:
                            st.markdown("---")
            else:
                st.info("暂无配置的 Gemini 密钥。请在上方添加你的第一个密钥。")

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

        # 显示现有用户密钥
        st.markdown("### 现有密钥")
        stats_data = get_cached_stats()
        if stats_data:
            total_user_keys = stats_data.get('user_keys', 0)
            active_user_keys = stats_data.get('active_user_keys', 0)

            if total_user_keys > 0:
                st.info(f"📊 共有 {total_user_keys} 个用户密钥，其中 {active_user_keys} 个处于激活状态")

                # 创建模拟的用户密钥列表
                data = []
                for i in range(min(total_user_keys, 10)):  # 最多显示10个
                    data.append({
                        'ID': i + 1,
                        '描述': f'密钥 {i + 1}' if i % 3 != 0 else '生产环境密钥',
                        '密钥预览': f"sk-{'•' * 15}...",
                        '状态': '激活' if i < active_user_keys else '停用',
                        '创建时间': '2024-01-01',
                        '最后使用': '2024-01-15' if i < active_user_keys else '从未'
                    })

                df = pd.DataFrame(data)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'ID': st.column_config.NumberColumn(width='small'),
                        '状态': st.column_config.TextColumn(width='small')
                    }
                )
            else:
                st.info("暂无用户密钥。请在上方生成你的第一个访问密钥。")

elif page == "设置":
    st.title("⚙️ 设置")
    st.markdown("配置高级功能和系统行为")

    # 由于无法直接修改远程配置，这里主要显示当前状态
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

        col1, col2 = st.columns(2)

        with col1:
            st.metric("思考模式", "启用" if thinking_enabled else "禁用")
            st.metric("思考预算", f"{thinking_budget} tokens" if thinking_budget >= 0 else "自动")

        with col2:
            st.metric("显示思考过程", "是" if include_thoughts else "否")

        st.info("💡 要修改思考模式配置，请通过API直接调用或在服务器端修改配置文件。")

        # 显示思考模式说明
        with st.expander("思考模式说明"):
            st.markdown("""
            **思考模式功能：**
            - 启用后，模型会在生成响应前进行内部推理
            - 可以提高复杂问题的回答质量
            - 支持设置思考预算来控制推理深度

            **预算设置：**
            - `-1`: 自动模式，由模型决定思考深度
            - `0`: 禁用思考功能
            - `1-32768`: 固定的思考token预算
            """)

    with tab2:
        st.markdown("### 提示词注入")
        st.markdown("自动为所有 API 请求添加自定义指令。")

        inject_config = stats_data.get('inject_config', {})

        # 显示当前状态
        inject_enabled = inject_config.get('enabled', False)
        inject_content = inject_config.get('content', '')
        inject_position = inject_config.get('position', 'system')

        col1, col2 = st.columns(2)

        with col1:
            st.metric("注入状态", "启用" if inject_enabled else "禁用")
            st.metric("注入位置", inject_position)

        with col2:
            content_preview = inject_content[:50] + "..." if len(inject_content) > 50 else inject_content
            st.metric("内容预览", content_preview if content_preview else "无")

        if inject_content:
            with st.expander("完整注入内容"):
                st.text_area("注入的提示词", inject_content, disabled=True, height=150)

        st.info("💡 要修改提示词注入配置，请通过API直接调用或在服务器端修改配置文件。")

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

elif page == "模型":
    st.title("🤖 模型配置")
    st.markdown("查看模型状态和使用情况")

    stats_data = get_cached_stats()
    status_data = get_cached_status()

    if not stats_data or not status_data:
        st.error("❌ 无法获取模型数据")
        st.stop()

    models = status_data.get('models', [])
    usage_stats = stats_data.get('usage_stats', {})

    if not models:
        st.warning("暂无可用模型")
        st.stop()

    st.info(f"当前支持 {len(models)} 个模型")

    # 显示每个模型的状态
    for model in models:
        with st.container():
            st.markdown(f"### {model}")

            stats = usage_stats.get(model,
                                    {'minute': {'requests': 0, 'tokens': 0}, 'day': {'requests': 0, 'tokens': 0}})

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 分钟级使用")
                st.metric("请求数", f"{stats['minute']['requests']:,}")
                st.metric("令牌数", f"{stats['minute']['tokens']:,}")

            with col2:
                st.markdown("#### 日级使用")
                st.metric("请求数", f"{stats['day']['requests']:,}")
                st.metric("令牌数", f"{stats['day']['tokens']:,}")

            with col3:
                st.markdown("#### 模型特性")
                if '2.5' in model:
                    st.success("✅ 支持思考模式")
                else:
                    st.info("ℹ️ 标准模式")

                if 'flash' in model:
                    st.info("⚡ 快速响应")
                elif 'pro' in model:
                    st.info("🎯 专业版本")

            st.divider()

# --- 页脚 ---
st.markdown(
    """
    <div style='text-align: center; color: #86868b; font-size: 0.8125rem; margin-top: 4rem; padding: 2rem 0; border-top: 1px solid rgba(0, 0, 0, 0.06); letter-spacing: -0.01em;'>
        Gemini API 轮询服务 | 
        <a href='{api_url}' target='_blank' style='color: #86868b; text-decoration: none;'>API文档</a> | 
        <a href='{api_url}/health' target='_blank' style='color: #86868b; text-decoration: none;'>健康检查</a>
    </div>
    """.format(api_url=API_BASE_URL),
    unsafe_allow_html=True
)