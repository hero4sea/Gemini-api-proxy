import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import threading
import time
import os
from datetime import datetime

from database import Database
from api_server import run_api_server

# --- 页面配置 ---
st.set_page_config(
    page_title="Gemini API 轮询",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- 初始化与缓存 ---
@st.cache_resource
def get_database():
    """初始化数据库连接"""
    return Database()


@st.cache_resource
def start_api_server():
    """在后台线程中启动API服务器"""

    def run_server():
        port = int(os.environ.get("PORT", 8000))
        run_api_server(port=port)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)  # 等待服务器启动
    return True


db = get_database()

# --- 自定义CSS样式 - 高级感设计 ---
st.markdown("""
<style>
    /* 全局字体优化 - 使用系统字体栈 */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro SC", "SF Pro Display", "Helvetica Neue", "PingFang SC", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* 优化整体布局 */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    /* 极简度量卡片 */
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

    /* 度量值样式 */
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-weight: 500;
        color: #000;
        font-size: 1.75rem;
        letter-spacing: -0.02em;
    }

    /* 度量标签样式 */
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: #86868b;
        font-size: 0.875rem;
        font-weight: 400;
        letter-spacing: -0.01em;
    }

    /* 极简按钮样式 */
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

    /* Primary按钮 - 类似苹果风格 */
    .stButton > button[kind="primary"] {
        background-color: #000;
        color: #fff;
    }

    .stButton > button[kind="primary"]:hover {
        background-color: #333;
    }

    /* Secondary按钮 */
    .stButton > button[kind="secondary"] {
        background-color: #f5f5f7;
        color: #000;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: #e8e8ed;
    }

    /* 输入框样式 - 更加精致 */
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

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #0066cc;
        box-shadow: 0 0 0 4px rgba(0, 102, 204, 0.1);
        outline: none;
    }

    /* 标签页样式 - 更精致，增加更大间距 */
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

    .stTabs [data-testid="stTabBar"] button:hover {
        color: #000;
    }

    /* 容器样式 */
    [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }

    /* Expander样式 - 更精致 */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #000;
        background-color: rgba(0, 0, 0, 0.02);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-size: 0.9375rem;
        letter-spacing: -0.01em;
    }

    .streamlit-expanderHeader:hover {
        background-color: rgba(0, 0, 0, 0.04);
    }

    /* 侧边栏样式 - 类似 macOS */
    section[data-testid="stSidebar"] {
        background-color: rgba(246, 246, 246, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0, 0, 0, 0.06);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Radio 按钮样式优化 */
    .stRadio > div[role="radiogroup"] > label {
        background-color: transparent;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        margin: 0.125rem 0;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 500;
        color: #000;
        font-size: 0.9375rem;
        letter-spacing: -0.01em;
    }

    .stRadio > div[role="radiogroup"] > label:hover {
        background-color: rgba(0, 0, 0, 0.04);
    }

    .stRadio > div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child {
        display: none;
    }

    /* 分隔线样式 - 减少margin */
    hr {
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid rgba(0, 0, 0, 0.06);
    }

    /* 标题样式 - 减少margin */
    h1 {
        font-size: 2.5rem;
        font-weight: 600;
        color: #000;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }

    h2 {
        font-size: 1.75rem;
        font-weight: 600;
        color: #000;
        margin-top: 2rem;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }

    h3 {
        font-size: 1.25rem;
        font-weight: 600;
        color: #000;
        margin-bottom: 0.75rem;
        letter-spacing: -0.02em;
    }

    h4 {
        font-size: 0.8125rem;
        font-weight: 600;
        color: #86868b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.5rem;
    }

    /* 描述文字 */
    p {
        color: #86868b;
        font-size: 0.9375rem;
        line-height: 1.5;
        letter-spacing: -0.01em;
        margin-bottom: 0.75rem;
    }

    /* 代码块样式 */
    code {
        background-color: rgba(0, 0, 0, 0.04);
        padding: 0.125rem 0.375rem;
        border-radius: 6px;
        font-size: 0.875rem;
        color: #000;
        font-family: "SF Mono", Monaco, monospace;
    }

    .stCodeBlock {
        background-color: rgba(0, 0, 0, 0.02);
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 10px;
    }

    /* 信息框样式 */
    .stAlert {
        border-radius: 10px;
        border: none;
        font-size: 0.9375rem;
        padding: 0.75rem 1rem;
        backdrop-filter: blur(10px);
        margin-bottom: 0.75rem;
    }

    .stAlert[data-baseweb="notification"][aria-label*="info"] {
        background-color: rgba(0, 122, 255, 0.08);
        color: #0066cc;
    }

    .stAlert[data-baseweb="notification"][aria-label*="success"] {
        background-color: rgba(52, 199, 89, 0.08);
        color: #34c759;
    }

    .stAlert[data-baseweb="notification"][aria-label*="warning"] {
        background-color: rgba(255, 149, 0, 0.08);
        color: #ff9500;
    }

    .stAlert[data-baseweb="notification"][aria-label*="error"] {
        background-color: rgba(255, 59, 48, 0.08);
        color: #ff3b30;
    }

    /* Toggle开关样式 - 类似 iOS */
    button[kind="secondary"][aria-pressed] {
        background-color: #34c759 !important;
        color: white !important;
        padding: 0.25rem 0.75rem !important;
        font-size: 0.8125rem !important;
        border-radius: 20px !important;
    }

    button[kind="secondary"][aria-pressed="false"] {
        background-color: rgba(0, 0, 0, 0.08) !important;
        color: #86868b !important;
    }

    /* 页脚样式 */
    .footer {
        text-align: center; 
        color: #86868b; 
        font-size: 0.8125rem;
        margin-top: 4rem;
        padding: 2rem 0;
        border-top: 1px solid rgba(0, 0, 0, 0.06);
        letter-spacing: -0.01em;
    }

    /* 数据表格样式 */
    [data-testid="stDataFrame"] {
        border: none;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }

    /* 成功消息样式 */
    div[data-testid="stNotification"] {
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-size: 0.9375rem;
    }

    /* 减少元素间距 */
    .element-container {
        margin-bottom: 0.5rem;
    }

    /* 减少容器内部间距 */
    .stContainer > div {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    /* 减少metric容器之间的间距 */
    [data-testid="stMetricContainer"] {
        margin-bottom: 0.375rem;
    }

    /* 优化小屏幕显示 */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.5rem;
        }

        h1 {
            font-size: 2rem;
        }

        h2 {
            font-size: 1.5rem;
            margin-top: 1.5rem;
        }
    }

    /* 滚动条样式 - 类似 macOS */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 0, 0, 0.3);
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

    # 服务状态
    st.markdown("#### 服务状态")

    # 启动API服务器
    if start_api_server():
        st.success("API 服务运行中")
        with st.expander("连接详情"):
            port = int(os.environ.get("PORT", 8000))
            api_url = f"http://localhost:{port}" if port == 8000 else f"https://your-app.onrender.com"
            st.code(api_url, language=None)
            st.caption("OpenAI 兼容接口")
    else:
        st.error("API 服务离线")

    st.markdown("---")

    # 快速统计
    st.markdown("#### 系统状态")
    active_keys = len([k for k in db.get_all_gemini_keys() if k['status'] == 1])
    total_keys = len(db.get_all_gemini_keys())
    st.metric("活跃密钥", f"{active_keys}/{total_keys}")

    thinking_enabled = db.get_thinking_config()['enabled']
    st.metric("思考模式", "开启" if thinking_enabled else "关闭")

# --- 主页面内容 ---
if page == "概览":
    st.title("概览")
    st.markdown("监控 API 网关性能和使用指标")

    # 刷新按钮
    col1, col2 = st.columns([10, 1])
    with col2:
        if st.button("↻", help="刷新数据", key="refresh_dashboard"):
            st.rerun()

    # 核心指标
    st.markdown("## 核心指标")
    col1, col2, col3, col4 = st.columns(4)

    active_keys = [k for k in db.get_all_gemini_keys() if k['status'] == 1]
    active_user_keys = [k for k in db.get_all_user_keys() if k['status'] == 1]

    with col1:
        st.metric(
            "Gemini 密钥",
            len(active_keys),
            delta=f"共 {len(db.get_all_gemini_keys())} 个"
        )
    with col2:
        st.metric(
            "用户密钥",
            len(active_user_keys),
            delta=f"共 {len(db.get_all_user_keys())} 个"
        )
    with col3:
        st.metric("默认模型", db.get_config('default_model_name', 'gemini-2.5-flash'))
    with col4:
        thinking_status = "已启用" if db.get_thinking_config()['enabled'] else "已禁用"
        st.metric("思考功能", thinking_status)

    # 使用率图表
    st.markdown("## 使用率分析")

    model_configs = [mc for mc in db.get_all_model_configs() if mc['status'] == 1]
    usage_stats = db.get_all_usage_stats()

    if model_configs:
        # 准备数据
        model_data = []
        for config in model_configs:
            model_name = config['model_name']
            stats = usage_stats.get(model_name, {'minute': {'requests': 0}, 'day': {'requests': 0}})

            rpm_used = stats['minute']['requests']
            rpm_limit = config['total_rpm_limit']
            rpm_percent = (rpm_used / rpm_limit * 100) if rpm_limit > 0 else 0

            rpd_used = stats['day']['requests']
            rpd_limit = config['total_rpd_limit']
            rpd_percent = (rpd_used / rpd_limit * 100) if rpd_limit > 0 else 0

            model_data.append({
                'Model': model_name,
                'RPM Used': rpm_used,
                'RPM Limit': rpm_limit,
                'RPM %': rpm_percent,
                'RPD Used': rpd_used,
                'RPD Limit': rpd_limit,
                'RPD %': rpd_percent
            })

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
        st.info("暂无活跃模型。请在模型页面中启用至少一个模型。")

elif page == "模型":
    st.title("模型配置")
    st.markdown("配置速率限制并管理模型可用性")

    # 获取激活的key数量
    active_keys = [k for k in db.get_all_gemini_keys() if k['status'] == 1]
    active_key_count = len(active_keys)

    if active_key_count == 0:
        st.warning("暂无活跃的 Gemini 密钥。请先在密钥页面添加并激活密钥。")
    else:
        st.info(f"速率限制基于 {active_key_count} 个活跃 API 密钥计算。")

    # 模型配置列表
    for config in db.get_all_model_configs():
        with st.container():
            model_name = config['model_name']
            stats = db.get_all_usage_stats().get(
                model_name,
                {'minute': {'requests': 0, 'tokens': 0}, 'day': {'requests': 0, 'tokens': 0}}
            )

            # 模型标题行
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"### {model_name}")
            with col2:
                is_enabled = config['status'] == 1
                if st.toggle("启用", value=is_enabled, key=f"toggle_{model_name}"):
                    if not is_enabled:
                        db.update_model_config(model_name, status=1)
                        st.rerun()
                else:
                    if is_enabled:
                        db.update_model_config(model_name, status=0)
                        st.rerun()

            # 配置详情
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 单密钥限制")
                new_rpm = st.number_input(
                    "每分钟请求数",
                    min_value=1,
                    value=config['single_api_rpm_limit'],
                    key=f"rpm_{model_name}"
                )
                new_tpm = st.number_input(
                    "每分钟令牌数",
                    min_value=1000,
                    value=config['single_api_tpm_limit'],
                    key=f"tpm_{model_name}"
                )
                new_rpd = st.number_input(
                    "每日请求数",
                    min_value=1,
                    value=config['single_api_rpd_limit'],
                    key=f"rpd_{model_name}"
                )

            with col2:
                st.markdown(f"#### 系统总限制")
                st.caption(f"基于 {active_key_count} 个密钥")
                st.metric("总 RPM", f"{config['total_rpm_limit']:,}")
                st.metric("总 TPM", f"{config['total_tpm_limit']:,}")
                st.metric("总 RPD", f"{config['total_rpd_limit']:,}")

            with col3:
                st.markdown("#### 当前使用")
                rpm_usage = stats['minute']['requests']
                rpm_percent = (rpm_usage / config['total_rpm_limit'] * 100) if config['total_rpm_limit'] > 0 else 0
                st.metric("分钟请求", f"{rpm_usage:,}", delta=f"{rpm_percent:.1f}%")

                st.metric("分钟令牌", f"{stats['minute']['tokens']:,}")

                rpd_usage = stats['day']['requests']
                rpd_percent = (rpd_usage / config['total_rpd_limit'] * 100) if config['total_rpd_limit'] > 0 else 0
                st.metric("今日请求", f"{rpd_usage:,}", delta=f"{rpd_percent:.1f}%")

            # 更新按钮
            if st.button(f"更新配置", key=f"update_{model_name}", type="primary"):
                db.update_model_config(
                    model_name,
                    single_api_rpm_limit=new_rpm,
                    single_api_tpm_limit=new_tpm,
                    single_api_rpd_limit=new_rpd
                )
                st.success(f"{model_name} 配置已更新")
                time.sleep(1)
                st.rerun()

            st.divider()

elif page == "密钥":
    st.title("密钥管理")
    st.markdown("管理 Gemini API 密钥和用户访问令牌")

    tab1, tab2 = st.tabs(["Gemini 密钥", "用户密钥"])

    with tab1:
        # 添加新Key
        st.markdown("### 添加新密钥")
        col1, col2 = st.columns([5, 1])
        with col1:
            new_key = st.text_input(
                "Gemini API 密钥",
                type="password",
                placeholder="输入你的 Gemini API 密钥...",
                label_visibility="collapsed"
            )
        with col2:
            if st.button("添加", type="primary", use_container_width=True):
                if new_key:
                    if db.add_gemini_key(new_key):
                        st.success("密钥添加成功")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("添加失败，密钥可能已存在。")
                else:
                    st.warning("请输入有效的 API 密钥。")

        st.divider()

        # 现有Keys列表
        st.markdown("### 现有密钥")
        gemini_keys = db.get_all_gemini_keys()

        if gemini_keys:
            for idx, key in enumerate(gemini_keys):
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 1, 1])
                    with col1:
                        st.markdown(f"**#{key['id']}**")
                    with col2:
                        masked_key = f"{key['key'][:20]}{'•' * 20}{key['key'][-10:]}"
                        st.code(masked_key, language=None)
                    with col3:
                        st.caption(f"添加于 {key['created_at'][:10]}")
                    with col4:
                        is_enabled = key['status'] == 1
                        status_label = "激活" if is_enabled else "停用"
                        if st.toggle(status_label, value=is_enabled, key=f"toggle_gemini_{key['id']}"):
                            if not is_enabled:
                                db.update_gemini_key(key['id'], status=1)
                                st.rerun()
                        else:
                            if is_enabled:
                                db.update_gemini_key(key['id'], status=0)
                                st.rerun()
                    with col5:
                        if st.button("删除", key=f"delete_gemini_{key['id']}", type="secondary"):
                            db.delete_gemini_key(key['id'])
                            st.rerun()

                if idx < len(gemini_keys) - 1:
                    st.markdown("---")
        else:
            st.info("暂无配置的 Gemini 密钥。请在上方添加你的第一个密钥。")

    with tab2:
        # 生成新的用户Key
        st.markdown("### 生成访问密钥")
        col1, col2 = st.columns([4, 1])
        with col1:
            key_name = st.text_input(
                "密钥描述",
                placeholder="例如：生产环境密钥",
                label_visibility="collapsed"
            )
        with col2:
            if st.button("生成", type="primary", use_container_width=True):
                new_key_val = db.generate_user_key(key_name)
                st.session_state.latest_generated_key = new_key_val

        # 显示最新生成的Key
        if 'latest_generated_key' in st.session_state:
            st.warning("请立即保存此密钥，它不会再次显示。")
            st.code(st.session_state.latest_generated_key, language=None)
            if st.button("我已保存", use_container_width=True):
                del st.session_state.latest_generated_key
                st.rerun()

        st.divider()

        # 用户Keys列表
        st.markdown("### 现有密钥")
        user_keys = db.get_all_user_keys()

        if user_keys:
            # 创建数据表
            data = []
            for k in user_keys:
                data.append({
                    'ID': k['id'],
                    '描述': k['name'] or '-',
                    '密钥预览': f"{k['key'][:15]}...",
                    '状态': '激活' if k['status'] == 1 else '停用',
                    '创建时间': k['created_at'][:10],
                    '最后使用': k['last_used'][:16] if k['last_used'] else '从未'
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

            # 操作区
            st.markdown("### 密钥操作")
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                selected_id = st.selectbox(
                    "选择密钥",
                    options=[k['id'] for k in user_keys],
                    format_func=lambda
                        x: f"密钥 #{x} - {next((k['name'] or '未命名' for k in user_keys if k['id'] == x), '未知')}"
                )
            with col2:
                if st.button("切换状态", use_container_width=True):
                    db.toggle_user_key_status(selected_id)
                    st.success(f"密钥 #{selected_id} 状态已更新")
                    time.sleep(1)
                    st.rerun()
            with col3:
                if st.button("删除", type="secondary", use_container_width=True):
                    db.delete_user_key(selected_id)
                    st.success(f"密钥 #{selected_id} 已删除")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("暂无用户密钥。请在上方生成你的第一个访问密钥。")

elif page == "设置":
    st.title("设置")
    st.markdown("配置高级功能和系统行为")

    tab1, tab2, tab3 = st.tabs(["思考模式", "提示词注入", "系统"])

    with tab1:
        st.markdown("### 思考模式配置")
        st.markdown("启用内部推理以提高复杂查询的响应质量。")

        thinking_config = db.get_thinking_config()

        # 启用开关
        new_thinking_enabled = st.checkbox(
            "启用思考模式",
            value=thinking_config['enabled'],
            help="启用后，模型将在生成响应前进行内部推理"
        )

        if new_thinking_enabled != thinking_config['enabled']:
            db.set_thinking_config(enabled=new_thinking_enabled)
            st.success("思考模式设置已更新")
            time.sleep(1)
            st.rerun()

        if new_thinking_enabled:
            st.divider()

            # 显示思考过程
            new_include_thoughts = st.checkbox(
                "在 API 响应中包含思考过程",
                value=thinking_config['include_thoughts'],
                help="启用后，API 响应将包含模型的推理过程"
            )

            if new_include_thoughts != thinking_config['include_thoughts']:
                db.set_thinking_config(include_thoughts=new_include_thoughts)
                st.success("思考显示设置已更新")
                time.sleep(1)
                st.rerun()

            st.divider()

            # 思考预算
            st.markdown("#### 令牌预算")

            budget_options = {
                "自动": -1,
                "禁用": 0,
                "低 (4k)": 4096,
                "中 (8k)": 8192,
                "flash最高 (24k)": 24576,
                "pro最高 (32k)": 32768,
                "自定义": "custom"
            }

            current_budget = thinking_config['budget']
            current_option = next((k for k, v in budget_options.items() if v == current_budget), "自定义")

            col1, col2 = st.columns([3, 2])
            with col1:
                selected_option = st.selectbox(
                    "预算预设",
                    options=list(budget_options.keys()),
                    index=list(budget_options.keys()).index(current_option)
                )

            if selected_option == "自定义":
                with col2:
                    new_budget = st.number_input(
                        "自定义令牌数",
                        min_value=-1,
                        max_value=32768,
                        value=current_budget if current_budget > 0 else 4096
                    )
            else:
                new_budget = budget_options[selected_option]

            if st.button("更新令牌预算", type="primary"):
                db.set_thinking_config(budget=new_budget)
                st.success("令牌预算已更新")
                time.sleep(1)
                st.rerun()

    with tab2:
        st.markdown("### 提示词注入")
        st.markdown("自动为所有 API 请求添加自定义指令。")

        inject_config = db.get_inject_prompt_config()

        # 启用开关
        new_inject_enabled = st.checkbox(
            "启用提示词注入",
            value=inject_config['enabled'],
            help="启用后，所有请求都会包含你的自定义提示词"
        )

        if new_inject_enabled != inject_config['enabled']:
            db.set_inject_prompt_config(enabled=new_inject_enabled)
            st.success("提示词注入设置已更新")
            time.sleep(1)
            st.rerun()

        if new_inject_enabled:
            st.divider()

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
                index=list(position_options.keys()).index(inject_config['position'])
            )

            if new_position != inject_config['position']:
                db.set_inject_prompt_config(position=new_position)
                st.success("注入位置已更新")
                time.sleep(1)
                st.rerun()

            st.divider()

            # Prompt 内容
            st.markdown("#### 提示词内容")
            new_content = st.text_area(
                "输入你的自定义提示词",
                value=inject_config['content'],
                height=150,
                placeholder="你是一个专业的 AI 助手...",
                label_visibility="collapsed"
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("保存提示词", type="primary", use_container_width=True):
                    db.set_inject_prompt_config(content=new_content)
                    st.success("提示词内容已保存")
                    time.sleep(1)
                    st.rerun()
            with col2:
                if st.button("清除提示词", type="secondary", use_container_width=True):
                    db.set_inject_prompt_config(content="")
                    st.success("提示词内容已清除")
                    time.sleep(1)
                    st.rerun()

    with tab3:
        st.markdown("### 系统配置")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 模型设置")

            # 默认模型
            current_model = db.get_config('default_model_name', 'gemini-2.5-flash')
            supported_models = db.get_supported_models()
            new_model = st.selectbox(
                "默认模型",
                options=supported_models,
                index=supported_models.index(current_model),
                help="请求未指定模型时使用的默认模型"
            )
            if new_model != current_model:
                db.set_config('default_model_name', new_model)
                st.success("默认模型已更新")
                time.sleep(1)
                st.rerun()

            # 负载均衡策略
            current_strategy = db.get_config('load_balance_strategy', 'least_used')
            strategy_options = {
                'least_used': '最少使用优先',
                'round_robin': '轮询'
            }
            new_strategy = st.selectbox(
                "负载均衡",
                options=list(strategy_options.keys()),
                format_func=lambda x: strategy_options[x],
                index=0 if current_strategy == 'least_used' else 1,
                help="在多个 API 密钥间分配请求的策略"
            )
            if new_strategy != current_strategy:
                db.set_config('load_balance_strategy', new_strategy)
                st.success("负载均衡策略已更新")
                time.sleep(1)
                st.rerun()

        with col2:
            st.markdown("#### 网络设置")

            # 请求超时
            current_timeout = int(db.get_config('request_timeout', '60'))
            new_timeout = st.number_input(
                "请求超时（秒）",
                min_value=10,
                max_value=300,
                value=current_timeout,
                step=10,
                help="等待 API 响应的最长时间"
            )
            if new_timeout != current_timeout:
                db.set_config('request_timeout', str(new_timeout))
                st.success("超时设置已更新")
                time.sleep(1)
                st.rerun()

            # 最大重试次数
            current_retries = int(db.get_config('max_retries', '3'))
            new_retries = st.number_input(
                "最大重试次数",
                min_value=1,
                max_value=10,
                value=current_retries,
                help="请求失败时的重试次数"
            )
            if new_retries != current_retries:
                db.set_config('max_retries', str(new_retries))
                st.success("重试设置已更新")
                time.sleep(1)
                st.rerun()

# --- 页脚 ---
st.markdown(
    """
    <div class='footer'>
        Gemini API 轮询
    </div>
    """,
    unsafe_allow_html=True
)