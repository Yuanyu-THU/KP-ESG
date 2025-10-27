import io, os, math
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import base64

# ========== 欢迎界面 ==========
if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    st.markdown("""
        <style>
        html, body, [class*="css"] {
            height: 100%;
            margin: 0;
            padding: 0;
        }

        /* ✅ 让主容器充满屏幕宽度 */
        .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100% !important;
            width: 100vw !important;
        }

        /* 页面整体垂直水平居中 */
        .welcome-wrapper {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100vh;
            width: 100vw;
            text-align: center;
            background: linear-gradient(to bottom, #f6fff9, #ffffff);
        }

        .welcome-title {
            font-size: 40px;
            color: #003366;
            font-weight: 700;
            margin-bottom: 15px;
            max-width: 90vw;
            line-height: 1.3;
        }

        .welcome-subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 60px;
        }

        /* ✅ 按钮区域充满整行 */
        div.stButton {
            width: 100vw !important;
            display: flex;
            justify-content: center;
        }

        /* ✅ 按钮全宽 & 无边框样式 */
        div.stButton > button:first-child {
            width: 100vw;                /* 宽度占满全屏 */
            padding: 1.2em 0;
            font-size: 1.4em;
            font-weight: 600;
            color: white;
            background-color: #3cb371;   /* 草绿色背景 */
            border: none;                /* 去掉边框 */
            border-radius: 0;            /* 去掉圆角 */
            cursor: pointer;
            transition: background 0.3s ease-in-out, transform 0.1s ease-in-out;
        }

        div.stButton > button:first-child:hover {
            background-color: #2e8b57;   /* hover 变深一点 */
            transform: scale(1.01);
        }

        /* ✅ 移动端自适应 */
        @media (max-width: 768px) {
            .welcome-title {
                font-size: 26px;
                padding: 0 5vw;
            }
            .welcome-subtitle {
                font-size: 16px;
                margin-bottom: 40px;
            }
            div.stButton > button:first-child {
                font-size: 1.1em;
                padding: 1em 0;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # 页面主体
    st.markdown("""
        <div class="welcome-wrapper">
            <h1 class="welcome-title">
                Use the RUC_AHP–EW Evaluation System to Score Your Suppliers
            </h1>
            <p class="welcome-subtitle">
                —— Empowering ESG Decision Making with Smart Analytics ——
            </p>
    """, unsafe_allow_html=True)

    # ✅ 全屏按钮版本
    if st.button("Enter 🚀 (Please double-click.)"):
        st.session_state.entered = True

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ===== Demo 主体：保留横幅、标题、分割线、三个按钮，点击后仅提示 Coming Soon 😊 =====
import streamlit as st
from PIL import Image

def show_demo_main():
    # === 恢复自然排版 + 控制宽度 ===
    st.markdown(
        """
        <style>
        /* 保持宽度自然，不强制100% */
        .block-container {
            max-width: 1200px;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        /* 恢复标题字体层次与色彩 */
        h1, h2, h3 {
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
            color: #202020;
        }
        h1 {
            font-size: 2.2em;
            font-weight: 700;
            color: #002B5B;
        }
        h2 {
            color: #004C70;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ==== 侧边栏步骤导航 ====
    st.sidebar.title("步骤导航")
    st.sidebar.markdown("""
    - **Ⅰ. 数据准备 🗂️**
    - **Ⅱ. 权重计算 ⚙️**
    - **Ⅲ. 权重融合与结果 📊**
    """)

    banner_path = "ESG_banner.png"
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)

    st.markdown("<hr style='border: 1px solid #888;'>", unsafe_allow_html=True)
    st.title("RUC_AHP–EW 综合评价系统")

    st.header("Ⅰ. 数据准备")
    st.info("请选择层级结构数据来源：")
    st.write("请选择使用方式：")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("📂 上传自定义层级文件")
    with col2:
        if st.button("📥 下载示范数据查看结构"):
            st.info("Coming Soon！😊")
    with col3:
        if st.button("🚀 直接使用示范数据"):
            st.info("Coming Soon！😊")

# 入口逻辑
if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    # 欢迎页逻辑已在上方执行
    pass
else:
    show_demo_main()
