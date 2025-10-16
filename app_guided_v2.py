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

# ---------------- 页面设置 ----------------
st.set_page_config(
    page_title="RUC_AHP–EW 综合评价系统",
    layout="wide"
)

# ========== 页面横幅 ==========

banner_path = "ESG_banner.png"  # 你的横幅图片放在 app 同目录下

if os.path.exists(banner_path):
    # 展示横幅图片（铺满宽度）
    st.image(banner_path, use_container_width=True)
else:
    # 如果图片缺失，提示信息
    st.warning("⚠️ 未找到 ESG_banner.png，请将横幅图片放在当前目录下。")

# 分割线
st.markdown("<hr style='border: 1px solid #888;'>", unsafe_allow_html=True)

# ---------------- 原主标题（可保留） ----------------
st.title("RUC_AHP–EW 综合评价系统")


# ---------------- Utilities ----------------
RI_TABLE = {1:0.00,2:0.00,3:0.58,4:0.90,5:1.12,6:1.24,7:1.32,8:1.41,9:1.45,10:1.49,11:1.51,12:1.48}

def eig_weight_consistency(A):
    vals, vecs = np.linalg.eig(A)
    idx = np.argmax(np.real(vals))
    lam = np.real(vals[idx])
    w = np.real(vecs[:, idx])
    w = np.maximum(w, 0)
    w = w / (w.sum() if w.sum()>0 else 1.0)
    n = A.shape[0]
    CI = (lam - n) / (n - 1) if n>1 else 0.0
    RI = RI_TABLE.get(n, 1.49)
    CR = CI/RI if RI>0 else 0.0
    return w, lam, CI, CR

def row_arith_weight(A):
    col_sum = A.sum(axis=0); col_sum[col_sum==0] = 1.0
    N = A/col_sum
    w = N.mean(axis=1)
    w = np.maximum(w, 0); w = w/(w.sum() if w.sum()>0 else 1.0)
    return w

def row_geo_weight(A):
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.prod(A, axis=1)**(1.0/A.shape[1])
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    w = g/(g.sum() if g.sum()>0 else 1.0)
    return w

def ruc_fuse(wE, wA, wG, CR, prior=(0.4,0.3,0.3), metric="L1", mode="geo"):
    W = np.vstack([wE,wA,wG]); wbar = W.mean(axis=0)
    if metric=="L2":
        d = np.array([1.0+np.linalg.norm(W[0]-wbar,2),
                      1.0+np.linalg.norm(W[1]-wbar,2),
                      1.0+np.linalg.norm(W[2]-wbar,2)])
    else:
        d = np.array([1.0+np.linalg.norm(W[0]-wbar,1),
                      1.0+np.linalg.norm(W[1]-wbar,1),
                      1.0+np.linalg.norm(W[2]-wbar,1)])
    pE,pA,pG = prior
    rE = max(0.0, min(1.0, 1.0-CR/0.10))
    gtilde = (np.array([pE,pA,pG])*np.array([rE,1.0,1.0]))/d
    gamma = gtilde/(gtilde.sum() if gtilde.sum()>0 else 1.0)
    if mode=="lin":
        wf = gamma[0]*wE + gamma[1]*wA + gamma[2]*wG
        wf = np.maximum(wf,0); wf = wf/(wf.sum() if wf.sum()>0 else 1.0)
        return wf, gamma
    else:
        wf = (np.maximum(wE,1e-12)**gamma[0])*(np.maximum(wA,1e-12)**gamma[1])*(np.maximum(wG,1e-12)**gamma[2])
        wf = wf/(wf.sum() if wf.sum()>0 else 1.0)
        return wf, gamma

def normalize_minmax(x):
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    rng = xmax-xmin
    if rng<=0: return np.zeros_like(x)
    return (x-xmin)/(rng+1e-12)

# -------------- Step 0: Load hierarchy --------------
st.sidebar.header("步骤导航")
st.sidebar.markdown("""
## 🧭 步骤导航

**Ⅰ. 数据准备**
- 🗂️ 加载层级关系（hierarchy.csv）

**Ⅱ. 权重计算**
- ⚖️ AHP 两两比较（逐层引导）
- 📈 Entropy Weight（根据企业得分自动计算）

**Ⅲ. 权重融合与结果**
- 🔄 主客观权重融合（AHP × Entropy weight）
- 🏁 综合评分与结果导出
""")

uploaded_hier = st.file_uploader("上传层级关系 CSV（列：level,id,name,parent_id,type；已为你生成 hierarchy.csv）", type=["csv"])
default_path = "hierarchy.csv"
if uploaded_hier is not None:
    df_hier = pd.read_csv(uploaded_hier)
elif os.path.exists(default_path):
    df_hier = pd.read_csv(default_path)
    st.caption("已加载本地 hierarchy.csv")
else:
    st.error("请先上传层级关系CSV。"); st.stop()

# Validate
need_cols = {"level","id","name","parent_id","type"}
if not need_cols.issubset(set([c.lower() for c in df_hier.columns])):
    st.error("CSV 必须包含列：level,id,name,parent_id,type"); st.stop()

# Standardize cols
df_hier.columns = [c.lower() for c in df_hier.columns]
df_hier["level"] = df_hier["level"].astype(int)

L1_nodes = df_hier[df_hier["level"]==1].copy()
L2_nodes = df_hier[df_hier["level"]==2].copy()
L3_nodes = df_hier[df_hier["level"]==3].copy()

st.success(f"已加载层级：一级 {len(L1_nodes)}，二级 {len(L2_nodes)}，三级 {len(L3_nodes)}（叶子指标）。")

# Groupings
children_L1 = {pid: L2_nodes[L2_nodes["parent_id"]==pid]["id"].tolist() for pid in L1_nodes["id"]}
children_L2 = {pid: L3_nodes[L3_nodes["parent_id"]==pid]["id"].tolist() for pid in L2_nodes["id"]}
name_map = df_hier.set_index("id")["name"].to_dict()
type_map = df_hier[df_hier["level"]==3].set_index("id")["type"].to_dict()

# ---------------- 示例文件下载 ----------------
st.subheader("📘 示例文件下载")

# 示例 hierarchy.csv 内容
example_hier = """level,id,name,parent_id,type
1,C1,ESG,,
1,C2,KPI,,
1,C3,DPI,,
2,S1,E,C1,
2,S2,S,C1,
2,S3,G,C1,
3,T1,碳排放,S1,cost
3,T2,水资源管理,S1,cost
3,T3,废物管理与循环利用,S1,benefit
3,T4,化学品管理,S1,benefit
3,T5,可持续面料使用,S1,benefit
3,T6,劳工条件和员工权益,S2,benefit
3,T7,多元化与包容性,S2,benefit
3,T8,供应链透明度与公平交易,S2,benefit
3,T9,教育与培训,S2,benefit
3,T10,企业治理结构与透明度,S3,benefit
3,T11,合规性与法律遵守,S3,benefit
3,T12,数据保护与信息安全,S3,cost
3,T13,风险管理与危机应对,S3,benefit
2,K1,成本效益管理,C2,
2,K2,供应链响应能力,C2,
2,K3,质量管理,C2,
2,K4,安全合规性,C2,
3,T14,成本控制率,K1,cost
3,T15,单位产品成本下降率,K1,benefit
3,T16,成本优化创新实践,K1,benefit
3,T17,货期达成率,K2,benefit
3,T18,供应链中断快速恢复能力,K2,cost
3,T19,供应链敏捷性,K2,benefit
3,T20,产品一致性,K3,benefit
3,T21,客户满意度指数,K3,benefit
3,T22,质量改进方案实施率,K3,benefit
3,T23,安全事故发生频率,K4,cost
3,T24,法规合规检查通过率,K4,benefit
3,T25,安全风险预防措施实施程度,K4,benefit
2,D1,核心能力与领导力,C3,
2,D2,创新与研发能力,C3,
2,D3,运营执行力,C3,
2,D4,供应链管理能力,C3,
3,T26,长期战略规划能力,D1,benefit
3,T27,市场趋势分析与决策能力,D1,benefit
3,T28,风险预见与应对能力,D1,benefit
3,T29,新产品研发周期,D2,cost
3,T30,技术创新投入比例,D2,benefit
3,T31,知识产权保护与利用,D2,benefit
3,T32,资源整合与利用效率,D3,benefit
3,T33,成本控制能力,D3,benefit
3,T34,绩效驱动的团队管理,D3,benefit
3,T35,供应商关系管理能力,D4,benefit
3,T36,库存与物流优化能力,D4,cost
3,T37,供应链协同创新,D4,benefit
"""

# 示例 company_scores_demo.csv 内容
example_scores = """Company,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30,T31,T32,T33,T34,T35,T36,T37
公司A,80,72,88,83,77,92,86,84,79,88,91,73,85,82,87,89,90,75,86,88,85,87,70,89,84,88,85,75,90,88,86,83,82,84,85,78,89
公司B,85,76,90,85,79,90,85,82,81,84,87,70,83,80,85,87,88,74,85,87,82,86,72,88,83,85,84,74,88,86,85,84,83,85,86,80,87
公司C,88,74,91,86,80,93,88,85,82,86,88,72,86,83,88,89,90,76,87,89,84,88,73,89,85,87,86,76,91,87,86,85,84,86,87,82,88
公司D,78,70,85,80,75,88,82,80,77,83,86,68,82,78,83,85,86,70,82,84,80,84,68,85,81,84,82,70,85,83,82,80,79,81,83,76,84
公司E,92,80,94,90,85,96,92,88,85,90,95,76,89,88,92,93,95,80,90,92,89,92,75,93,88,91,90,78,95,91,90,88,87,89,90,84,91
公司F,84,75,89,84,78,91,85,82,80,85,89,70,84,81,86,88,89,73,84,86,82,86,71,87,83,86,84,73,87,85,84,83,82,84,85,79,86
公司G,90,78,93,88,83,95,90,86,83,88,93,74,87,86,90,92,93,78,89,91,87,90,74,91,86,89,88,77,93,89,88,86,85,87,89,83,90
公司H,82,73,87,82,76,89,83,81,78,84,87,69,83,79,84,86,87,72,83,85,81,84,69,85,81,84,83,71,86,84,83,81,80,82,84,77,85
公司I,88,77,92,87,81,94,89,85,82,87,91,73,86,84,89,90,91,76,88,90,85,88,73,89,85,88,86,75,90,87,86,84,83,85,87,81,88
公司J,86,75,90,85,79,92,87,83,80,85,89,71,84,82,87,88,89,74,85,87,83,86,71,87,83,86,84,73,88,85,84,83,82,84,86,79,87
"""

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="📥 下载示例层级文件（hierarchy_demo.csv）",
        data=example_hier.encode("utf-8-sig"),
        file_name="hierarchy_demo.csv",
        mime="text/csv",
        help="用于演示的层级结构文件，可直接上传体验 AHP 计算流程"
    )
with col2:
    st.download_button(
        label="📊 下载示例企业得分文件（company_scores_demo.csv）",
        data=example_scores.encode("utf-8-sig"),
        file_name="company_scores_demo.csv",
        mime="text/csv",
        help="用于演示的企业×三级指标打分数据，可直接上传体验熵权与融合流程"
    )

st.info("💡 如果您还没有准备好的数据，可以先下载上方示例文件，再上传以体验完整流程。")


# -------------- Step 1: AHP pairwise per parent --------------
st.header("步骤一：AHP 两两比较")
priorE = st.slider("RUC-AHP 先验：特征值法", 0.0, 1.0, 0.4, 0.05)
priorA = st.slider("RUC-AHP 先验：算术平均法", 0.0, 1.0, 0.3, 0.05)
priorG = st.slider("RUC-AHP 先验：几何平均法", 0.0, 1.0, 0.3, 0.05)
metric = st.selectbox("偏离度度量", ["L1","L2"], index=0)
mode = st.selectbox("融合方式", ["geo（乘性，推荐）","lin（线性）"], index=0)
mode_key = "geo" if mode.startswith("geo") else "lin"

def render_pairwise(node_ids, title_key):
    n = len(node_ids)
    st.markdown(f"**{title_key}** — 成员：{[name_map[x] for x in node_ids]}")
    keyprefix = "m_"+title_key
    # initialize A in session
    if keyprefix not in st.session_state or st.session_state[keyprefix]["shape"]!=(n,n) or st.session_state[keyprefix]["ids"]!=tuple(node_ids):
        A = np.ones((n,n), dtype=float)
        st.session_state[keyprefix] = {"A":A, "shape":(n,n), "ids":tuple(node_ids)}
    else:
        A = st.session_state[keyprefix]["A"]

    with st.form("form_"+keyprefix):
        cols = st.columns(n+1)
        cols[0].markdown("**i/j**")
        for j in range(n): cols[j+1].markdown(f"**{name_map[node_ids[j]]}**")
        for i in range(n):
            row = st.columns(n+1)
            row[0].write(f"**{name_map[node_ids[i]]}**")
            for j in range(n):
                if i==j:
                    row[j+1].write("1")
                    A[i,j]=1.0
                elif i<j:
                    k = f"{keyprefix}_{i}_{j}"
                    default = float(A[i,j]) if A[i,j]!=1.0 else 1.0
                    v = row[j+1].number_input(f"{name_map[node_ids[i]]} vs {name_map[node_ids[j]]}",
                                              min_value=1.0, max_value=9.0, value=float(default), step=1.0, key=k)
                    A[i,j]=float(v); A[j,i]=1.0/float(v)
                else:
                    row[j+1].write(f"{A[i,j]:.3f}")
        st.form_submit_button("更新矩阵")
    st.session_state[keyprefix]["A"]=A

    # methods & fuse
    wE, lam, CI, CR = eig_weight_consistency(A)
    wA = row_arith_weight(A)
    wG = row_geo_weight(A)
    wf, gamma = ruc_fuse(wE,wA,wG,CR, prior=(priorE,priorA,priorG), metric=metric, mode=mode_key)

    df = pd.DataFrame({
        "id": node_ids,
        "name": [name_map[x] for x in node_ids],
        "E": wE, "A": wA, "G": wG, "fused": wf
    }).set_index("id")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("λ_max", f"{lam:.4f}"); m2.metric("CI", f"{CI:.4f}")
    m3.metric("CR(≤0.10为佳)", f"{CR:.4f}"); m4.metric("n", f"{n}")
    st.dataframe(df[["name","E","A","G","fused"]], use_container_width=True)
    return df["fused"]

# L1 matrix
st.subheader("① 一级：ESG / KPI / DPI 重要性比较")
w_L1 = render_pairwise(L1_nodes["id"].tolist(), "Level1")

# L2 per parent
st.subheader("② 二级：在各自一级之下的比较")
w_L2_local = {}
for pid, children in children_L1.items():
    with st.expander(f"父节点：{name_map[pid]}（包含 {len(children)} 个二级）", expanded=False):
        w = render_pairwise(children, f"Level2_{pid}")
        w_L2_local.update(w.to_dict())

# L3 per parent
st.subheader("③ 三级：在各自二级之下的比较")
w_L3_local = {}
for pid, children in children_L2.items():
    with st.expander(f"父节点（二级）：{name_map[pid]}（包含 {len(children)} 个三级）", expanded=False):
        w = render_pairwise(children, f"Level3_{pid}")
        w_L3_local.update(w.to_dict())

# Compute global AHP weights for leaves
st.subheader("④ 计算三级指标全局 AHP 权重")
# First make series for L1 & L2
wL1 = pd.Series(w_L1.to_dict(), name="wL1")  # id->weight
wL2 = pd.Series(w_L2_local, name="wL2_local")
wL3 = pd.Series(w_L3_local, name="wL3_local")

# Build global mapping
global_ahp = {}
for leaf_id in L3_nodes["id"]:
    parent2 = L3_nodes[L3_nodes["id"]==leaf_id]["parent_id"].iloc[0]
    parent1 = L2_nodes[L2_nodes["id"]==parent2]["parent_id"].iloc[0]
    global_w = wL1[parent1]*wL2[parent2]*wL3[leaf_id]
    global_ahp[leaf_id] = global_w
global_ahp = pd.Series(global_ahp).sort_index()
global_ahp = global_ahp / (global_ahp.sum() if global_ahp.sum()>0 else 1.0)

df_RUC_AHP = pd.DataFrame({
    "id": global_ahp.index,
    "name": [name_map[x] for x in global_ahp.index],
    "type": [type_map.get(x,"benefit") for x in global_ahp.index],
    "RUC_AHP": global_ahp.values
}).set_index("id")
st.dataframe(df_RUC_AHP, use_container_width=True)

# ---------------- Step 2: Upload company scores ----------------
st.header("步骤二：上传企业×三级指标得分")
st.caption("要求：列名为叶子指标 id（如 T1..T37）或中文名称（系统会自动识别），行=公司")
scores_file = st.file_uploader("上传 CSV（Company列 + 37列指标）", type=["csv"], key="scorescsv")
winsor = st.slider("Winsorize 去极值（每端百分比）", 0.0, 20.0, 0.0, 1.0)

if scores_file is None:
    st.info("等待上传公司得分表……")
    st.stop()

df_scores = pd.read_csv(scores_file)
# Detect company col
first_col = df_scores.columns[0]
if first_col.lower() in ["company","firm","企业","公司","name"]:
    df_scores = df_scores.set_index(first_col)
# Try map columns by id or name
cols_by_id = [c for c in df_scores.columns if c in L3_nodes["id"].values]
if len(cols_by_id)==len(L3_nodes):
    S = df_scores[cols_by_id].copy()
else:
    # try by name -> id
    name_to_id = {name_map[i]:i for i in L3_nodes["id"]}
    try_cols = []
    for c in df_scores.columns:
        if c in name_to_id:
            try_cols.append(name_to_id[c])
    if len(try_cols)==len(L3_nodes):
        S = df_scores[ [name_map[i] for i in L3_nodes["id"]] ].copy()
        S.columns = L3_nodes["id"].tolist()
    else:
        st.error("列名需为 T1..T37 或对应中文名称，且必须完整匹配。"); st.stop()

# Winsorize
if winsor>0:
    low, high = winsor, 100-winsor
    for col in S.columns:
        lo = np.nanpercentile(S[col].values, low)
        hi = np.nanpercentile(S[col].values, high)
        S[col] = S[col].clip(lo, hi)

st.write("原始打分（S）")

# 显示时将指标 id 替换为中文名
S_display = S.copy()
S_display.columns = [name_map.get(c, c) for c in S.columns]
st.dataframe(S_display, use_container_width=True)


# ---------------- Step 3: Entropy weights ----------------
st.header("步骤三：Entropy weight（客观权重）")
# Normalize per indicator with direction
Z = pd.DataFrame(index=S.index, columns=S.columns, dtype=float)
for col in S.columns:
    x = S[col].astype(float).values
    z = normalize_minmax(x)
    if type_map.get(col,"benefit")=="cost":
        z = 1.0 - z
    Z[col] = np.maximum(z, 0.0)

st.write("标准化矩阵 Z（用于熵权与加权评分）")

# 显示时将指标 id 替换为中文名
Z_display = Z.copy()
Z_display.columns = [name_map.get(c, c) for c in Z.columns]
st.dataframe(Z_display, use_container_width=True)


m = Z.shape[0]; k = 1.0/np.log(m) if m>1 else 0.0
P = Z.div(Z.sum(axis=0).replace(0.0, np.nan), axis=1).fillna(0.0)
E = -k * (P.replace(0.0, np.nan).applymap(lambda v: v*np.log(v) if v>0 else 0.0)).sum(axis=0).fillna(0.0)
D = 1.0 - E
ENT = (D / (D.sum() if D.sum()>0 else 1.0)).rename("ENT")

# ========= 显示时替换列名 =========
ENT_display = ENT.to_frame().copy()
ENT_display.index = [name_map.get(i, i) for i in ENT_display.index]
st.dataframe(ENT_display, use_container_width=True)


# ---------------- Step 4: Fusion ----------------
st.header("步骤四：主客观权重融合")
alpha = st.slider("AHP 占比 α", 0.0, 1.0, 0.5, 0.05)
ahp_vec = df_RUC_AHP["RUC_AHP"].reindex(Z.columns).values
ent_vec = ENT.reindex(Z.columns).values
w_fused = (np.maximum(ahp_vec,1e-12)**alpha) * (np.maximum(ent_vec,1e-12)**(1-alpha))
w_fused = w_fused / (w_fused.sum() if w_fused.sum()>0 else 1.0)

weights_tbl = pd.DataFrame({
    "id": Z.columns,
    "name": [name_map[i] for i in Z.columns],
    "type": [type_map.get(i,"benefit") for i in Z.columns],
    "RUC_AHP": ahp_vec,
    "ENT": ENT.reindex(Z.columns).values,
    "FUSED": w_fused
}).set_index("id")
st.dataframe(weights_tbl, use_container_width=True)

# ---------------- Step 5: Final scores ----------------
st.header("步骤五：最终公司综合得分与导出")
final_scores = (Z * w_fused).sum(axis=1).to_frame(name="FinalScore").sort_values("FinalScore", ascending=False)
st.dataframe(final_scores, use_container_width=True)

# Downloads
buf_weights = io.BytesIO(weights_tbl.to_csv(index=True).encode("utf-8"))
buf_scores = io.BytesIO(final_scores.to_csv(index=True).encode("utf-8"))
st.download_button("下载 权重表 (CSV)", data=buf_weights, file_name="weights_fused.csv", mime="text/csv")
st.download_button("下载 公司评分 (CSV)", data=buf_scores, file_name="company_scores.csv", mime="text/csv")

with st.expander("方法说明 / Notes"):
    st.markdown("""
- **多层级 AHP**：按父节点逐块两两比较，计算本地权重与 CR；通过 RUC-AHP 稳健融合得到每块的本地权重；层层相乘得到**三级全局 AHP 权重**。
- **Entropy weight**：基于企业×指标得分的标准化矩阵 Z 计算客观权重 ENT。成本型指标自动做方向反转。
- **主客观融合**：`w ∝ (AHP^α)*(ENT^(1-α))`，默认 α=0.5（相乘开平方）。
- **最终得分**：`Score(company) = Σ w_i * Z_{company,i}`。
""")
