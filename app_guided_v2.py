
import io, os, math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AHP–熵权 综合评价系统（多层级引导版）", layout="wide")
st.title("AHP–熵权 综合评价系统（多层级引导版）")

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
- 📈 熵权法（根据企业得分自动计算）

**Ⅲ. 权重融合与结果**
- 🔄 主客观权重融合（AHP × 熵权）
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

# -------------- Step 1: AHP pairwise per parent --------------
st.header("步骤一：AHP 两两比较（逐父节点）")
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

df_ahp_global = pd.DataFrame({
    "id": global_ahp.index,
    "name": [name_map[x] for x in global_ahp.index],
    "type": [type_map.get(x,"benefit") for x in global_ahp.index],
    "AHP_global": global_ahp.values
}).set_index("id")
st.dataframe(df_ahp_global, use_container_width=True)

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
st.dataframe(S, use_container_width=True)

# ---------------- Step 3: Entropy weights ----------------
st.header("步骤三：熵权（客观权重）")
# Normalize per indicator with direction
Z = pd.DataFrame(index=S.index, columns=S.columns, dtype=float)
for col in S.columns:
    x = S[col].astype(float).values
    z = normalize_minmax(x)
    if type_map.get(col,"benefit")=="cost":
        z = 1.0 - z
    Z[col] = np.maximum(z, 0.0)

st.write("标准化矩阵 Z（用于熵权与加权评分）")
st.dataframe(Z, use_container_width=True)

m = Z.shape[0]; k = 1.0/np.log(m) if m>1 else 0.0
P = Z.div(Z.sum(axis=0).replace(0.0, np.nan), axis=1).fillna(0.0)
E = -k * (P.replace(0.0, np.nan).applymap(lambda v: v*np.log(v) if v>0 else 0.0)).sum(axis=0).fillna(0.0)
D = 1.0 - E
ENT = (D / (D.sum() if D.sum()>0 else 1.0)).rename("ENT")
st.dataframe(ENT.to_frame(), use_container_width=True)

# ---------------- Step 4: Fusion ----------------
st.header("步骤四：主客观权重融合")
alpha = st.slider("AHP 占比 α", 0.0, 1.0, 0.5, 0.05)
ahp_vec = df_ahp_global["AHP_global"].reindex(Z.columns).values
ent_vec = ENT.reindex(Z.columns).values
w_fused = (np.maximum(ahp_vec,1e-12)**alpha) * (np.maximum(ent_vec,1e-12)**(1-alpha))
w_fused = w_fused / (w_fused.sum() if w_fused.sum()>0 else 1.0)

weights_tbl = pd.DataFrame({
    "id": Z.columns,
    "name": [name_map[i] for i in Z.columns],
    "type": [type_map.get(i,"benefit") for i in Z.columns],
    "AHP_global": ahp_vec,
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
- **熵权法**：基于企业×指标得分的标准化矩阵 Z 计算客观权重 ENT。成本型指标自动做方向反转。
- **主客观融合**：`w ∝ (AHP^α)*(ENT^(1-α))`，默认 α=0.5（相乘开平方）。
- **最终得分**：`Score(company) = Σ w_i * Z_{company,i}`。
""")
