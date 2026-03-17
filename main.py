"""
app.py  –  run with:  streamlit run app.py

CSV attesi
  a.csv  →  colonne: j, r, value      (inventario atteso)
  b.csv  →  colonne: i, r, value      (produzione attesa)
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog
from itertools import product

# ── Solver ────────────────────────────────────────────────────────────────────

def solve(a: dict, b: dict, lam: float = 0.1):
    """Ritorna x[(i,j,r)] ottimale, status e valore obiettivo."""
    I = sorted({i for i, _ in b})
    J = sorted({j for j, _ in a})
    R = sorted({r for _, r in a})

    x_idx = {k: n for n, k in enumerate(product(I, J, R))}
    nx = len(x_idx)
    print(f"DEBUG: nx={nx}, nu={len(J)*len(R)}, nv={len(I)*len(R)}, N={nx + len(J)*len(R) + len(I)*len(R)}")
    u_idx = {k: nx + n for n, k in enumerate(product(J, R))}
    nu = len(u_idx)
    v_idx = {k: nx + nu + n for n, k in enumerate(product(I, R))}
    nv = len(v_idx)
    N = nx + nu + nv

    c = np.zeros(N)
    for idx in x_idx.values(): c[idx] = lam
    for idx in u_idx.values(): c[idx] = 1.0
    for idx in v_idx.values(): c[idx] = 1.0

    rows_A, rows_b = [], []
    for j, r in product(J, R):
        row = np.zeros(N)
        for i in I: row[x_idx[i, j, r]] = 1.0
        row[u_idx[j, r]] = -1.0
        rows_A.append(row.copy()); rows_b.append(a.get((j, r), 0))
        row *= -1; row[u_idx[j, r]] = -1.0
        rows_A.append(row); rows_b.append(-a.get((j, r), 0))

    for i, r in product(I, R):
        row = np.zeros(N)
        for j in J: row[x_idx[i, j, r]] = 1.0
        row[v_idx[i, r]] = -1.0
        rows_A.append(row.copy()); rows_b.append(b.get((i, r), 0))
        row *= -1; row[v_idx[i, r]] = -1.0
        rows_A.append(row); rows_b.append(-b.get((i, r), 0))

    res = linprog(c, A_ub=np.array(rows_A), b_ub=np.array(rows_b),
                  bounds=[(0, None)] * N, method="highs")

    x = {k: round(res.x[idx], 4) for k, idx in x_idx.items()}
    return x, res.message, res.fun, I, J, R

# ── Helper: matrice 2D da slice ───────────────────────────────────────────────

def slice_df(x, I, J, R, dim, val):
    if dim == "i (produttore)":
        rows, cols = J, R
        data = [[x.get((val, j, r), 0) for r in R] for j in J]
        return pd.DataFrame(data, index=[f"j={j}" for j in J],
                                  columns=[f"r={r}" for r in R])
    elif dim == "j (destinatario)":
        rows, cols = I, R
        data = [[x.get((i, val, r), 0) for r in R] for i in I]
        return pd.DataFrame(data, index=[f"i={i}" for i in I],
                                  columns=[f"r={r}" for r in R])
    else:  # r
        data = [[x.get((i, j, val), 0) for j in J] for i in I]
        return pd.DataFrame(data, index=[f"i={i}" for i in I],
                                  columns=[f"j={j}" for j in J])

# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Supply Optimizer", layout="wide")
st.title("🌾 Supply Allocation Optimizer")

col1, col2 = st.columns(2)
with col1:
    fa = st.file_uploader("a.csv — inventario (j, r, value)", type="csv")
with col2:
    fb = st.file_uploader("b.csv — produzione (i, r, value)", type="csv")

lam = st.slider("λ sparsità", 0.0, 1.0, 0.1, 0.05)

if fa and fb:
    a_df = pd.read_csv(fa)
    b_df = pd.read_csv(fb)
    a = {(row.j, row.r): row.value for _, row in a_df.iterrows()}
    b = {(row.i, row.r): row.value for _, row in b_df.iterrows()}

    if st.button("▶ Ottimizza"):
        with st.spinner("Solving..."):
            x, msg, obj, I, J, R = solve(a, b, lam)

        st.success(f"{msg}  |  Obiettivo: {obj:.4f}")

        st.subheader("Matrice i-j per ogni r")
        for r in R:
            st.markdown(f"### r = {r}")
            data = [[x.get((i, j, r), 0) for j in J] for i in I]
            df = pd.DataFrame(data, index=[f"i={i}" for i in I], columns=[f"j={j}" for j in J])
            st.dataframe(df.style.background_gradient(cmap="YlGn", axis=None))

        st.subheader("Tutti i flussi x_ijr > 0")
        rows = [{"i": i, "j": j, "r": r, "x": v}
                for (i, j, r), v in x.items() if v > 1e-6]
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("Nessun flusso positivo.")
else:
    st.info("Carica entrambi i CSV per procedere.")
    with st.expander("Formato CSV atteso"):
        st.code("# a.csv\nj,r,value\n0,0,12\n0,1,8\n1,0,10\n...", language="csv")
        st.code("# b.csv\ni,r,value\n0,0,15\n0,1,9\n1,0,11\n...", language="csv")