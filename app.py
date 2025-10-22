import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import sklearn.compose._column_transformer

# Configuración de la página
st.set_page_config(
    page_title="PREDICTOR DE ÉXITO ACADÉMICO EN EDUCACIÓN EN LINEA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS universales compatibles con todos los navegadores y temas
st.markdown("""
<style>
    /* ===== VARIABLES DE COLOR (FUNCIONAN EN CLARO/OSCURO) ===== */
    :root {
        --primary-color: #2563eb;       /* Azul profesional */
        --primary-light: #3b82f6;
        --primary-dark: #1d4ed8;
        --success-color: #059669;       /* Verde éxito */
        --warning-color: #d97706;       /* Amarillo advertencia */
        --danger-color: #dc2626;        /* Rojo peligro */
        --text-primary: #1f2937;        /* Texto oscuro */
        --text-secondary: #4b5563;      /* Texto secundario */
        --text-light: #6b7280;          /* Texto claro */
        --bg-light: #f8fafc;            /* Fondo claro */
        --bg-card: #ffffff;             /* Fondo tarjetas */
        --border-color: #e5e7eb;        /* Bordes suaves */
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* ===== ESTILOS BASE UNIVERSALES ===== */
    .stApp {
        background-color: var(--bg-light) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .stMarkdown, .stText, .stWrite, .stAlert {
        color: var(--text-primary) !important;
        line-height: 1.6;
    }

    /* ===== HEADERS Y TÍTULOS ===== */
    .main-header {
        font-size: 2.5rem;
        color: var(--primary-color) !important;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 1rem;
    }

    .section-header {
        font-size: 1.4rem;
        color: var(--primary-color) !important;
        border-bottom: 2px solid var(--primary-light);
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
    }

    .subsection-header {
        font-size: 1.1rem;
        color: var(--text-primary) !important;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
    }

    /* ===== COMPONENTES DE LA APLICACIÓN ===== */
    .metric-card {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
        color: var(--text-primary) !important;
    }

    .recommendation-box {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--success-color);
        margin: 1rem 0;
        box-shadow: var(--shadow);
        color: var(--text-primary) !important;
    }

    .risk-high { 
        background: #fef2f2 !important;
        border: 1px solid #fecaca !important;
        border-left: 4px solid var(--danger-color) !important;
        color: var(--text-primary) !important;
    }

    .risk-medium { 
        background: #fffbeb !important;
        border: 1px solid #fed7aa !important;
        border-left: 4px solid var(--warning-color) !important;
        color: var(--text-primary) !important;
    }

    .risk-low { 
        background: #f0fdf4 !important;
        border: 1px solid #bbf7d0 !important;
        border-left: 4px solid var(--success-color) !important;
        color: var(--text-primary) !important;
    }

    /* ===== BADGES Y ELEMENTOS ESPECIALES ===== */
    .rf-badge {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        color: white !important;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: var(--shadow);
        border: none;
        display: inline-block;
        margin: 0.5rem 0;
    }

    .cluster-badge {
        background: linear-gradient(135deg, #2196F3 0%, #03A9F4 100%);
        color: white !important;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: var(--shadow);
        border: none;
        display: inline-block;
        margin: 0.5rem 0;
    }

    .feature-importance-box {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        color: var(--text-primary) !important;
    }

    .cluster-info-box {
        background: rgba(33, 150, 243, 0.08) !important;
        border: 1px solid #2196F3 !important;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2196F3 !important;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        color: var(--text-primary) !important;
    }

    .cluster-detail-box {
        background: rgba(33, 150, 243, 0.05) !important;
        border: 1px solid #2196F3 !important;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        color: var(--text-primary) !important;
    }

    /* ===== GARANTIZAR LEGIBILIDAD DEL TEXTO ===== */
    .recommendation-box h1,
    .recommendation-box h2,
    .recommendation-box h3,
    .recommendation-box h4,
    .recommendation-box h5,
    .recommendation-box h6,
    .recommendation-box p,
    .recommendation-box li,
    .recommendation-box span,
    .recommendation-box div,
    .recommendation-box strong,
    .recommendation-box b {
        color: var(--text-primary) !important;
    }

    .feature-importance-box h1,
    .feature-importance-box h2,
    .feature-importance-box h3,
    .feature-importance-box h4,
    .feature-importance-box h5,
    .feature-importance-box h6,
    .feature-importance-box p,
    .feature-importance-box li,
    .feature-importance-box span,
    .feature-importance-box div,
    .feature-importance-box strong,
    .feature-importance-box b {
        color: var(--text-primary) !important;
    }

    .cluster-info-box h1,
    .cluster-info-box h2,
    .cluster-info-box h3,
    .cluster-info-box h4,
    .cluster-info-box h5,
    .cluster-info-box h6,
    .cluster-info-box p,
    .cluster-info-box li,
    .cluster-info-box span,
    .cluster-info-box div,
    .cluster-info-box strong,
    .cluster-info-box b {
        color: var(--text-primary) !important;
    }

    .cluster-detail-box h1,
    .cluster-detail-box h2,
    .cluster-detail-box h3,
    .cluster-detail-box h4,
    .cluster-detail-box h5,
    .cluster-detail-box h6,
    .cluster-detail-box p,
    .cluster-detail-box li,
    .cluster-detail-box span,
    .cluster-detail-box div,
    .cluster-detail-box strong,
    .cluster-detail-box b {
        color: var(--text-primary) !important;
    }

    /* ===== SIDEBAR (COMPATIBLE CON MODO CLARO/OSCURO) ===== */
    .css-1d391kg, .css-1y4p8pa {
        background-color: var(--bg-card) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    .css-1d391kg p, .css-1y4p8pa p, 
    .css-1d391kg label, .css-1y4p8pa label,
    .css-1d391kg .stMarkdown, .css-1y4p8pa .stMarkdown,
    .css-1d391kg .stRadio, .css-1y4p8pa .stRadio,
    .css-1d391kg .stSelectbox, .css-1y4p8pa .stSelectbox {
        color: var(--text-primary) !important;
    }

    /* ===== FORMULARIOS Y CONTROLES ===== */
    .stSelectbox, .stSlider, .stRadio, .stNumberInput, .stTextInput {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }

    .stSelectbox div, .stSlider div, .stRadio div, .stNumberInput div {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }

    .stSelectbox label, .stSlider label, .stRadio label, .stNumberInput label {
        color: var(--text-primary) !important;
    }

    /* ===== EFECTOS HOVER ===== */
    .metric-card:hover {
        transform: translateY(-2px);
        transition: all 0.2s ease;
        box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.1);
    }

    .recommendation-box:hover {
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }

    /* ===== RESPONSIVE DESIGN ===== */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .section-header {
            font-size: 1.2rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }

    /* ===== ESTILOS ESPECÍFICOS PARA STREAMLIT ===== */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* ===== SCROLLBAR PERSONALIZADO ===== */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-light);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary-light);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }

    /* ===== ESTILOS PARA TABLAS ===== */
    .stDataFrame, .stTable {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: var(--shadow);
    }

    /* ===== ESTILOS PARA EXPANDERS ===== */
    .streamlit-expanderHeader {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px;
    }

    .streamlit-expanderContent {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }

    /* ===== CONTRASTE MEJORADO PARA ACCESIBILIDAD ===== */
    .stAlert {
        border-radius: 8px;
        border: 1px solid var(--border-color);
        color: var(--text-primary) !important;
    }

    .stSuccess {
        background-color: #f0fdf4 !important;
        border-left: 4px solid var(--success-color) !important;
        color: var(--text-primary) !important;
    }

    .stWarning {
        background-color: #fffbeb !important;
        border-left: 4px solid var(--warning-color) !important;
        color: var(--text-primary) !important;
    }

    .stError {
        background-color: #fef2f2 !important;
        border-left: 4px solid var(--danger-color) !important;
        color: var(--text-primary) !important;
    }

    .stInfo {
        background-color: #f0f9ff !important;
        border-left: 4px solid var(--primary-color) !important;
        color: var(--text-primary) !important;
    }

    /* ===== ICONOS Y ELEMENTOS GRÁFICOS ===== */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--primary-light));
    }

    /* ===== TEXTO GENERAL MEJORADO ===== */
    p, li, span, div {
        color: var(--text-primary) !important;
    }

    strong, b {
        color: var(--text-primary) !important;
    }

    /* ===== CORRECCIÓN ESPECÍFICA PARA TEXTO EN ELEMENTOS INTERACTIVOS ===== */
    .st-bb, .st-bc, .st-bd, .st-be, .st-bf, .st-bg, .st-bh, .st-bi, .st-bj, .st-bk, .st-bl, .st-bm, .st-bn, .st-bo, .st-bp, .st-bq, .st-br, .st-bs, .st-bt, .st-bu, .st-bv, .st-bw, .st-bx, .st-by, .st-bz, .st-c0, .st-c1, .st-c2, .st-c3, .st-c4, .st-c5, .st-c6, .st-c7, .st-c8, .st-c9, .st-ca, .st-cb, .st-cc, .st-cd, .st-ce, .st-cf, .st-cg, .st-ch, .st-ci, .st-cj, .st-ck, .st-cl, .st-cm, .st-cn, .st-co, .st-cp, .st-cq, .st-cr, .st-cs, .st-ct, .st-cu, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz, .st-d0, .st-d1, .st-d2, .st-d3, .st-d4, .st-d5, .st-d6, .st-d7, .st-d8, .st-d9, .st-da, .st-db, .st-dc, .st-dd, .st-de, .st-df, .st-dg, .st-dh, .st-di, .st-dj, .st-dk, .st-dl, .st-dm, .st-dn, .st-do, .st-dp, .st-dq, .st-dr, .st-ds, .st-dt, .st-du, .st-dv, .st-dw, .st-dx, .st-dy, .st-dz, .st-e0, .st-e1, .st-e2, .st-e3, .st-e4, .st-e5, .st-e6, .st-e7, .st-e8, .st-e9, .st-ea, .st-eb, .st-ec, .st-ed, .st-ee, .st-ef, .st-eg, .st-eh, .st-ei, .st-ej, .st-ek, .st-el, .st-em, .st-en, .st-eo, .st-ep, .st-eq, .st-er, .st-es, .st-et, .st-eu, .st-ev, .st-ew, .st-ex, .st-ey, .st-ez, .st-f0, .st-f1, .st-f2, .st-f3, .st-f4, .st-f5, .st-f6, .st-f7, .st-f8, .st-f9, .st-fa, .st-fb, .st-fc, .st-fd, .st-fe, .st-ff, .st-fg, .st-fh, .st-fi, .st-fj, .st-fk, .st-fl, .st-fm, .st-fn, .st-fo, .st-fp, .st-fq, .st-fr, .st-fs, .st-ft, .st-fu, .st-fv, .st-fw, .st-fx, .st-fy, .st-fz, .st-g0, .st-g1, .st-g2, .st-g3, .st-g4, .st-g5, .st-g6, .st-g7, .st-g8, .st-g9, .st-ga, .st-gb, .st-gc, .st-gd, .st-ge, .st-gf, .st-gg, .st-gh, .st-gi, .st-gj, .st-gk, .st-gl, .st-gm, .st-gn, .st-go, .st-gp, .st-gq, .st-gr, .st-gs, .st-gt, .st-gu, .st-gv, .st-gw, .st-gx, .st-gy, .st-gz, .st-h0, .st-h1, .st-h2, .st-h3, .st-h4, .st-h5, .st-h6, .st-h7, .st-h8, .st-h9, .st-ha, .st-hb, .st-hc, .st-hd, .st-he, .st-hf, .st-hg, .st-hh, .st-hi, .st-hj, .st-hk, .st-hl, .st-hm, .st-hn, .st-ho, .st-hp, .st-hq, .st-hr, .st-hs, .st-ht, .st-hu, .st-hv, .st-hw, .st-hx, .st-hy, .st-hz, .st-i0, .st-i1, .st-i2, .st-i3, .st-i4, .st-i5, .st-i6, .st-i7, .st-i8, .st-i9, .st-ia, .st-ib, .st-ic, .st-id, .st-ie, .st-if, .st-ig, .st-ih, .st-ii, .st-ij, .st-ik, .st-il, .st-im, .st-in, .st-io, .st-ip, .st-iq, .st-ir, .st-is, .st-it, .st-iu, .st-iv, .st-iw, .st-ix, .st-iy, .st-iz, .st-j0, .st-j1, .st-j2, .st-j3, .st-j4, .st-j5, .st-j6, .st-j7, .st-j8, .st-j9, .st-ja, .st-jb, .st-jc, .st-jd, .st-je, .st-jf, .st-jg, .st-jh, .st-ji, .st-jj, .st-jk, .st-jl, .st-jm, .st-jn, .st-jo, .st-jp, .st-jq, .st-jr, .st-js, .st-jt, .st-ju, .st-jv, .st-jw, .st-jx, .st-jy, .st-jz, .st-k0, .st-k1, .st-k2, .st-k3, .st-k4, .st-k5, .st-k6, .st-k7, .st-k8, .st-k9, .st-ka, .st-kb, .st-kc, .st-kd, .st-ke, .st-kf, .st-kg, .st-kh, .st-ki, .st-kj, .st-kk, .st-kl, .st-km, .st-kn, .st-ko, .st-kp, .st-kq, .st-kr, .st-ks, .st-kt, .st-ku, .st-kv, .st-kw, .st-kx, .st-ky, .st-kz, .st-l0, .st-l1, .st-l2, .st-l3, .st-l4, .st-l5, .st-l6, .st-l7, .st-l8, .st-l9, .st-la, .st-lb, .st-lc, .st-ld, .st-le, .st-lf, .st-lg, .st-lh, .st-li, .st-lj, .st-lk, .st-ll, .st-lm, .st-ln, .st-lo, .st-lp, .st-lq, .st-lr, .st-ls, .st-lt, .st-lu, .st-lv, .st-lw, .st-lx, .st-ly, .st-lz, .st-m0, .st-m1, .st-m2, .st-m3, .st-m4, .st-m5, .st-m6, .st-m7, .st-m8, .st-m9, .st-ma, .st-mb, .st-mc, .st-md, .st-me, .st-mf, .st-mg, .st-mh, .st-mi, .st-mj, .st-mk, .st-ml, .st-mm, .st-mn, .st-mo, .st-mp, .st-mq, .st-mr, .st-ms, .st-mt, .st-mu, .st-mv, .st-mw, .st-mx, .st-my, .st-mz, .st-n0, .st-n1, .st-n2, .st-n3, .st-n4, .st-n5, .st-n6, .st-n7, .st-n8, .st-n9, .st-na, .st-nb, .st-nc, .st-nd, .st-ne, .st-nf, .st-ng, .st-nh, .st-ni, .st-nj, .st-nk, .st-nl, .st-nm, .st-nn, .st-no, .st-np, .st-nq, .st-nr, .st-ns, .st-nt, .st-nu, .st-nv, .st-nw, .st-nx, .st-ny, .st-nz, .st-o0, .st-o1, .st-o2, .st-o3, .st-o4, .st-o5, .st-o6, .st-o7, .st-o8, .st-o9, .st-oa, .st-ob, .st-oc, .st-od, .st-oe, .st-of, .st-og, .st-oh, .st-oi, .st-oj, .st-ok, .st-ol, .st-om, .st-on, .st-oo, .st-op, .st-oq, .st-or, .st-os, .st-ot, .st-ou, .st-ov, .st-ow, .st-ox, .st-oy, .st-oz, .st-p0, .st-p1, .st-p2, .st-p3, .st-p4, .st-p5, .st-p6, .st-p7, .st-p8, .st-p9, .st-pa, .st-pb, .st-pc, .st-pd, .st-pe, .st-pf, .st-pg, .st-ph, .st-pi, .st-pj, .st-pk, .st-pl, .st-pm, .st-pn, .st-po, .st-pp, .st-pq, .st-pr, .st-ps, .st-pt, .st-pu, .st-pv, .st-pw, .st-px, .st-py, .st-pz, .st-q0, .st-q1, .st-q2, .st-q3, .st-q4, .st-q5, .st-q6, .st-q7, .st-q8, .st-q9, .st-qa, .st-qb, .st-qc, .st-qd, .st-qe, .st-qf, .st-qg, .st-qh, .st-qi, .st-qj, .st-qk, .st-ql, .st-qm, .st-qn, .st-qo, .st-qp, .st-qq, .st-qr, .st-qs, .st-qt, .st-qu, .st-qv, .st-qw, .st-qx, .st-qy, .st-qz, .st-r0, .st-r1, .st-r2, .st-r3, .st-r4, .st-r5, .st-r6, .st-r7, .st-r8, .st-r9, .st-ra, .st-rb, .st-rc, .st-rd, .st-re, .st-rf, .st-rg, .st-rh, .st-ri, .st-rj, .st-rk, .st-rl, .st-rm, .st-rn, .st-ro, .st-rp, .st-rq, .st-rr, .st-rs, .st-rt, .st-ru, .st-rv, .st-rw, .st-rx, .st-ry, .st-rz, .st-s0, .st-s1, .st-s2, .st-s3, .st-s4, .st-s5, .st-s6, .st-s7, .st-s8, .st-s9, .st-sa, .st-sb, .st-sc, .st-sd, .st-se, .st-sf, .st-sg, .st-sh, .st-si, .st-sj, .st-sk, .st-sl, .st-sm, .st-sn, .st-so, .st-sp, .st-sq, .st-sr, .st-ss, .st-st, .st-su, .st-sv, .st-sw, .st-sx, .st-sy, .st-sz, .st-t0, .st-t1, .st-t2, .st-t3, .st-t4, .st-t5, .st-t6, .st-t7, .st-t8, .st-t9, .st-ta, .st-tb, .st-tc, .st-td, .st-te, .st-tf, .st-tg, .st-th, .st-ti, .st-tj, .st-tk, .st-tl, .st-tm, .st-tn, .st-to, .st-tp, .st-tq, .st-tr, .st-ts, .st-tt, .st-tu, .st-tv, .st-tw, .st-tx, .st-ty, .st-tz, .st-u0, .st-u1, .st-u2, .st-u3, .st-u4, .st-u5, .st-u6, .st-u7, .st-u8, .st-u9, .st-ua, .st-ub, .st-uc, .st-ud, .st-ue, .st-uf, .st-ug, .st-uh, .st-ui, .st-uj, .st-uk, .st-ul, .st-um, .st-un, .st-u0, .st-u1, .st-u2, .st-u3, .st-u4, .st-u5, .st-u6, .st-u7, .st-u8, .st-u9, .st-ua, .st-ub, .st-uc, .st-ud, .st-ue, .st-uf, .st-ug, .st-uh, .st-ui, .st-uj, .st-uk, .st-ul, .st-um, .st-un, .st-uo, .st-up, .st-uq, .st-ur, .st-us, .st-ut, .st-uu, .st-uv, .st-uw, .st-ux, .st-uy, .st-uz, .st-v0, .st-v1, .st-v2, .st-v3, .st-v4, .st-v5, .st-v6, .st-v7, .st-v8, .st-v9, .st-va, .st-vb, .st-vc, .st-vd, .st-ve, .st-vf, .st-vg, .st-vh, .st-vi, .st-vj, .st-vk, .st-vl, .st-vm, .st-vn, .st-vo, .st-vp, .st-vq, .st-vr, .st-vs, .st-vt, .st-vu, .st-vv, .st-vw, .st-vx, .st-vy, .st-vz, .st-w0, .st-w1, .st-w2, .st-w3, .st-w4, .st-w5, .st-w6, .st-w7, .st-w8, .st-w9, .st-wa, .st-wb, .st-wc, .st-wd, .st-we, .st-wf, .st-wg, .st-wh, .st-wi, .st-wj, .st-wk, .st-wl, .st-wm, .st-wn, .st-wo, .st-wp, .st-wq, .st-wr, .st-ws, .st-wt, .st-wu, .st-wv, .st-ww, .st-wx, .st-wy, .st-wz, .st-x0, .st-x1, .st-x2, .st-x3, .st-x4, .st-x5, .st-x6, .st-x7, .st-x8, .st-x9, .st-xa, .st-xb, .st-xc, .st-xd, .st-xe, .st-xf, .st-xg, .st-xh, .st-xi, .st-xj, .st-xk, .st-xl, .st-xm, .st-xn, .st-xo, .st-xp, .st-xq, .st-xr, .st-xs, .st-xt, .st-xu, .st-xv, .st-xw, .st-xx, .st-xy, .st-xz, .st-y0, .st-y1, .st-y2, .st-y3, .st-y4, .st-y5, .st-y6, .st-y7, .st-y8, .st-y9, .st-ya, .st-yb, .st-yc, .st-yd, .st-ye, .st-yf, .st-yg, .st-yh, .st-yi, .st-yj, .st-yk, .st-yl, .st-ym, .st-yn, .st-yo, .st-yp, .st-yq, .st-yr, .st-ys, .st-yt, .st-yu, .st-yv, .st-yw, .st-yx, .st-yy, .st-yz, .st-z0, .st-z1, .st-z2, .st-z3, .st-z4, .st-z5, .st-z6, .st-z7, .st-z8, .st-z9, .st-za, .st-zb, .st-zc, .st-zd, .st-ze, .st-zf, .st-zg, .st-zh, .st-zi, .st-zj, .st-zk, .st-zl, .st-zm, .st-zn, .st-zo, .st-zp, .st-zq, .st-zr, .st-zs, .st-zt, .st-zu, .st-zv, .st-zw, .st-zx, .st-zy, .st-zz {
        color: var(--text-primary) !important;
    }

</style>
""", unsafe_allow_html=True)


# Título principal con badges
st.markdown("""
<div style="text-align: center;">
    <h1 class="main-header">PREDICTOR DE ÉXITO ACADÉMICO EN EDUCACIÓN EN LÍNEA</h1>
    <div style="display: flex; justify-content: center; gap: 1rem; margin-bottom: 1rem;">
        <span class="rf-badge">Random Forest Optimizado</span>
        <span class="cluster-badge">Análisis de Clusters</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Crear pestañas
tab1, tab2 = st.tabs(["🎯 PREDICCIÓN Y ANÁLISIS", "📊 ANÁLISIS DE CLUSTERS"])

with tab1:
    st.markdown("""
    **Sistema inteligente mejorado** que predice la probabilidad de éxito en educación en línea con **Random Forest optimizado** y **análisis de segmentación por clusters**.
    Precisión del **89.8%** | ROC-AUC de **0.898** | **4 clusters identificados**
    """)

# Mapeos para las variables
MAPEOS = {
    'si_no': {'Sí': 1, 'No': 0},
    'sexo': {'Hombre': 0, 'Mujer': 1, 'Otro': 2},
    'genero': {'Femenino': 0, 'Masculino': 1, 'Transgénero': 2, 'No binario': 3, 'Otro': 4},
    'situacion_conyugal': {'Soltero(a)': 0, 'Unión libre': 1, 'Casado(a)': 2, 'Divorciado(a)': 3, 'Separado(a)': 4, 'Viudo(a)': 5},
    'calificacion': {'Excelente': 0, 'Bueno': 1, 'Regular': 2, 'Malo': 3},
    'regimen_secundaria': {'Pública': 0, 'Privada': 1},
    'tipo_secundaria': {'General': 0, 'Técnica': 1, 'Telesecundaria': 2, 'Abierta': 3, 'Para adultos': 4},
    'edad_categoria': {'14-18': 0, '19-25': 1, '26-35': 2, '36-45': 3, '45+': 4}
}

@st.cache_resource
def cargar_modelos():
    """
    Función para cargar todos los modelos con parche de compatibilidad
    """
    try:
        # PARCHE: Crear el atributo faltante si no existe
        if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
            # Crear una clase dummy que simule el comportamiento esperado
            class _RemainderColsList(list):
                """Clase de compatibilidad para versiones antiguas de sklearn"""
                pass
            
            # Agregar el atributo al módulo
            sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
            setattr(sklearn.compose._column_transformer, '_RemainderColsList', _RemainderColsList)
        
        # Cargar modelo Random Forest
        pipeline_rf = None
        metadata = None
        modelo_clusters = None
        
        # Intentar cargar el modelo RF
        try:
            if os.path.exists('modelo_rf_streamlit_compatible.joblib'):
                pipeline_rf = joblib.load('modelo_rf_streamlit_compatible.joblib')
                metadata = joblib.load('metadatos_compatible.joblib')
                st.sidebar.success("✅ Modelo RF cargado con workaround")
            else:
                st.warning("⚠️ Modelo RF no encontrado, usando modelo alternativo")
        except Exception as e:
            st.warning(f"⚠️ Error cargando RF: {e}")
        
        # Intentar cargar el modelo de clustering
        try:
            modelo_clusters = joblib.load('modelo_clusterizacion.pkl')
            st.sidebar.success("✅ Modelo de clustering cargado")
        except Exception as e:
            st.warning(f"⚠️ Modelo de clustering no disponible: {e}")
            modelo_clusters = None
        
        return pipeline_rf, metadata, modelo_clusters
        
    except Exception as e:
        st.error(f"❌ Error al cargar modelos: {str(e)}")
        return None, None, None

def generar_calificacion_final(row):
    # Factores que influyen positivamente
    factores_positivos = 0
    
    if row.get('estudios_previos_bachillerato') == 1:  # 'Sí' mapeado a 1
        factores_positivos += 0.5
    if row.get('cursos_linea_3anos') == 1:  # 'Sí' mapeado a 1
        factores_positivos += 0.3
    if row.get('trabaja') == 0:  # 'No' mapeado a 0
        factores_positivos += 0.2
    
    # Factores que influyen negativamente
    factores_negativos = 0
    
    if row.get('horas_trabajo_numeric', 0) >= 32:
        factores_negativos += 0.3
    
    # Calificación base + factores ajustados
    calificacion_base = np.random.normal(7.5, 1.5)
    calificacion_ajustada = calificacion_base + factores_positivos - factores_negativos
    
    # Ajustar a escala 0-10
    calificacion_final = np.clip(calificacion_ajustada, 5.0, 10.0)
    
    return round(calificacion_final, 1)

def crear_formulario():
    """Crear formulario interactivo completo - VERSIÓN CORREGIDA"""
    with st.sidebar:
        st.markdown("### FORMULARIO")
        
        with st.form("formulario_estudiante"):
            # ===== SECCIÓN 1: DATOS DEMOGRÁFICOS =====
            st.markdown('<div class="section-header">👤 Datos demográficos</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                edad = st.slider("Edad", 14, 70, 25)
            with col2:
                sexo = st.selectbox("Sexo", options=list(MAPEOS['sexo'].keys()))
            
            genero = st.selectbox("Género", options=list(MAPEOS['genero'].keys()))
            situacion_conyugal = st.selectbox("Situación conyugal", options=list(MAPEOS['situacion_conyugal'].keys()))
            
            # ===== SECCIÓN 2: SALUD Y ORIGEN ===== (CORREGIDO)
            st.markdown('<div class="section-header">🏥 Salud y origen</div>', unsafe_allow_html=True)
            
            # SOLUCIÓN: Usar selectbox en lugar de radio para mejor compatibilidad
            discapacidad = st.selectbox("¿Tiene discapacidad?", options=list(MAPEOS['si_no'].keys()))
            indigena = st.selectbox("¿Se considera indígena?", options=list(MAPEOS['si_no'].keys()))
            
            # ===== SECCIÓN 3: SITUACIÓN ECONÓMICA ===== (CORREGIDO)
            st.markdown('<div class="section-header">💰 Situación económica</div>', unsafe_allow_html=True)
            
            # SOLUCIÓN: Selectbox en lugar de radio
            trabaja = st.selectbox("¿Trabaja actualmente?", options=list(MAPEOS['si_no'].keys()))
            
            if trabaja == 'Sí':
                horas_trabajo = st.slider("Horas de trabajo semanales", 0, 60, 40)
            else:
                horas_trabajo = 0
            
            ingresos_hogar = st.select_slider(
                "Ingresos mensuales del hogar (MXN)",
                options=[3000, 7500, 12500, 17500, 22500, 30000],
                value=12500,
                format_func=lambda x: f"${x:,.0f}"
            )
            
            # SOLUCIÓN: Selectbox en lugar de radio
            beca = st.selectbox("¿Recibe alguna beca?", options=list(MAPEOS['si_no'].keys()))
            
            # ===== SECCIÓN 4: TRAYECTORIA ACADÉMICA =====
            st.markdown('<div class="section-header">📚 Trayectoria académica</div>', unsafe_allow_html=True)
            
            col5, col6 = st.columns(2)
            with col5:
                regimen_secundaria = st.selectbox("Régimen de secundaria", options=list(MAPEOS['regimen_secundaria'].keys()))
            with col6:
                tipo_secundaria = st.selectbox("Tipo de secundaria", options=list(MAPEOS['tipo_secundaria'].keys()))
            
            # SOLUCIÓN: Selectbox en lugar de radio
            estudios_previos = st.selectbox("¿Tiene estudios previos de bachillerato?", options=list(MAPEOS['si_no'].keys()))
            cursos_linea = st.selectbox("¿Ha tomado cursos en línea antes?", options=list(MAPEOS['si_no'].keys()))
            
            # ===== SECCIÓN 5: HABILIDADES Y RECURSOS =====
            st.markdown('<div class="section-header">💻 Habilidades y recursos</div>', unsafe_allow_html=True)
            
            col7, col8 = st.columns(2)
            with col7:
                recursos_tec = st.slider("Recursos tecnológicos", 1, 5, 3)
            with col8:
                responsabilidades = st.slider("Responsabilidades", 1, 7, 3)
            
            comunicacion = st.select_slider("Habilidad de comunicación", 
                                          options=list(MAPEOS['calificacion'].keys()), value="Bueno")
            evaluacion = st.select_slider("Habilidad evaluación información", 
                                        options=list(MAPEOS['calificacion'].keys()), value="Bueno")
            organizacion = st.select_slider("Habilidad de organización", 
                                          options=list(MAPEOS['calificacion'].keys()), value="Bueno")
            
            # Calcular categoría de edad automáticamente
            if edad <= 18:
                edad_categoria = '14-18'
            elif edad <= 25:
                edad_categoria = '19-25'
            elif edad <= 35:
                edad_categoria = '26-35'
            elif edad <= 45:
                edad_categoria = '36-45'
            else:
                edad_categoria = '45+'
            
            # Botón de enviar
            submitted = st.form_submit_button("🎓 Predecir con Random Forest + Clusters", use_container_width=True)
            
            datos = {
                'edad': edad, 'sexo': sexo, 'genero': genero, 'situacion_conyugal': situacion_conyugal,
                'discapacidad': discapacidad, 'indigena': indigena, 'trabaja': trabaja,
                'horas_trabajo_numeric': horas_trabajo, 'ingresos_hogar_numeric': ingresos_hogar,
                'beca': beca, 'regimen_secundaria': regimen_secundaria, 'tipo_secundaria': tipo_secundaria,
                'estudios_previos_bachillerato': estudios_previos, 'cursos_linea_3anos': cursos_linea,
                'score_recursos_tecnologicos': recursos_tec, 'score_responsabilidades': responsabilidades,
                'comunicacion_escrita': comunicacion, 'evaluacion_informacion': evaluacion,
                'organizacion_plataforma': organizacion, 'edad_categoria': edad_categoria
            }
            
            return submitted, datos

def preprocesar_datos(datos):
    """Preprocesar datos para el modelo"""
    datos_procesados = {}
    
    for key, value in datos.items():
        if key in ['edad', 'horas_trabajo_numeric', 'ingresos_hogar_numeric', 
                  'score_recursos_tecnologicos', 'score_responsabilidades']:
            datos_procesados[key] = float(value)
        else:
            # Buscar en los mapeos correspondientes
            for mapeo_key, mapeo in MAPEOS.items():
                if value in mapeo:
                    datos_procesados[key] = mapeo[value]
                    break
            else:
                datos_procesados[key] = value
    
    return datos_procesados

def crear_dataframe_modelo(datos_procesados):
    """Crear DataFrame con la estructura que el modelo espera"""
    columnas_esperadas = [
        'edad', 'sexo', 'genero', 'situacion_conyugal', 'discapacidad', 'indigena',
        'trabaja', 'horas_trabajo_numeric', 'ingresos_hogar_numeric', 'beca',
        'regimen_secundaria', 'tipo_secundaria', 'estudios_previos_bachillerato',
        'cursos_linea_3anos', 'score_recursos_tecnologicos', 'score_responsabilidades',
        'comunicacion_escrita', 'evaluacion_informacion', 'organizacion_plataforma',
        'edad_categoria'
    ]
    
    df = pd.DataFrame(columns=columnas_esperadas)
    
    for columna in columnas_esperadas:
        if columna in datos_procesados:
            df[columna] = [datos_procesados[columna]]
        else:
            df[columna] = [0]  # Valor por defecto
    
    return df

def predecir_cluster(datos_procesados, modelo_clusters):
    """Predecir el cluster del estudiante"""
    try:
        # Calcular calificación final EXACTA como en el código original
        calificacion_final = generar_calificacion_final(datos_procesados)
        datos_procesados['calificacion_final'] = calificacion_final * 10  # Convertir a 0-100
        
        features_clustering = modelo_clusters['features_clustering']
        df_input = pd.DataFrame([datos_procesados])
        
        features_faltantes = [f for f in features_clustering if f not in df_input.columns]
        if features_faltantes:
            st.warning(f"⚠️ Algunas features faltan para clustering: {features_faltantes}")
            return None
        
        X_scaled = modelo_clusters['scaler'].transform(df_input[features_clustering])
        cluster = modelo_clusters['kmeans_model'].predict(X_scaled)[0]
        distancias = modelo_clusters['kmeans_model'].transform(X_scaled)[0]
        
        confianza = 1 / (1 + distancias.min())
        
        return {
            'cluster': cluster,
            'distancias': distancias,
            'confianza': confianza,
            'features_utilizadas': features_clustering,
            'calificacion_final_calculada': calificacion_final
        }
    except Exception as e:
        st.error(f"❌ Error en predicción de cluster: {str(e)}")
        return None

def crear_gauge_chart(probabilidad):
    """Crear gráfico tipo gauge para la probabilidad"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probabilidad * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilidad de Éxito (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [0, 40], 'color': "#FFCDD2"},
                {'range': [40, 60], 'color': "#FFECB3"},
                {'range': [60, 80], 'color': "#C8E6C9"},
                {'range': [80, 100], 'color': "#A5D6A7"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50}}))
    
    fig.update_layout(height=300)
    return fig

def mostrar_feature_importance_personalizada(datos_usuario):
    """Mostrar feature importance personalizada"""
    # Feature importance del Random Forest (basada en tus resultados)
    features_rf = {
        'edad': 0.548,
        'edad_categoria': 0.102,
        'estudios_previos_bachillerato': 0.051,
        'horas_trabajo_numeric': 0.050,
        'ingresos_hogar_numeric': 0.034,
        'score_recursos_tecnologicos': 0.033,
        'cursos_linea_3anos': 0.032,
        'tipo_secundaria': 0.027,
        'score_responsabilidades': 0.023
    }
    
    st.markdown('<div class="section-header">📊 Importancia de factores para el caso particular</div>', unsafe_allow_html=True)
    
    # Crear gráfico de barras
    fig_importance = px.bar(
        x=list(features_rf.values())[:6],
        y=list(features_rf.keys())[:6],
        orientation='h',
        title="Top 6 factores más importantes (Random Forest)",
        color=list(features_rf.values())[:6],
        color_continuous_scale='Greens'
    )
    fig_importance.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Análisis personalizado
    st.markdown("**🔍 Análisis de tu perfil:**")
    
    if datos_usuario['edad'] > 30:
        st.markdown(f'<div class="feature-importance-box">📊 **Edad ({datos_usuario["edad"]} años)**: Factor dominante pero menos que en otros modelos (55% vs 80% en GB)</div>', unsafe_allow_html=True)
    
    if datos_usuario['ingresos_hogar_numeric'] < 15000:
        st.markdown('<div class="feature-importance-box">💰 **Ingresos**: Nivel económico bajo puede ser un factor de riesgo importante</div>', unsafe_allow_html=True)
    
    if datos_usuario['horas_trabajo_numeric'] > 30:
        st.markdown('<div class="feature-importance-box">⏰ **Carga laboral**: Muchas horas de trabajo pueden afectar el rendimiento académico</div>', unsafe_allow_html=True)

def mostrar_resultados_cluster(resultado_cluster):
    """Mostrar resultados del análisis de cluster"""
    if not resultado_cluster:
        return
    
    cluster = resultado_cluster['cluster']
    confianza = resultado_cluster['confianza']
    calificacion = resultado_cluster.get('calificacion_final_calculada', 'N/A')
    
    st.markdown('<div class="section-header">🎯 ANÁLISIS DE SEGMENTACIÓN (CLUSTERS)</div>', unsafe_allow_html=True)
    
    # Descripciones de clusters
    descripciones_clusters = {
        0: {
            "nombre": "🎓 Estudiantes con Ventaja Socioeconómica",
            "descripcion": "Jóvenes con buenos recursos tecnológicos y económicos, menor carga de responsabilidades.",
            "tamaño": "21.5% de estudiantes",
            "exito_promedio": "59.1%"
        },
        1: {
            "nombre": "💼 Estudiantes Trabajadores",
            "descripcion": "Adultos jóvenes que combinan estudio con trabajo extensivo, alta carga de responsabilidades.",
            "tamaño": "27.6% de estudiantes",
            "exito_promedio": "45.1%"
        },
        2: {
            "nombre": "🌟 Estudiantes Maduros Resilientes", 
            "descripcion": "Adultos con mayor edad y responsabilidades, pero alto rendimiento académico.",
            "tamaño": "21.7% de estudiantes",
            "exito_promedio": "65.8%"
        },
        3: {
            "nombre": "📚 Estudiantes Jóvenes en Desventaja",
            "descripcion": "Adolescentes con recursos limitados, requieren apoyo prioritario.",
            "tamaño": "29.2% de estudiantes",
            "exito_promedio": "35.2%"
        }
    }
    
    info_cluster = descripciones_clusters.get(cluster, {})
    
    # Métricas del cluster
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Cluster Asignado", f"Cluster {cluster}")
    
    with col2:
        st.metric("Confianza Cluster", f"{confianza:.1%}")
    
    with col3:
        st.metric("Calificación Calculada", f"{calificacion}/10")
    
    with col4:
        st.metric("Tamaño del Grupo", info_cluster.get("tamaño", "N/A"))
    
    with col5:
        st.metric("Éxito Promedio", info_cluster.get("exito_promedio", "N/A"))
    
    # Información detallada del cluster
    st.markdown(f'<div class="cluster-info-box">'
                f'<h3>📋 {info_cluster.get("nombre", "Cluster Desconocido")}</h3>'
                f'<p><strong>Descripción:</strong> {info_cluster.get("descripcion", "Información no disponible")}</p>'
                f'<p><strong>Características principales:</strong></p>'
                f'<ul>'
                f'<li><strong>Tamaño:</strong> {info_cluster.get("tamaño", "N/A")}</li>'
                f'<li><strong>Éxito académico promedio:</strong> {info_cluster.get("exito_promedio", "N/A")}</li>'
                f'<li><strong>Calificación estimada:</strong> {calificacion}/10</li>'
                f'<li><strong>Confianza de asignación:</strong> {confianza:.1%}</li>'
                f'</ul>'
                f'</div>', unsafe_allow_html=True)
    
    # Gráfico de distancias a clusters
    fig_distancias = go.Figure(data=[
        go.Bar(
            x=[f'Cluster {i}' for i in range(4)],
            y=resultado_cluster['distancias'],
            marker_color=['#2196F3' if i == cluster else '#64B5F6' for i in range(4)],
            text=[f'{d:.2f}' for d in resultado_cluster['distancias']],
            textposition='auto',
        )
    ])
    
    fig_distancias.update_layout(
        title="Distancias a los Centroides de Cada Cluster",
        xaxis_title="Cluster",
        yaxis_title="Distancia",
        height=400
    )
    
    st.plotly_chart(fig_distancias, use_container_width=True)
    
    # Recomendaciones específicas por cluster
    recomendaciones_clusters = {
        0: [
            "**Potencial de liderazgo**: Participar como mentor de otros estudiantes",
            "**Programas de excelencia**: Explorar oportunidades académicas avanzadas",
            "**Proyectos especiales**: Involucrarse en iniciativas institucionales"
        ],
        1: [
            "**Gestión del tiempo**: Solicitar flexibilidad en horarios de entrega",
            "**Comunidad de apoyo**: Unirse a grupos de estudiantes trabajadores",
            "**Asesoría académica**: Recibir orientación específica para balance trabajo-estudio"
        ],
        2: [
            "**Redes de apoyo**: Participar en comunidades de estudiantes maduros", 
            "**Recursos tecnológicos**: Acceder a programas de actualización digital",
            "**Experiencia compartida**: Contribuir como referente para otros estudiantes"
        ],
        3: [
            "**Apoyo prioritario**: Solicitar asistencia económica y tecnológica urgente",
            "**Mentorías personalizadas**: Participar en programas de acompañamiento",
            "**Seguimiento estrecho**: Mantener contacto frecuente con asesores académicos"
        ]
    }
    
    st.markdown(f'<div class="section-header">💡 RECOMENDACIONES ESPECÍFICAS PARA CLUSTER {cluster}</div>', unsafe_allow_html=True)
    
    for i, rec in enumerate(recomendaciones_clusters.get(cluster, []), 1):
        st.markdown(f'<div class="recommendation-box">**{i}.** {rec}</div>', unsafe_allow_html=True)

def mostrar_resultados_prediccion(probabilidad, prediccion, datos_originales, metadata):
    """Mostrar resultados de la predicción de éxito académico"""
    # Determinar nivel de riesgo con RF
    if probabilidad < 0.35:
        nivel_riesgo = "MUY ALTO"
        color_clase = "risk-high"
        emoji = "🔴"
        confianza = "Baja"
    elif probabilidad < 0.5:
        nivel_riesgo = "ALTO"  
        color_clase = "risk-high"
        emoji = "🟠"
        confianza = "Media"
    elif probabilidad < 0.7:
        nivel_riesgo = "MEDIO"
        color_clase = "risk-medium"
        emoji = "🟡"
        confianza = "Buena"
    elif probabilidad < 0.85:
        nivel_riesgo = "BAJO"
        color_clase = "risk-low"
        emoji = "🟢"
        confianza = "Alta"
    else:
        nivel_riesgo = "MUY BAJO"
        color_clase = "risk-low"
        emoji = "✅"
        confianza = "Muy alta"
    
    # Mostrar métricas principales
    st.markdown('<div class="section-header">🌲 PREDICCIÓN DE ÉXITO ACADÉMICO</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Probabilidad de éxito", f"{probabilidad:.1%}")
    
    with col2:
        resultado = "✅ ÉXITO PROBABLE" if prediccion == 1 else "⚠️ RIESGO ALTO"
        st.metric("Predicción", resultado)
    
    with col3:
        st.metric("Nivel de riesgo", f"{emoji} {nivel_riesgo}")
    
    with col4:
        st.metric("Confianza", confianza)
    
    # Gráfico gauge interactivo
    st.plotly_chart(crear_gauge_chart(probabilidad), use_container_width=True)
    
    # Mostrar feature importance personalizada
    mostrar_feature_importance_personalizada(datos_originales)

def generar_recomendaciones_combinadas(probabilidad, datos, cluster_resultado):
    """Generar recomendaciones combinando predicción y clusterización"""
    st.markdown('<div class="section-header">🎯 RECOMENDACIONES INTEGRALES</div>', unsafe_allow_html=True)
    
    recomendaciones = []
    
    # Recomendaciones basadas en la predicción
    if probabilidad < 0.4:
        recomendaciones.append("**Intervención integral inmediata**: El modelo RF indica múltiples factores de riesgo")
        recomendaciones.append("**Contacto urgente con asesor académico** para plan personalizado")
    elif probabilidad < 0.6:
        recomendaciones.append("**Plan de mejora multifactor**: RF identifica áreas específicas de mejora")
        recomendaciones.append("**Programa de acompañamiento** personalizado recomendado")
    else:
        recomendaciones.append("**Perfil favorable**: RF predice alta probabilidad de éxito")
        recomendaciones.append("**Mantener estrategia actual** con seguimiento regular")
    
    # Recomendaciones específicas basadas en features importantes en RF
    if datos['edad'] > 35:
        recomendaciones.append("**Programa para adultos mayores**: Estrategias específicas para estudiantes maduros")
    
    if datos['ingresos_hogar_numeric'] < 12000:
        recomendaciones.append("**Apoyo económico crítico**: RF identifica ingresos como factor clave")
        recomendaciones.append("**Solicitar beca o apoyo financiero** urgentemente")
    
    if datos['horas_trabajo_numeric'] > 35:
        recomendaciones.append("**Gestión tiempo-trabajo crítica**: RF muestra que es factor importante")
        recomendaciones.append("**Negociar flexibilidad laboral** si es posible")
    
    # Recomendaciones basadas en el cluster
    if cluster_resultado:
        cluster = cluster_resultado['cluster']
        if cluster == 0:
            recomendaciones.append("**Aprovechar ventajas socioeconómicas**: Potencial de liderazgo académico")
        elif cluster == 1:
            recomendaciones.append("**Balance trabajo-estudio**: Estrategias específicas para estudiantes trabajadores")
        elif cluster == 2:
            recomendaciones.append("**Capitalizar resiliencia**: Compartir experiencias como estudiante maduro")
        elif cluster == 3:
            recomendaciones.append("**Apoyo prioritario integral**: Enfoque multidimensional para estudiantes jóvenes")
    
    # Mostrar recomendaciones
    for i, rec in enumerate(recomendaciones, 1):
        st.markdown(f'<div class="recommendation-box">**{i}.** {rec}</div>', unsafe_allow_html=True)

def main():
    """Función principal"""
    pipeline, metadata, modelo_clusters = cargar_modelos()
    
    # Crear formulario en sidebar
    submitted, datos_usuario = crear_formulario()
    
    # Área principal para resultados (solo en la pestaña 1)
    with tab1:
        if submitted:
            try:
                with st.spinner('🌲 Random Forest analizando... Modelo más balanceado procesando datos...'):
                    # Preprocesar datos
                    datos_procesados = preprocesar_datos(datos_usuario)
                    
                    # Crear DataFrame
                    X_nuevo = crear_dataframe_modelo(datos_procesados)
                    
                    # Hacer predicción de éxito académico
                    probabilidad = None
                    prediccion = None
                    
                    if pipeline is not None:
                        probabilidad = pipeline.predict_proba(X_nuevo)[0, 1]
                        prediccion = pipeline.predict(X_nuevo)[0]
                    else:
                        # Simular predicción si el modelo no está disponible
                        st.warning("⚠️ Usando predicción simulada - modelo RF no disponible")
                        probabilidad = 0.65
                        prediccion = 1
                    
                    # Hacer predicción de cluster
                    resultado_cluster = None
                    if modelo_clusters is not None:
                        with st.spinner('🎯 Analizando segmentación por clusters...'):
                            resultado_cluster = predecir_cluster(datos_procesados, modelo_clusters)
                
                # Mostrar resultados
                st.success("🌲 ¡Análisis completado exitosamente! (Predicción + Clusterización)")
                
                # Mostrar resultados de predicción
                if probabilidad is not None:
                    mostrar_resultados_prediccion(probabilidad, prediccion, datos_usuario, metadata)
                
                # Mostrar resultados de clusterización
                if resultado_cluster:
                    mostrar_resultados_cluster(resultado_cluster)
                else:
                    st.warning("⚠️ No se pudo realizar el análisis de clusters")
                
                # Mostrar recomendaciones combinadas
                if probabilidad is not None:
                    generar_recomendaciones_combinadas(probabilidad, datos_usuario, resultado_cluster)
                
                # Información técnica
                with st.expander("🔧 DETALLES TÉCNICOS COMPLETOS"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🌲 Random Forest:**")
                        if probabilidad is not None:
                            st.write(f"🎯 Probabilidad exacta: {probabilidad:.6f}")
                        st.write(f"⚖️ Umbral de clasificación: 0.5")
                        if metadata:
                            st.write(f"📈 ROC-AUC: {metadata.get('roc_auc', 'N/A')}")
                            st.write(f"🎯 Accuracy: {metadata.get('accuracy', 'N/A')}")
                    
                    with col2:
                        st.write("**🎯 Clusterización:**")
                        if resultado_cluster:
                            st.write(f"🔢 Cluster asignado: {resultado_cluster['cluster']}")
                            st.write(f"🎯 Confianza cluster: {resultado_cluster['confianza']:.3f}")
                            if modelo_clusters:
                                st.write(f"📊 Número de clusters: {modelo_clusters['kmeans_model'].n_clusters}")
                                st.write(f"🔍 Features utilizadas: {len(resultado_cluster['features_utilizadas'])}")
                    
            except Exception as e:
                st.error(f"❌ Error en el análisis: {str(e)}")
                st.info("ℹ️ Verifica que todos los campos estén completos correctamente.")
        
        # Información cuando no hay predicción
        else:
            st.info("""
            👈 **Complete el formulario** para obtener un análisis completo con Random Forest optimizado y segmentación por clusters.
            
            **🔍 Análisis que recibirás:**
            -  **🌲 Predicción de éxito académico** con Random Forest optimizado
            -  **🎯 Segmentación por cluster** para identificar tu perfil estudiantil
            -  **💡 Recomendaciones personalizadas** basadas en ambos análisis
            -  **📊 Visualizaciones interactivas** de resultados
            
            ** Ventajas del sistema integrado:**
            -  **Predicción precisa** (89.8% ROC-AUC)
            -  **Segmentación inteligente** (4 clusters identificados)
            -  **Recomendaciones contextualizadas** por perfil
            -  **Enfoque integral** para el éxito académico
            """)
            
            # Información sobre clusters
            st.markdown('<div class="section-header">🎯 CLUSTERS IDENTIFICADOS</div>', unsafe_allow_html=True)
            
            clusters_info = {
                "Cluster 0": "🎓 Estudiantes con Ventaja Socioeconómica (21.5%) - Éxito: 59.1%",
                "Cluster 1": "💼 Estudiantes Trabajadores (27.6%) - Éxito: 45.1%", 
                "Cluster 2": "🌟 Estudiantes Maduros Resilientes (21.7%) - Éxito: 65.8%",
                "Cluster 3": "📚 Estudiantes Jóvenes en Desventaja (29.2%) - Éxito: 35.2%"
            }
            
            for cluster, desc in clusters_info.items():
                st.markdown(f"• **{cluster}**: {desc}")

    # Pestaña 2: Análisis de Clusters
    with tab2:
        st.markdown('<div class="section-header">📊 ANÁLISIS DETALLADO DE CLUSTERS</div>', unsafe_allow_html=True)
        
        # Mostrar las imágenes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Visualización 2D de Clusters (PCA)")
            try:
                st.image("images/clusters_PCA_2D.png", use_column_width=True, 
                        caption="Distribución de estudiantes en los 4 clusters identificados")
            except FileNotFoundError:
                st.warning("⚠️ No se encontró la imagen 'clusters_PCA_2D.png'. Asegúrate de que esté en el mismo directorio.")
        
        with col2:
            st.markdown("### Análisis de Componentes Principales")
            try:
                st.image("images/radar_cluster_estandarizado.png", use_column_width=True, 
                        caption="Radar con las características de los estudiantes")
            except FileNotFoundError:
                st.warning("⚠️ No se encontró la imagen 'radar_cluster_estandarizado.png'. Asegúrate de que esté en el mismo directorio.")
        
        # Texto descriptivo de los clusters
        st.markdown('<div class="section-header">🎯 CARACTERIZACIÓN DETALLADA DE PERFILES</div>', unsafe_allow_html=True)
        
        # Cluster 2
        st.markdown("""
        <div class="cluster-detail-box">
            <h3>🌟 Cluster 2: "Estudiantes maduros con alta responsabilidad"</h3>
            <p><strong>Perfil:</strong> Adultos mayores (45.6 años promedio) con alta carga de responsabilidades</p>
            <p><strong>Fortalezas:</strong> Mayor tasa de éxito académico (65.8%) a pesar de responsabilidades</p>
            <p><strong>Desafíos:</strong> Recursos tecnológicos limitados (2.98/4.0)</p>
            <p><strong>Oportunidad:</strong> Estudiantes resilientes que podrían beneficiarse de mejor acceso tecnológico</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cluster 0
        st.markdown("""
        <div class="cluster-detail-box">
            <h3>🎓 Cluster 0: "Estudiantes privilegiados jóvenes"</h3>
            <p><strong>Perfil:</strong> Jóvenes (20 años) con alto nivel socioeconómico y recursos tecnológicos</p>
            <p><strong>Fortalezas:</strong> Alto éxito académico (59.1%) apoyado por buenos recursos</p>
            <p><strong>Ventaja:</strong> Menos responsabilidades y mayor acceso a recursos educativos</p>
            <p><strong>Potencial:</strong> Podrían servir como mentores o grupos de referencia</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cluster 1
        st.markdown("""
        <div class="cluster-detail-box">
            <h3>💼 Cluster 1: "Estudiantes trabajadores"</h3>
            <p><strong>Perfil:</strong> Adultos jóvenes (21.9 años) que combinan estudio con trabajo extensivo (27h/semana)</p>
            <p><strong>Desafío:</strong> Bajo éxito académico (45.1%) posiblemente por carga laboral</p>
            <p><strong>Necesidad:</strong> Flexibilidad horaria y apoyo específico para estudiantes trabajadores</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cluster 3
        st.markdown("""
        <div class="cluster-detail-box">
            <h3>📚 Cluster 3: "Estudiantes jóvenes con desventaja socioeconómica"</h3>
            <p><strong>Perfil:</strong> Adolescentes (17.6 años) con bajos ingresos familiares y recursos limitados</p>
            <p><strong>Alerta crítica:</strong> Menor tasa de éxito (35.2%) requiere intervención prioritaria</p>
            <p><strong>Factores de riesgo:</strong> Bajos ingresos, recursos tecnológicos insuficientes</p>
            <p><strong>Estrategia:</strong> Apoyo económico y tecnológico urgente</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Resumen estadístico
        st.markdown('<div class="section-header">📈 RESUMEN ESTADÍSTICO DE CLUSTERS</div>', unsafe_allow_html=True)
        
        # Crear una tabla resumen
        resumen_data = {
            'Cluster': ['2 - Maduros', '0 - Privilegiados', '1 - Trabajadores', '3 - En desventaja'],
            'Edad Promedio': ['45.6 años', '20 años', '21.9 años', '17.6 años'],
            'Éxito Académico': ['65.8%', '59.1%', '45.1%', '35.2%'],
            'Tamaño': ['21.7%', '21.5%', '27.6%', '29.2%'],
            'Prioridad': ['Media', 'Baja', 'Media', 'Alta']
        }
        
        df_resumen = pd.DataFrame(resumen_data)
        st.dataframe(df_resumen, use_container_width=True, hide_index=True)
        
        # Recomendaciones generales
        st.markdown('<div class="section-header">💡 RECOMENDACIONES ESTRATÉGICAS</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="recommendation-box">
        <strong>Estrategias diferenciadas por cluster:</strong>
        <ul>
        <li><strong>Cluster 2:</strong> Programas de actualización tecnológica y horarios flexibles</li>
        <li><strong>Cluster 0:</strong> Programas de liderazgo y mentoría estudiantil</li>
        <li><strong>Cluster 1:</strong> Flexibilidad en entregas y asesoría para balance trabajo-estudio</li>
        <li><strong>Cluster 3:</strong> Apoyo económico urgente, becas tecnológicas y acompañamiento intensivo</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()


