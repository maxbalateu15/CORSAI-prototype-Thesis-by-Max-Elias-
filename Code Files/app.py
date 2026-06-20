"""
CORSAI — Curatorial Intelligence
Run with:  streamlit run app.py
Expects ./output/ to contain: features_final.csv, distance_matrix.npy
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="CORSAI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300&family=Inconsolata:wght@300;400&display=swap');

html, body, [class*="css"] {
    background-color: #080808 !important;
    color: #c8c4bc !important;
    font-family: 'Inconsolata', monospace !important;
}
section[data-testid="stSidebar"] {
    background-color: #0b0b0b !important;
    border-right: 1px solid #181818 !important;
}
h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    letter-spacing: 0.08em !important;
    color: #f0ede8 !important;
}
h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.5rem !important; }
p, li, span, div, label { color: #c8c4bc !important; }
.stSelectbox label, .stSlider label, .stNumberInput label,
.stTextInput label, .stRadio label { color: #888 !important; }
div[data-baseweb="select"] div,
div[data-baseweb="input"] input {
    color: #c8c4bc !important;
    background-color: #0d0d0d !important;
}
input::placeholder { color: #333 !important; }
.corsai-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.9rem;
    font-weight: 300;
    letter-spacing: 0.28em;
    color: #f0ede8;
    text-transform: uppercase;
    line-height: 1;
}
.corsai-sub {
    font-family: 'Inconsolata', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.22em;
    color: #aaa;
    text-transform: uppercase;
    margin-top: 3px;
}
.intro-block {
    border-left: 1px solid #c9a84c;
    padding: 16px 22px;
    margin: 20px 0 28px 0;
    background: #0a0a0a;
}
.intro-block p {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    font-weight: 300;
    font-style: italic;
    line-height: 1.8;
    color: #a8a49c;
    margin: 0;
}
div[role="radiogroup"] label {
    font-family: 'Inconsolata', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #555 !important;
    padding: 5px 0 !important;
}
div[role="radiogroup"] label:hover { color: #c9a84c !important; }
.divider {
    border: none;
    border-top: 1px solid #141414;
    margin: 16px 0;
}
.tag-chip {
    display: inline-block;
    font-family: 'Inconsolata', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #666;
    border: 1px solid #1e1e1e;
    padding: 1px 7px;
    margin: 2px 1px;
}
.desc-bar-wrap {
    font-family: 'Inconsolata', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.07em;
    color: #555;
    text-transform: uppercase;
    text-align: center;
}
.track-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.0rem;
    font-weight: 300;
    color: #e8e4de;
    letter-spacing: 0.03em;
}
.track-meta {
    font-family: 'Inconsolata', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    color: #555;
    text-transform: uppercase;
    margin-top: 1px;
}
.map-frame {
    border: 1px solid #c9a84c;
    padding: 3px;
    margin: 12px 0;
}
.stButton > button {
    font-family: 'Inconsolata', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: 1px solid #1e1e1e !important;
    color: #666 !important;
    border-radius: 0 !important;
}
.stButton > button:hover {
    border-color: #c9a84c !important;
    color: #c9a84c !important;
}
div[data-testid="stSlider"] label {
    font-family: 'Inconsolata', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #555 !important;
}
details summary {
    font-family: 'Inconsolata', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: #555 !important;
}
div[data-testid="stCaptionContainer"] p {
    font-family: 'Inconsolata', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.12em !important;
    color: #aaa !important;
    text-transform: uppercase !important;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./output"

DESCRIPTOR_COLS = [
    "brightness", "compactness", "density", "rhythm_strength",
    "texture", "warmth", "tonality", "punch", "bass_weight", "groove_regularity",
]

CLUSTER_COLORS = [
    "#c9a84c", "#8b6f47", "#4a6741", "#5c7a8a", "#8a5c6b",
    "#6b5c8a", "#8a7a5c", "#4a6b6b", "#7a5c4a", "#5c6b4a",
]

HOVER_STYLE = dict(
    bgcolor="#0d0d0d",
    font=dict(family="Inconsolata", size=11, color="#c8c4bc"),
    bordercolor="#2a2a2a",
)

PLOT_BASE = dict(
    plot_bgcolor="#040404",
    paper_bgcolor="#080808",
    font=dict(family="Inconsolata", color="#555"),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    legend=dict(font=dict(family="Inconsolata", size=9, color="#bbb"),
                bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    hoverlabel=HOVER_STYLE,
    margin=dict(l=0, r=0, t=8, b=0),
)

# ── loaders ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading corpus …")
def load_data():
    for csv_name in ["features_final.csv", "features_v2_final.csv"]:
        csv_path = os.path.join(OUTPUT_DIR, csv_name)
        if os.path.exists(csv_path): break
    else:
        st.error(f"features_final.csv not found in {OUTPUT_DIR}. Run the pipeline first.")
        st.stop()

    for dist_name in ["distance_matrix.npy", "distance_matrix_v2.npy"]:
        dist_path = os.path.join(OUTPUT_DIR, dist_name)
        if os.path.exists(dist_path): break
    else:
        st.error(f"distance_matrix.npy not found in {OUTPUT_DIR}. Run the pipeline first.")
        st.stop()

    df   = pd.read_csv(csv_path)
    dist = np.load(dist_path)

    annot_path = os.path.join(OUTPUT_DIR, "annotations.csv")
    if os.path.exists(annot_path):
        annot = pd.read_csv(annot_path)
        df    = df.merge(annot, on="track_id", how="left", suffixes=("", "_annot"))

    return df, dist
MANUAL_TAGS_PATH = os.path.join(OUTPUT_DIR, "manual_tags.csv")

def load_manual_tags():
    if os.path.exists(MANUAL_TAGS_PATH):
        return pd.read_csv(MANUAL_TAGS_PATH)
    return pd.DataFrame(columns=["track_id", "manual_tags", "notes"])

def save_manual_tags(df_manual):
    df_manual.to_csv(MANUAL_TAGS_PATH, index=False)

df, dist_matrix = load_data()


# apply manual tag overrides
_manual = load_manual_tags()
if not _manual.empty:
    df = df.merge(_manual[["track_id", "manual_tags"]], on="track_id", how="left")
    mask = df["manual_tags"].notna()
    df.loc[mask, "tags"] = df.loc[mask, "manual_tags"]
    df.drop(columns=["manual_tags"], inplace=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def get_similar(track_id, k=10):
    matches = df.index[df["track_id"] == track_id].tolist()
    if not matches: return pd.DataFrame()
    i        = matches[0]
    dists    = dist_matrix[i].copy()
    dists[i] = np.inf
    idx      = np.argsort(dists)[:k]
    result   = df.iloc[idx].copy()
    result["distance"] = dists[idx]
    return result.reset_index(drop=True)

def descriptor_bar(value, width=68):
    filled = int(value * width)
    return (
        f'<div style="display:inline-block;width:{width}px;height:2px;'
        f'background:#181818;vertical-align:middle">'
        f'<div style="width:{filled}px;height:2px;background:#c9a84c"></div>'
        f'</div>'
    )

def track_card(row, show_distance=False):
    key     = row.get("key", "")
    camelot = row.get("camelot", "")
    tempo   = row.get("tempo", "")
    tags    = row.get("tags", "")
    meta    = []
    if key:     meta.append(str(key))
    if camelot: meta.append(str(camelot))
    if tempo:   meta.append(f"{float(tempo):.0f} bpm")
    if show_distance and "distance" in row.index:
        meta.append(f"dist {row['distance']:.3f}")
    st.markdown(f'<div class="track-name">{row["track_id"]}</div>', unsafe_allow_html=True)
    if meta:
        st.markdown(f'<div class="track-meta">{" · ".join(meta)}</div>', unsafe_allow_html=True)
    if tags and str(tags) != "nan":
        chips = "".join(f'<span class="tag-chip">{t.strip()}</span>' for t in str(tags).split(","))
        st.markdown(chips, unsafe_allow_html=True)
    desc_present = [c for c in DESCRIPTOR_COLS if c in row.index]
    if desc_present:
        cols = st.columns(len(desc_present))
        for col, c in zip(cols, desc_present):
            val = float(row[c]) if pd.notna(row.get(c)) else 0.0
            col.markdown(
                f'<div class="desc-bar-wrap">{c}<br>'
                f'{descriptor_bar(val)}<br>{val:.2f}</div>',
                unsafe_allow_html=True,
            )
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

def camelot_compat(c1, c2):
    if not c1 or not c2 or c1 == "?" or c2 == "?": return 0.5
    if c1 == c2: return 1.0
    try:
        n1, l1 = int(c1[:-1]), c1[-1]
        n2, l2 = int(c2[:-1]), c2[-1]
        if (l1 == l2 and abs(n1-n2) <= 1) or (l1 != l2 and n1 == n2): return 0.7
    except: pass
    return 0.0
MANUAL_TAGS_PATH = os.path.join(OUTPUT_DIR, "manual_tags.csv")

def load_manual_tags():
    if os.path.exists(MANUAL_TAGS_PATH):
        return pd.read_csv(MANUAL_TAGS_PATH)
    return pd.DataFrame(columns=["track_id", "manual_tags", "notes"])

def save_manual_tags(df_manual):
    df_manual.to_csv(MANUAL_TAGS_PATH, index=False)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="corsai-title">CORSAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="corsai-sub">Curatorial Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    tab = st.radio(
    "nav",
    ["Nebula Map", "Similarity", "Feature Profile", "Search & Filter", "Manual Tags"],
    label_visibility="collapsed",)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    n_cl  = int(df["cluster"].nunique()) if "cluster" in df.columns else "—"
    n_ann = df["groove"].notna().sum() if "groove" in df.columns else 0
    st.markdown(
        f'<div class="corsai-sub">{len(df)} tracks</div>'
        f'<div class="corsai-sub">{n_cl} clusters</div>'
        f'<div class="corsai-sub">{n_ann} annotated</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="corsai-sub" style="margin-bottom:8px">Corpus folder</div>', unsafe_allow_html=True)
    tracks_dir_input = st.text_input(
        "tracks_dir", value="", placeholder="/Users/you/Music/CORS",
        label_visibility="collapsed",
    )
    if tracks_dir_input:
        if os.path.isdir(tracks_dir_input):
            audio_files = (
                list(Path(tracks_dir_input).rglob("*.mp3")) +
                list(Path(tracks_dir_input).rglob("*.MP3")) +
                list(Path(tracks_dir_input).rglob("*.wav")) +
                list(Path(tracks_dir_input).rglob("*.WAV"))
            )
            st.markdown(
                f'<div class="corsai-sub">{len(audio_files)} files found</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="corsai-sub" style="color:#8a3a3a">path not found</div>',
                unsafe_allow_html=True,
            )

# ── intro ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="intro-block"><p>
CORSAI maps the sonic territory of CORS Records — not by genre, but by feel.<br>
Each point is a track. Each cluster, a gravitational field of shared texture, rhythm and depth.<br>
Navigate by proximity, uncover hidden connections, and let the machine surface what the ear already knows.
</p></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — NEBULA MAP
# ══════════════════════════════════════════════════════════════════════════════
if tab == "Nebula Map":
    st.markdown("## Nebula Map of Proximities")
    st.caption("UMAP 2-D projection — each point a track, each cluster a sonic family. Hover for details.")

    if "umap_x" not in df.columns:
        st.warning("UMAP coordinates not found. Re-run the pipeline.")
        st.stop()

    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        color_by = st.selectbox("Colour by", ["cluster"] + DESCRIPTOR_COLS, index=0)
    with c2:
        pt_size = st.slider("Size", 4, 18, 8)
    with c3:
        hl_tag = st.selectbox(
            "Highlight tag",
            ["—"] + sorted([c.replace("tag_", "") for c in df.columns if c.startswith("tag_")]),
        )

    fig = go.Figure()

    if color_by == "cluster" and "cluster" in df.columns:
        for idx, cl in enumerate(sorted(df["cluster"].unique())):
            sub   = df[df["cluster"] == cl]
            hover = [
                f"<b>{r['track_id']}</b><br>"
                + (f"{r.get('key','')} {r.get('camelot','')} · {float(r.get('tempo',0)):.0f} bpm<br>" if "key" in r.index else "")
                + "<br>".join(f"{c}: {r[c]:.2f}" for c in DESCRIPTOR_COLS if c in r.index)
                + (f"<br><i>{r['tags']}</i>" if "tags" in r.index and pd.notna(r.get("tags")) else "")
                for _, r in sub.iterrows()
            ]
            fig.add_trace(go.Scatter(
                x=sub["umap_x"], y=sub["umap_y"], mode="markers",
                name=f"cluster {cl}",
                marker=dict(
                    size=pt_size,
                    color=CLUSTER_COLORS[idx % len(CLUSTER_COLORS)],
                    opacity=0.92,
                    line=dict(width=0.6, color="rgba(255,255,255,0.18)"),
                ),
                text=hover, hovertemplate="%{text}<extra></extra>",
            ))
    else:
        col   = color_by if color_by in df.columns else "brightness"
        hover = [
            f"<b>{r['track_id']}</b><br>" +
            "<br>".join(f"{c}: {r[c]:.2f}" for c in DESCRIPTOR_COLS if c in r.index)
            for _, r in df.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=df["umap_x"], y=df["umap_y"], mode="markers",
            marker=dict(
                size=pt_size, color=df[col],
                colorscale=[[0, "#0d0800"], [0.5, "#6b4f28"], [1, "#c9a84c"]],
                showscale=True,
                colorbar=dict(title=dict(text=col, font=dict(family="Inconsolata", size=9, color="#aaa")), tickfont=dict(family="Inconsolata", color="#aaa", size=8)),
                opacity=0.78,
                line=dict(width=0.3, color="rgba(255,255,255,0.04)"),
            ),
            text=hover, hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    if hl_tag != "—":
        tc = f"tag_{hl_tag}"
        if tc in df.columns:
            sub_t = df[df[tc] == True]
            fig.add_trace(go.Scatter(
                x=sub_t["umap_x"], y=sub_t["umap_y"], mode="markers",
                name=f"◈ {hl_tag}",
                marker=dict(
                    size=pt_size + 7, color="rgba(0,0,0,0)",
                    symbol="circle-open",
                    line=dict(width=1.5, color="#c9a84c"),
                ),
                text=sub_t["track_id"],
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))

    fig.update_layout(height=700, **PLOT_BASE)
    st.plotly_chart(fig, use_container_width=True)

    if "cluster" in df.columns:
        with st.expander("cluster breakdown"):
            for cl in sorted(df["cluster"].unique()):
                sub = df[df["cluster"] == cl]
                top = (
                    ", ".join(
                        sub["tags"].dropna().str.split(", ").explode()
                        .value_counts().head(3).index.tolist()
                    ) if "tags" in sub.columns else "—"
                )
                st.markdown(
                    f'<div class="corsai-sub" style="margin:4px 0">'
                    f'cluster {cl} · {len(sub)} tracks · {top}</div>',
                    unsafe_allow_html=True,
                )
    # ── single-feature nebula ──────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### Single-feature nebula")
    st.caption("Colour the map by one descriptor and filter to tracks above a threshold.")

    sf1, sf2 = st.columns([2, 2])
    with sf1:
        feature_focus = st.selectbox("Feature", DESCRIPTOR_COLS, key="sf_feat")
    with sf2:
        threshold = st.slider("Minimum value", 0.0, 1.0, 0.0, step=0.05, key="sf_thresh")

    if feature_focus in df.columns:
        df_focus = df[df[feature_focus] >= threshold].copy()
        df_muted = df[df[feature_focus] <  threshold].copy()

        fig_sf = go.Figure()

        if not df_muted.empty:
            fig_sf.add_trace(go.Scatter(
                x=df_muted["umap_x"], y=df_muted["umap_y"],
                mode="markers", name="below threshold",
                marker=dict(size=5, color="#2a2a2a", opacity=0.7,
                            line=dict(width=0.3, color="#383838")),
                text=df_muted["track_id"],
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))

        if not df_focus.empty:
            hover_sf = [
                f"<b>{r['track_id']}</b><br>"
                f"{feature_focus}: {r[feature_focus]:.2f}<br>"
                + (f"{r.get('key','')} {r.get('camelot','')} · {float(r.get('tempo',0)):.0f} bpm<br>" if "key" in r.index else "")
                + "<br>".join(f"{c}: {r[c]:.2f}" for c in DESCRIPTOR_COLS if c in r.index and c != feature_focus)
                + (f"<br><i>{r['tags']}</i>" if "tags" in r.index and pd.notna(r.get("tags")) else "")
                for _, r in df_focus.iterrows()
            ]
            fig_sf.add_trace(go.Scatter(
                x=df_focus["umap_x"], y=df_focus["umap_y"],
                mode="markers", name=f"{feature_focus} ≥ {threshold}",
                marker=dict(
                    size=pt_size + 2,
                    color=df_focus[feature_focus],
                    colorscale=[[0, "#3d2800"], [0.3, "#a07840"], [0.6, "#c9a84c"], [0.85, "#e8cc80"], [1, "#fff8dc"]],
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=feature_focus, font=dict(family="Inconsolata", size=9, color="#aaa")),
                        tickfont=dict(family="Inconsolata", size=8, color="#aaa"),
                    ),
                    opacity=0.95,
                    line=dict(width=0.5, color="rgba(255,255,255,0.12)"),
                ),
                text=hover_sf,
                hovertemplate="%{text}<extra></extra>",
            ))

        fig_sf.update_layout(height=680, **PLOT_BASE)
        st.plotly_chart(fig_sf, use_container_width=True)

        st.markdown(
            f'<div class="corsai-sub">{len(df_focus)} tracks with {feature_focus} ≥ {threshold} · {len(df_muted)} muted</div>',
            unsafe_allow_html=True,
        )



# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════
elif tab == "Similarity":
    st.markdown("## Similarity")
    st.caption("Select a track to surface its nearest neighbours in acoustic space.")

    cs, ck = st.columns([4, 1])
    with cs:
        selected = st.selectbox("Track", df["track_id"].tolist(), label_visibility="collapsed")
    with ck:
        k = st.number_input("k", min_value=3, max_value=30, value=10)

    sel_row = df[df["track_id"] == selected].iloc[0]
    st.markdown("### Selected")
    track_card(sel_row)

    if "umap_x" in df.columns:
        similar_df   = get_similar(selected, k=k)
        neighbor_ids = similar_df["track_id"].tolist() if not similar_df.empty else []

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["umap_x"], y=df["umap_y"], mode="markers", name="corpus",
            marker=dict(size=4, color="#161616", opacity=0.7,
                        line=dict(width=0.3, color="#242424")),
            text=df["track_id"], hovertemplate="<b>%{text}</b><extra></extra>",
        ))
        if neighbor_ids:
            nbr = df[df["track_id"].isin(neighbor_ids)]
            fig2.add_trace(go.Scatter(
                x=nbr["umap_x"], y=nbr["umap_y"], mode="markers", name="neighbours",
                marker=dict(size=11, color="rgba(201,168,76,0.55)",
                            line=dict(width=1.2, color="#c9a84c")),
                text=nbr["track_id"], hovertemplate="<b>%{text}</b><extra></extra>",
            ))
        fig2.add_trace(go.Scatter(
            x=[sel_row["umap_x"]], y=[sel_row["umap_y"]],
            mode="markers", name="selected",
            marker=dict(size=15, color="#f0ede8", symbol="circle",
                        line=dict(width=2, color="#c9a84c")),
            text=[selected], hovertemplate="<b>%{text}</b><extra></extra>",
        ))
        fig2.update_layout(height=300, **PLOT_BASE)

        st.markdown('<div class="map-frame">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"### {k} nearest")
    similar_df = get_similar(selected, k=k)
    if similar_df.empty:
        st.info("No similarity data.")
    else:
        for _, row in similar_df.iterrows():
            track_card(row, show_distance=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE PROFILE
# ══════════════════════════════════════════════════════════════════════════════
elif tab == "Feature Profile":
    st.markdown("## Feature Profile")
    st.caption("Compare two tracks descriptor by descriptor, or find tracks close to a target value on a single feature.")

    st.markdown("### Track compatibility")
    ca, cb = st.columns(2)
    with ca:
        track_a = st.selectbox("Track A", df["track_id"].tolist(), key="ta")
    with cb:
        track_b = st.selectbox("Track B", df["track_id"].tolist(),
                               index=min(1, len(df)-1), key="tb")

    if track_a and track_b and track_a != track_b:
        row_a = df[df["track_id"] == track_a].iloc[0]
        row_b = df[df["track_id"] == track_b].iloc[0]

        profile = {}
        for d in DESCRIPTOR_COLS:
            if d in row_a.index and d in row_b.index:
                profile[d] = round(1 - abs(float(row_a[d]) - float(row_b[d])), 3)
        if "tempo" in row_a.index:
            bpm_diff         = abs(float(row_a["tempo"]) - float(row_b["tempo"]))
            profile["bpm"]   = round(max(0, 1 - bpm_diff / 20), 3)
        if "camelot" in row_a.index:
            profile["key"]   = camelot_compat(str(row_a.get("camelot","")), str(row_b.get("camelot","")))

        overall = sum(profile.values()) / len(profile)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f'<div class="track-name">{track_a}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="track-meta">'
                f'{row_a.get("key","")} {row_a.get("camelot","")} · '
                f'{float(row_a.get("tempo",0)):.0f} bpm</div>',
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(f'<div class="track-name">{track_b}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="track-meta">'
                f'{row_b.get("key","")} {row_b.get("camelot","")} · '
                f'{float(row_b.get("tempo",0)):.0f} bpm</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        dims   = list(profile.keys())
        scores = list(profile.values())
        colors = ["#c9a84c" if s > 0.7 else "#5c5c4a" if s > 0.4 else "#2a1a1a" for s in scores]

        fig_p = go.Figure(go.Bar(
            x=scores, y=dims,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{s:.2f}" for s in scores],
            textposition="outside",
            textfont=dict(family="Inconsolata", size=10, color="#888"),
        ))
        fig_p.add_vline(x=0.7, line=dict(color="#c9a84c", width=0.8, dash="dot"))
        fig_p.update_layout(
            height=380,
            xaxis=dict(range=[0, 1.15], showgrid=False, zeroline=False,
                       tickfont=dict(family="Inconsolata", size=9, color="#555")),
            yaxis=dict(showgrid=False,
                       tickfont=dict(family="Inconsolata", size=10, color="#888")),
            plot_bgcolor="#040404",
            paper_bgcolor="#080808",
            margin=dict(l=10, r=70, t=10, b=10),
            showlegend=False,
            hoverlabel=HOVER_STYLE,
        )
        st.plotly_chart(fig_p, use_container_width=True)
        st.markdown(
            f'<div class="corsai-sub">Overall compatibility: {overall:.2f}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("### Find tracks by descriptor value")
    st.caption("Surface tracks where a specific descriptor is close to a target value.")

    fd1, fd2, fd3 = st.columns([2, 2, 1])
    with fd1:
        target_desc = st.selectbox("Descriptor", DESCRIPTOR_COLS, key="td")
    with fd2:
        target_val  = st.slider("Target value", 0.0, 1.0, 0.7, step=0.05, key="tv")
    with fd3:
        tolerance   = st.slider("±", 0.05, 0.3, 0.1, step=0.05, key="tol")

    if target_desc in df.columns:
        mask    = (df[target_desc] - target_val).abs() <= tolerance
        results = df[mask].copy()
        results["dist_to_target"] = (results[target_desc] - target_val).abs()
        results = results.sort_values("dist_to_target").head(20)

        st.markdown(
            f'<div class="corsai-sub" style="margin:10px 0">'
            f'{len(results)} tracks with {target_desc} ≈ {target_val} (±{tolerance})</div>',
            unsafe_allow_html=True,
        )

        dcols = (["track_id", target_desc] +
                 [c for c in DESCRIPTOR_COLS if c != target_desc and c in results.columns] +
                 [c for c in ["tempo", "key", "camelot", "tags"] if c in results.columns])
        st.dataframe(
            results[dcols].reset_index(drop=True).style.format(
                {c: "{:.2f}" for c in DESCRIPTOR_COLS if c in results.columns}
            ),
            use_container_width=True,
            height=min(50 + 35 * len(results), 450),
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SEARCH & FILTER
# ══════════════════════════════════════════════════════════════════════════════
elif tab == "Search & Filter":
    st.markdown("## Search & Filter")
    st.caption("Filter by acoustic descriptors, key, BPM, cluster, or tags.")

    with st.expander("descriptors", expanded=True):
        fc                 = st.columns(4)
        descriptor_filters = {}
        for i, desc in enumerate(DESCRIPTOR_COLS):
            if desc in df.columns:
                with fc[i % 4]:
                    lo, hi = st.slider(desc, 0.0, 1.0, (0.0, 1.0), step=0.05, key=f"sl_{desc}")
                    descriptor_filters[desc] = (lo, hi)

    with st.expander("key / bpm / cluster / tag"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            k_opts  = ["Any"] + sorted(df["key"].dropna().unique().tolist()) if "key" in df.columns else ["Any"]
            sel_key = st.selectbox("Key", k_opts)
        with c2:
            if "tempo" in df.columns:
                bpm_range = st.slider("BPM", float(df["tempo"].min()),
                                      float(df["tempo"].max()),
                                      (float(df["tempo"].min()), float(df["tempo"].max())),
                                      step=1.0)
            else: bpm_range = None
        with c3:
            if "cluster" in df.columns:
                cl_opts = ["Any"] + [str(c) for c in sorted(df["cluster"].unique())]
                sel_cl  = st.selectbox("Cluster", cl_opts)
            else: sel_cl = "Any"
        with c4:
            tl      = [c.replace("tag_", "") for c in df.columns if c.startswith("tag_")]
            sel_tag = st.selectbox("Tag", ["Any"] + sorted(tl))

    with st.expander("search by name"):
        name_q = st.text_input("contains", "", label_visibility="collapsed")

    mask = pd.Series([True] * len(df), index=df.index)
    for desc, (lo, hi) in descriptor_filters.items():
        if desc in df.columns:
            mask &= (df[desc] >= lo) & (df[desc] <= hi)
    if sel_key != "Any" and "key" in df.columns:
        mask &= df["key"] == sel_key
    if bpm_range and "tempo" in df.columns:
        mask &= (df["tempo"] >= bpm_range[0]) & (df["tempo"] <= bpm_range[1])
    if sel_cl != "Any" and "cluster" in df.columns:
        mask &= df["cluster"].astype(str) == sel_cl
    if sel_tag != "Any":
        tc = f"tag_{sel_tag}"
        if tc in df.columns: mask &= df[tc] == True
    if name_q:
        mask &= df["track_id"].str.contains(name_q, case=False, na=False)

    filtered = df[mask].copy()
    st.markdown(
        f'<div class="corsai-sub" style="margin:14px 0">{len(filtered)} tracks</div>',
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.info("No tracks match.")
    else:
        sc1, sc2 = st.columns([2, 1])
        with sc1:
            sort_by = st.selectbox("Sort by", ["track_id"] + DESCRIPTOR_COLS + ["tempo", "key"], index=0)
        with sc2:
            sort_asc = st.radio("Order", ["↑", "↓"]) == "↑"
        if sort_by in filtered.columns:
            filtered = filtered.sort_values(sort_by, ascending=sort_asc)

        dcols  = ["track_id"] + [c for c in ["key","camelot","tempo","cluster","tags"] if c in filtered.columns]
        ddesc  = [c for c in DESCRIPTOR_COLS if c in filtered.columns]
        dcols += ddesc

        st.dataframe(
            filtered[dcols].reset_index(drop=True).style.format({c: "{:.2f}" for c in ddesc}),
            use_container_width=True,
            height=min(50 + 35 * len(filtered), 500),
        )
        st.download_button(
            label="export csv",
            data=filtered[dcols].to_csv(index=False),
            file_name="corsai_filtered.csv",
            mime="text/csv",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MANUAL TAGS
# ══════════════════════════════════════════════════════════════════════════════
elif tab == "Manual Tags":
    st.markdown("## Manual Tags")
    st.caption("Override computed tags with your own. Changes persist across sessions and pipeline re-runs.")

    df_manual = load_manual_tags()
    already_tagged = set(df_manual["track_id"].tolist()) if not df_manual.empty else set()

    # track selector
    mt1, mt2 = st.columns([3, 1])
    with mt1:
        tag_track = st.selectbox("Select track to tag", df["track_id"].tolist(), key="mt_track")
    with mt2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_all = st.checkbox("Show all tagged", value=False)

    # current state for selected track
    sel_row    = df[df["track_id"] == tag_track].iloc[0]
    auto_tags  = sel_row.get("tags", "unclassified")
    manual_row = df_manual[df_manual["track_id"] == tag_track]
    current_manual = manual_row["manual_tags"].iloc[0] if not manual_row.empty else ""
    current_notes  = manual_row["notes"].iloc[0] if not manual_row.empty else ""

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # show track info
    track_card(sel_row)

    col_auto, col_manual = st.columns(2)
    with col_auto:
        st.markdown('<div class="corsai-sub" style="margin-bottom:6px">Computed tags</div>', unsafe_allow_html=True)
        st.markdown(
            "".join(f'<span class="tag-chip">{t.strip()}</span>' for t in str(auto_tags).split(",")),
            unsafe_allow_html=True,
        )

    with col_manual:
        st.markdown('<div class="corsai-sub" style="margin-bottom:6px">Your tags</div>', unsafe_allow_html=True)
        if current_manual:
            st.markdown(
                "".join(f'<span class="tag-chip" style="border-color:#c9a84c;color:#c9a84c">{t.strip()}</span>'
                        for t in str(current_manual).split(",")),
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="corsai-sub">none yet</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # available tags from TAGS dict — pull from algorithm tags in df
    all_algo_tags = sorted(set(
        t.strip()
        for tags in df["tags"].dropna()
        for t in str(tags).split(",")
        if t.strip() != "unclassified"
    ))
    custom_tags   = ["ambient", "hypnotic", "experimental", "peak_time",
                 "warm_up", "closing", "b2b", "tool_track", "percussive"]
    tag_options   = sorted(set(all_algo_tags + custom_tags))

    new_tags = st.multiselect(
        "Select tags",
        options=tag_options,
        default=[t.strip() for t in str(current_manual).split(",") if t.strip()] if current_manual else [],
        key="mt_tags",
    )
    custom_input = st.text_input(
        "Or type custom tags (comma separated)",
        value="",
        placeholder="e.g. late_night, driving, b2b_friendly",
        key="mt_custom",
    )
    notes_input = st.text_input(
        "Notes (optional)",
        value=str(current_notes) if pd.notna(current_notes) and current_notes else "",
        placeholder="Any context about this tag decision",
        key="mt_notes",
    )

    bcol1, bcol2 = st.columns([1, 1])
    with bcol1:
        if st.button("Save tags"):
            combined = list(new_tags)
            if custom_input:
                combined += [t.strip() for t in custom_input.split(",") if t.strip()]
            tag_str = ", ".join(combined) if combined else "unclassified"

            if not df_manual.empty and tag_track in df_manual["track_id"].values:
                df_manual.loc[df_manual["track_id"] == tag_track, "manual_tags"] = tag_str
                df_manual.loc[df_manual["track_id"] == tag_track, "notes"]       = notes_input
            else:
                new_row = pd.DataFrame([{"track_id": tag_track,
                                         "manual_tags": tag_str,
                                         "notes": notes_input}])
                df_manual = pd.concat([df_manual, new_row], ignore_index=True)

            save_manual_tags(df_manual)
            st.success(f"Saved: {tag_str}")
            st.cache_data.clear()

    with bcol2:
        if not manual_row.empty:
            if st.button("Clear manual tags"):
                df_manual = df_manual[df_manual["track_id"] != tag_track]
                save_manual_tags(df_manual)
                st.success("Manual tags cleared.")
                st.cache_data.clear()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # all manually tagged tracks
    if show_all and not df_manual.empty:
        st.markdown("### All manually tagged tracks")
        st.dataframe(
            df_manual.reset_index(drop=True),
            use_container_width=True,
            height=min(50 + 35 * len(df_manual), 500),
        )
        st.download_button(
            label="export manual tags",
            data=df_manual.to_csv(index=False),
            file_name="manual_tags.csv",
            mime="text/csv",
        )
