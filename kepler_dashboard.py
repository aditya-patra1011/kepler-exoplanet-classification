
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---- PAGE CONFIGURATION -------
st.set_page_config(
  page_title = 'Kepler Exoplanet Search',
  page_icon = '🌎',
  layout = 'wide'
)


COLOR_MAP = {
    'CONFIRMED': '#4CAF50',
    'FALSE POSITIVE': '#F44336',
    'CANDIDATE': '#FF9800'
}

# ---- LOAD DATA -----------------
@st.cache_data
def load_data():
  df = pd.read_csv('kepler_clean.csv')
  return df

df = load_data()

# ---- SIDEBAR CONTROLS ----------
st.sidebar.title("🌎 Kepler Explorer")
st.sidebar.markdown("Filter and explore the kepler exoplanet dataset.")

selected_classes = st.sidebar.multiselect(
  "Disposition classes",
  options = ["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"],
  default = ["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"]
)

period_range = st.sidebar.slider(
  "Orbital period range (days)",
  min_value = 0.0,
  max_value = float(df["koi_period"].quantile(0.99)),   # FIX 1: typo 'quaantile' → 'quantile'
  value = (0.0, float(df["koi_period"].quantile(0.95)))
)

# FIX 2: radius_range slider was missing entirely — added it here
radius_range = st.sidebar.slider(
  "Planet radius range (Earth radii)",
  min_value = 0.0,
  max_value = float(df["koi_prad"].quantile(0.99)),
  value = (0.0, float(df["koi_prad"].quantile(0.95)))
)

x_axis = st.sidebar.selectbox(
  "X axis (scatter plot)",
  ["koi_period", "koi_prad", "koi_teq", "koi_insol", "koi_model_snr", "koi_score"],
  index = 0
)

y_axis = st.sidebar.selectbox(
  "Y axis (scatter plot)",
  ["koi_prad", "koi_period", "koi_teq", "koi_insol", "koi_model_snr", "koi_score"],
  index = 0
)

# ---- Filter Data ------------
filtered = df[
    df["koi_disposition"].isin(selected_classes) &
    df["koi_period"].between(*period_range) &
    df["koi_prad"].between(*radius_range)
]

# ---- Header --------------------
st.title("🪐 Kepler Exoplanet Search Results — Explorer")
st.markdown(f"Showing **{len(filtered):,}** of **{len(df):,}** objects")

# ---- Metric row ------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("✅ Confirmed",       len(filtered[filtered["koi_disposition"] == "CONFIRMED"]))
col2.metric("❌ False Positives", len(filtered[filtered["koi_disposition"] == "FALSE POSITIVE"]))
col3.metric("🔍 Candidates",      len(filtered[filtered["koi_disposition"] == "CANDIDATE"]))
col4.metric("Avg. Planet Radius", f'{filtered["koi_prad"].mean():.2f} R⊕')  # FIX 3: nested quotes fixed

st.markdown("---")

# ---- Scatter plot -----------------------------------------
col_a, col_b = st.columns([2, 1])

with col_a:
    st.subheader(f"{x_axis} vs {y_axis}")
    fig_scatter = px.scatter(
        filtered, x=x_axis, y=y_axis,
        color="koi_disposition",
        color_discrete_map=COLOR_MAP,
        hover_data=["koi_teq", "koi_insol", "koi_score"],
        log_x=True, log_y=True,
        opacity=0.6, template="plotly_dark",
        labels={"koi_disposition": "Disposition"}
    )
    fig_scatter.update_traces(marker=dict(size=4))
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_b:
    st.subheader("Class breakdown")
    counts = filtered["koi_disposition"].value_counts().reset_index()
    counts.columns = ["Disposition", "Count"]
    fig_pie = px.pie(counts, values="Count", names="Disposition",
                     color="Disposition", color_discrete_map=COLOR_MAP,
                     template="plotly_dark")
    st.plotly_chart(fig_pie, use_container_width=True)

# ---- Distribution plot --------------------------------------
st.subheader("Feature distribution by class")
feat = st.selectbox(
    "Select feature",
    ["koi_prad", "koi_period", "koi_teq", "koi_insol",
     "koi_model_snr", "koi_score", "koi_duration", "koi_depth"]
)

# FIX 4: same invisible bar bug as phase 3 — use go.Bar with manual bins for log features
if feat in ["koi_prad", "koi_period", "koi_insol"]:
    plot_df = filtered[[feat, "koi_disposition"]].dropna()
    plot_df = plot_df[plot_df[feat] > 0]
    if len(plot_df) > 0:
        bins = np.logspace(
            np.log10(plot_df[feat].min()),
            np.log10(plot_df[feat].max()),
            61
        )
        fig_hist = go.Figure()
        for disposition, color in COLOR_MAP.items():
            subset = plot_df[plot_df["koi_disposition"] == disposition][feat]
            counts_hist, edges = np.histogram(subset, bins=bins)
            fig_hist.add_trace(go.Bar(
                x=edges[:-1], y=counts_hist,
                name=disposition, marker_color=color,
                opacity=0.75, width=np.diff(edges), offset=0
            ))
        fig_hist.update_layout(
            barmode='overlay', bargap=0,
            xaxis=dict(title=feat + ' (log scale)', type='log'),
            yaxis_title='Count',
            template='plotly_dark',
            legend_title='Disposition'
        )
else:
    fig_hist = px.histogram(
        filtered, x=feat, color="koi_disposition",
        color_discrete_map=COLOR_MAP, barmode="overlay",
        opacity=0.7, nbins=60,
        template="plotly_dark",
        labels={"koi_disposition": "Disposition"}
    )
st.plotly_chart(fig_hist, use_container_width=True)

# ---- Correlation heatmap ------------------------------------
st.subheader("Correlation heatmap")
key_feats = ["koi_period", "koi_prad", "koi_teq", "koi_insol",
             "koi_model_snr", "koi_steff", "koi_slogg",
             "koi_srad", "koi_score", "koi_depth"]
corr = filtered[key_feats].corr().round(2)
fig_heat = go.Figure(go.Heatmap(
    z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
    colorscale="RdBu_r", zmid=0,
    text=corr.values, texttemplate="%{text}", textfont={"size": 9}
))
fig_heat.update_layout(template="plotly_dark", height=500)
st.plotly_chart(fig_heat, use_container_width=True)

# ---- Raw data table -----------------------------------------
with st.expander("📄 View filtered data table"):
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)
