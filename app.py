import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go

#Set Configuration
st.set_page_config(
    page_title='Kepler Exoplanet Classification',
    page_icon='🌎',
    layout='wide'
)

COLOR_MAP = {
    'CONFIRMED': '#4CAF50',
    'FALSE POSITIVE': '#F44336',
    'CANDIDATE': '#FF9800'
}

BASE = r'C:\Users\adity\Desktop\Kepler-Exoplanet Project'
DASHBOARD_DATA = os.path.join(BASE, 'dashboard_data')

#Data Loaders
@st.cache_data
def load_raw_data():
    return pd.read_csv(os.path.join(BASE, 'data', 'kepler_clean.csv'))

@st.cache_data
def load_model_metrics():
    with open(os.path.join(DASHBOARD_DATA, 'metrics.json')) as f:
        return json.load(f)

@st.cache_data
def load_predictions():
    with open(os.path.join(DASHBOARD_DATA, 'predictions.json')) as f:
        return json.load(f)

@st.cache_data
def load_test_metadata():
    meta = pd.read_csv(os.path.join(DASHBOARD_DATA, 'test_metadata.csv'))
    y_test = np.load(os.path.join(DASHBOARD_DATA, 'y_test.npy'))
    return meta, y_test

@st.cache_data
def load_shap(tier):
    fname = 'shap_values_full.npy' if tier == 'Full' else 'shap_values_raw.npy'
    xname = 'X_test_full.csv' if tier == 'Full' else 'X_test_raw.csv'
    shap_vals = np.load(os.path.join(DASHBOARD_DATA, fname))
    X_test = pd.read_csv(os.path.join(DASHBOARD_DATA, xname)) 
    return shap_vals, X_test

df = load_raw_data()

st.title("🪐 Kepler Exoplanet Classification Dashboard")

tab_explorer, tab_models, tab_shap, tab_misclass, tab_audit = st.tabs(
    ["🔭 Data Explorer", "📊 Model Comparison", "🧠 SHAP Explorer",
    "❌ Misclassification Bowser", "🔧 Methodology Audit"]
)

# TAB 1 - DATA EXPLORER
with tab_explorer:
    st.sidebar.title("🌎 Kepler Explorer")
    st.sidebar.markdown("Filter and explore the Kepler exoplanet dataset")

    selected_classes = st.sidebar.multiselect(
        "Disposition classes",
        options=["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"],
        default=["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"]
    )
    period_range = st.sidebar.slider(
        "Orbital period range (days)",
        min_value=0.0,
        max_value=float(df["koi_period"].quantile(0.99)),
        value=(0.0, float(df["koi_period"].quantile(0.95)))
    )

    radius_range = st.sidebar.slider(
        "Planet radius range (Earth Radii)",
        min_value=0.0,
        max_value=float(df["koi_prad"].quantile(0.99)),
        value=(0.0, float(df["koi_prad"].quantile(0.95)))
    )

    x_axis = st.sidebar.selectbox(
        "X axis (scatter plot)",
        ["koi_period", "koi_prad", "koi_teq", "koi_insol", "koi_model_snr", "koi_score"],
        index=0
    )

    y_axis = st.sidebar.selectbox(
        "Y axis (scatter plot)",
        ["koi_prad", "koi_period", "koi_teq", "koi_insol", "koi_model_snr", "koi_score"],
        index = 0
    )

    filtered = df[
        df["koi_disposition"].isin(selected_classes) &
        df["koi_period"].between(*period_range) &
        df["koi_prad"].between(*radius_range)
    ]

    st.markdown(f"Showing **{len(filtered):,}** of **{len(df):,}** objects")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("✅ Confirmed", len(filtered[filtered["koi_disposition"] == "CONFIRMED"]))
    col2.metric("❌ False Positives", len(filtered[filtered["koi_disposition"] == "FALSE POSITIVE"]))
    col3.metric("🔍 Candidates", len(filtered[filtered["koi_disposition"] == "CANDIDATE"]))
    col4.metric("Avg. Planet Radius", f"{filtered["koi_prad"].mean():.2f} R⊕")

    st.markdown("---")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader(f"{x_axis} vs {y_axis}")
        fig_scatter = px.scatter(
            filtered, x=x_axis, y=y_axis,
            color="koi_disposition", color_discrete_map=COLOR_MAP,
            hover_data=["koi_teq", "koi_insol", "koi_score"],
            log_x=True, log_y=True, opacity=0.6, template="plotly_dark",
            labels={"koi_disposition": "Disposition"}
        )
        fig_scatter.update_traces(marker=dict(size=4))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        st.subheader("Class Breakdown")
        counts = filtered["koi_disposition"].value_counts().reset_index()
        counts.columns = ["Disposition", "Count"]
        fig_pie = px.pie(counts, values="Count", names="Disposition",
                  color="Disposition", color_discrete_map=COLOR_MAP,
                  template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Feature distribution by class")
    feat = st.selectbox(
        "Select feature",
        ["koi_prad", "koi_period", "koi_teq", "koi_insol",
        "koi_model_snr", "koi_score", "koi_duration", "koi_depth"]
    )

    if feat in ["koi_prad", "koi_period", "koi_insol"]:
        plot_df = filtered[[feat, "koi_disposition"]].dropna()
        plot_df = plot_df[plot_df[feat] > 0]
        if len(plot_df) > 0:
            bins = np.logspace(np.log10(plot_df[feat].min()), np.log10(plot_df[feat].max()), 61)
            fig_hist = go.Figure()
            for disposition, color in COLOR_MAP.items():
                subset = plot_df[plot_df["koi_disposition"] == disposition][feat]
                counts_hist, edges = np.histogram(subset, bins=bins)
                fig_hist.add_trace(go.Bar(
                    x=edges[:-1], y=counts_hist, name=disposition, marker_color=color,
                    opacity=0.75, width=np.diff(edges), offset=0
                ))
            fig_hist.update_layout(
                barmode='overlay', bargap=0,
                xaxis=dict(title=feat + ' (log scale)', type='log'),
                yaxis_title='Count', template='plotly_dark', legend_title='Disposition'
            )
        else:
            fig_hist = px.histogram(
                filtered, x=feat, color="koi_dispositon", color_discrete_map=COLOR_MAP,
                barmode="overlay", opacity=0.7, nbins=60, template="plotly_dark",
                labels={"koi_dispostion": "Disposition"}
            )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Correlation Heatmap")
        key_feats = ["koi_period", "koi_prad", "koi_teq", "koi_insol",
                "koi_model_snr", "koi_steff", "koi_slogg",
                "koi_srad", "koi_score", "koi_depth"]
        corr = filtered[key_feats].corr().round(2)
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.to_list(), y=corr.columns.to_list(),
            colorscale="RdBu_r", zmid=0,
            text=corr.values, texttemplate="%{text}", textfont={"size":9}
        ))
        fig_heat.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("📄 View filtered data table"):
            st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

# ---------------------------------------------
# TAB 2 - MODEL COMPARISON (tier toggle)
# ---------------------------------------------
with tab_models:
    st.header("Model Performance Across Feature Tiers")
    st.markdown(
        "Compare how each model performs depending on which features it's allowed to see -"
        "from the full feature set down to raw physical measurements only."
    )

    metrics = load_model_metrics()
    tier_choice = st.radio(
        "Feature tier", options=list(metrics.keys()), horizontal=True,
        help="Full = all 16 features. No koi_score = drops the vetting score. "
        "Raw Physical = drops koi_score AND all koi_fpflag_* vetting flags."
    )

    tier_metrics = metrics[tier_choice]
    metric_df = pd.DataFrame([
        {'Model': name, 'F1 Score': vals['f1'], 'ROC-AUC': vals['auc']}
        for name, vals in tier_metrics.items()
    ]).sort_values('F1 Score', ascending=False).reset_index(drop=True)

    col1, col2, = st.columns([1, 1])
    with col1:
        st.dataframe(
            metric_df.style.format({'F1 Score': '{:.4f}', 'ROC-AUC': '{:.4f}'})
                        .background_gradient(subset=['F1 Score', 'ROC-AUC'], cmap='Greens'),
            use_container_width=True, hide_index=True
        )
    with col2:
        fig_bar = px.bar(
            metric_df, x='Model', y='F1 Score', color='Model',
            template='plotly_dark', text_auto='.4f'
        )
        fig_bar.update_layout(showlegend=False, yaxis_range=[metric_df['F1 Score'].min() - 0.02, 1.0])
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Confusion Matrix")
    model_choice = st.selectbox("Select model", options=list(tier_metrics.keys()))
    cm = np.array(tier_metrics[model_choice]['confusion_matrix'])
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=['CONFIRMED', 'FALSE POSITIVE'], y=['CONFIRMED', 'FALSE POSITIVE'],
        colorscale='Blues', text=cm, texttemplate="%{text}", textfont={"size": 16}
    ))
    fig_cm.update_layout(template='plotly_dark', xaxis_title='Predicted', yaxis_title='True', height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

    if tier_choice == 'Raw Physical':
        st.info(
            "📌 **Key finding**: dropping `koi_score` and all `koi_fpflag_*` columns causes a "
            "statistically significant drop in performance (McNemar's p < 0.0001) — these vetting-"
            "pipeline-derived features account for a real, meaningful share of the near-perfect "
            "full-feature scores. Raw physical measurements alone still yield a strong classifier "
            "(best model here: XGBoost, F1 ≈ 0.94), just not as strong as with vetting features included."
        )

# -----------------------------------------------
# TAB 3 - SHAP EXPLORER
# -----------------------------------------------
with tab_shap:
    st.header("SHAP Feature Importance")
    shap_tier = st.radio("Feature tier", options=['Full', 'Raw Physical'], horizontal=True, key='shap_tier')

    shap_values, X_test_shap = load_shap(shap_tier)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'Feature': X_test_shap.columns, 'Mean |SHAP value|': mean_abs_shap
    }).sort_values('Mean |SHAP value|', ascending=False)

    fig_shap = px.bar(
        shap_importance, x='Mean |SHAP value|', y='Feature', orientation='h',
        template='plotly_dark', color='Mean |SHAP value|', color_continuous_scale='Blues'
    )
    fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_shap, use_container_width=True)

# ------------------------------------------------
# TAB 4 - MISCLASSIFICATION BROWSER
# ------------------------------------------------
with tab_misclass:
    st.header("Misclassified Objects (Full Feature Model)")

    meta, y_test = load_test_metadata()
    predictions = load_predictions()
    model_for_misclass = st.selectbox(
        "Model", options=list(predictions['Full'].keys()), key='misclass_model'
    )
    y_pred = np.array(predictions['Full'][model_for_misclass]['y_pred'])

    meta = meta.copy()
    meta['predicted'] = np.where(y_pred == 0, 'CONFIRMED', 'FALSE POSITIVE')
    meta['correct'] = (y_pred == y_test)
    misclassified = meta[~meta['correct']].drop(columns='correct')

    st.markdown(f"**{len(misclassified)}** misclassified out of **{len(meta)}** test objects")
    st.dataframe(misclassified.reset_index(drop=True), use_container_width=True)

    if len(misclassified) > 0:
        correct_scores = meta.loc[meta['correct'], 'koi_score']
        mis_scores = meta.loc[~meta['correct'], 'koi_score']

        fig_score = go.Figure()

        # Distribution of correct scores as a histogram
        fig_score.add_trace(go.Histogram(
            x=correct_scores, name='Correct', marker_color='#4CAF50',
            opacity=0.6, nbinsx=40
        ))

        # Each misclassified point as a full-height vertical line
        for i, score in enumerate(mis_scores):
            fig_score.add_vline(
                x=score, line_color='#F44336', line_width=2, line_dash='dash',
                annotation_text=f'{score:.3f}', annotation_position='top',
                annotation_font_color='#F44336', annotation_font_size=10
            )

        fig_score.update_layout(
            template='plotly_dark',
            xaxis_title='koi_score',
            yaxis_title='Count (Correct)',
            yaxis_type='log',
            title=f'koi_score Distribution — {len(mis_scores)} Misclassified (red) vs. {len(correct_scores)} Correct (green)',
            showlegend=False,
            height=450
        )
        st.plotly_chart(fig_score, use_container_width=True)
# -------------------------------------------------
# TAB 5 - METHODOLOGY AUDIT
# -------------------------------------------------
with tab_audit:
    st.header("🔧 Methodology Audit - Bugs Found & Fixed")
    st.markdown(
        "This project went through a rigorous validation process. Six issues were identified "
        "and corrected during development, each verified with before/after results."
    )

    bugs = [
        ("McNemar's test 0/0 edge case",
        "An operator precedence bug (`&` vs `and`) caused identical-prediction pairs to be "
        "misreported as statistically significant. Fixed with an explicit guard and re-verified."),
        ("SVM trained on wrong feature scope (Cell 6)",
        "The 'full-feature' model comparison accidentally trained SVM on physical-only scaled "
        "data due to a leftover-variable dependency that only worked when cells ran out of order."),
        ("SVM trained on wrong feature scope (Cell 11)",
        "The mirror-image bug: the 'physical-only' ablation accidentally trained SVM on full-"
        "feature scaled data, canceling out the Cell 6 bug and masking both."),
        ("SMOTE applied before the CV split",
        "5-fold cross-validation was run on already-SMOTE'd data, letting synthetic points leak "
        "across folds. Corrected AUC: 0.9987 (down from an inflated 0.9996)."),
        ("Imputer fit before the CV split",
        "Median imputation used the full dataset before folds were created. Fixed by moving it "
        "inside the CV pipeline — verified to have negligible impact."),
        ("Host-star leakage across train/test",
        "17.4% of test rows shared a host star with a training row. Switched to a star-grouped "
        "split (GroupShuffleSplit) — verified negligible performance change, confirming the "
        "model wasn't relying on memorized stellar parameters."),
    ]

    for i, (title, desc) in enumerate(bugs, 1):
        with st.expander(f"Bug #{i}: {title}"):
            st.write(desc)

    st.markdown("---")
    st.subheader("Key finding: what actually drives the ~99% accuracy?")
    st.markdown(
        """
        A stricter ablation — dropping `koi_score` **and** all `koi_fpflag_*` columns —
        revealed that these vetting-pipeline-derived features account for a real, statistically
        significant share of the near-perfect full-feature performance (McNemar's p < 0.0001).

        **However**, raw physical measurements alone (period, radius, depth, duration, SNR,
        stellar parameters) still support a genuinely strong classifier (F1 ≈ 0.93–0.94),
        confirming real, independently learnable physical signal — not just agreement with
        Kepler's existing automated classifier.
        """
    )