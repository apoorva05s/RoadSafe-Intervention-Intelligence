import pathlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from causal import build_evidence
from counterfactual import scenarios
from explain import global_importance, load_bundle, local_explanation
from forecast import baseline_3yr, city_forecast, station_forecasts
from impact import recommend
from ml_model import intervention_relevance, load_crashes, make_features
from optimize import efficient_frontier, optimize_portfolio
from policy import portfolio_actions, station_brief
from risk import DEFAULT_W, compute, explain

DATA = pathlib.Path(__file__).parent / "data"
st.set_page_config(page_title="RoadSafe | Decision Intelligence", page_icon="RS", layout="wide")


def badge(status):
    colors = {"OBSERVED": "#176b87", "ESTIMATED": "#ad6b00", "MODELED": "#7b3f98", "QUASI-EXPERIMENTAL": "#287d46", "LITERATURE": "#6b7280"}
    return f'<span style="background:{colors.get(status, "#64748b")};color:white;padding:3px 8px;border-radius:3px;font-size:0.72rem;font-weight:700">{status}</span>'


def render_table(frame, **kwargs):
    """Render tables even when the launching environment has a broken PyArrow DLL."""
    try:
        st.dataframe(frame, **kwargs)
    except Exception as exc:
        if "arrow" not in str(exc).lower() and "dll" not in str(exc).lower():
            raise
        st.markdown(frame.to_html(index=False, border=0, classes="roadsafe-table"), unsafe_allow_html=True)


@st.cache_data
def load_system(weights):
    df, killed, total = compute(dict(weights))
    # Keep the risk module's attrs-backed city reference before merging ML features.
    df["why"] = [explain(row, df) for _, row in df.iterrows()]
    forecasts = station_forecasts(killed, df["station"])
    baseline = baseline_3yr(forecasts)
    all_recs, _, _ = recommend(df, baseline)
    all_recs["potential_reduction"] = all_recs["baseline_3yr"] * all_recs["attributable_share"] * all_recs["effect_size"] * (0.5 + 1 + 1) / 3
    all_recs["evidence_tier"] = 4
    ml_bundle, ml_metadata = load_bundle()
    ml_station = pd.DataFrame()
    ml_features = pd.DataFrame()
    ml_crashes = pd.DataFrame()
    ml_global = pd.DataFrame()
    if ml_bundle is not None:
        ml_crashes = load_crashes()
        ml_features, ml_crashes = make_features(ml_crashes, ml_bundle["feature_columns"])
        ml_crashes["ml_fatality_probability"] = ml_bundle["model"].predict_proba(ml_features)[:, 1]
        ml_station = ml_crashes.groupby("station").agg(
            ml_mean_probability=("ml_fatality_probability", "mean"),
            ml_median_probability=("ml_fatality_probability", "median"),
            ml_high_risk_share=("ml_fatality_probability", lambda values: (values >= 0.5).mean()),
        ).reset_index()
        ml_global = global_importance(ml_bundle, ml_features.sample(min(1500, len(ml_features)), random_state=42))
        df = df.merge(ml_station, on="station", how="left")
        lo, hi = df.ml_mean_probability.min(), df.ml_mean_probability.max()
        df["ml_severity_risk"] = 100 * (df.ml_mean_probability - lo) / (hi - lo) if hi > lo else 0.0
        df["composite_decision_risk"] = (0.5 * df.score + 0.5 * df.ml_severity_risk).round(1)
        profile = pd.read_csv(DATA / "ksp_station_profile.csv").set_index("station").to_dict("index")
        relevance = {station: float(df.loc[df.station == station, "ml_mean_probability"].iloc[0]) for station in df.station}
        all_recs["ml_relevance"] = [intervention_relevance(profile.get(row.station), row.id, relevance.get(row.station, 0.2)) for _, row in all_recs.iterrows()]
        all_recs["potential_reduction"] *= all_recs["ml_relevance"]
    breaks, evidence = build_evidence(killed)
    return df, killed, forecasts, baseline, all_recs, breaks, evidence, ml_bundle, ml_metadata, ml_crashes, ml_features, ml_global


st.markdown("<style>section.main > div {max-width: 1500px} .block-container {padding-top: 1.5rem}</style>", unsafe_allow_html=True)
st.title("RoadSafe")
st.caption("Bengaluru intervention intelligence | from WHERE and WHY to WHAT IF and WHAT CAN WE AFFORD")

with st.sidebar:
    st.header("Decision controls")
    weights = {}
    for key, label in [("fatal_share", "Burden"), ("fatal_rate", "Severity"), ("trend", "Trend")]:
        weights[key] = st.slider(label, 0.0, 1.0, DEFAULT_W[key], 0.05)
    if sum(weights.values()) == 0:
        weights = DEFAULT_W
    zones = st.multiselect("Zones", ["East", "West", "North", "South"], ["East", "West", "North", "South"])
    budget = st.select_slider("Safety budget (Cr)", options=[5, 10, 25, 50, 100], value=25)
    st.caption("All effects and costs are modeled potential estimates unless marked otherwise.")

df, killed, forecasts, baseline, recommendations, breaks, evidence, ml_bundle, ml_metadata, ml_crashes, ml_features, ml_global = load_system(tuple(sorted(weights.items())))
view = df[df.zone.isin(zones)]
critical = df[df.band.isin(["Critical", "High"])]
city_fc, _ = city_forecast()
portfolio, portfolio_summary = optimize_portfolio(recommendations[recommendations.station.isin(view.station)], budget)

pages = st.tabs(["Command Center", "Risk Intelligence", "ML & Explainability", "What Changed", "What Works", "Counterfactual Lab", "Budget Optimizer", "Action Plans", "Method & Limits"])

with pages[0]:
    st.subheader("The next safety decision, in one view")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Bengaluru deaths, 2023", "910")
    k2.metric("High / critical stations", len(critical))
    k3.metric("Modeled deaths, 2026-28", f"{city_fc.fit.sum():.0f}")
    k4.metric("Selected budget", f"Rs {budget} Cr")
    k5.metric("Potential portfolio impact", f"{portfolio_summary['potential_reduction']:.1f}")
    st.markdown(f"{badge('OBSERVED')} observed BTP totals &nbsp; {badge('MODELED')} forecast and portfolio impact", unsafe_allow_html=True)
    left, right = st.columns([1.7, 1])
    with left:
        map_df = view.dropna(subset=["lat", "lng"])
        # Coordinate plot is the reliable fallback: it needs no remote basemap tile.
        fig = px.scatter(map_df, x="lng", y="lat", size="killed_recent", color="composite_decision_risk" if "composite_decision_risk" in map_df else "score", hover_name="station", hover_data=["band", "zone", "score", "killed_recent"], color_continuous_scale="YlOrRd", range_color=(0, 100), height=560)
        blackspots = pd.read_csv(DATA / "blackspots.csv").dropna(subset=["lat", "lng"])
        fig.add_trace(go.Scatter(x=blackspots.lng, y=blackspots.lat, mode="markers", name="BTP black spots", marker=dict(size=9, color="black", symbol="x"), text=blackspots.location, hoverinfo="text"))
        if ml_crashes.empty is False:
            fatal_points = ml_crashes[ml_crashes.fatal == 1].dropna(subset=["lat", "lng"]).sample(min(1000, (ml_crashes.fatal == 1).sum()), random_state=42)
            fig.add_trace(go.Scatter(x=fatal_points.lng, y=fatal_points.lat, mode="markers", name="KSP fatal crashes", marker=dict(size=4, color="#8b0000", opacity=.35), text=fatal_points.station, hoverinfo="text"))
        fig.update_layout(xaxis_title="Longitude", yaxis_title="Latitude", margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Priority actions")
        priority = recommendations[recommendations.station.isin(critical.station)].sort_values("potential_reduction", ascending=False).head(5)
        render_table(priority[["station", "intervention", "cost_cr", "potential_reduction", "evidence_tier"]].round(1), hide_index=True, use_container_width=True)
        st.info(f"At Rs {budget} Cr, the exact portfolio selects {len(portfolio)} discrete packages across {portfolio.station.nunique() if not portfolio.empty else 0} locations for {portfolio_summary['potential_reduction']:.1f} potential fatalities prevented over the modeled horizon. This is decision support, not a guarantee.")

with pages[1]:
    st.subheader("Where is risk, and why?")
    selected = st.selectbox("Station", view.station.tolist(), key="risk_station")
    row = df[df.station == selected].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Statistical risk", f"{row.score:.0f}", row.band)
    b.metric("Recent deaths / year", f"{row.killed_recent:.1f}")
    c.metric("Fatality rate", f"{row.fatal_rate:.3f}")
    d.metric("Composite risk", "n/a" if "composite_decision_risk" not in row else f"{row.composite_decision_risk:.0f}")
    st.info(row.why)
    st.markdown(f"{badge('OBSERVED')} 2018-23 deaths and crashes; {badge('ESTIMATED')} 2024-25 deaths inferred from fatal crashes", unsafe_allow_html=True)
    chart, detail = st.columns([1.2, 1])
    with chart:
        history = killed.loc[selected].rename("deaths").reset_index().rename(columns={"index": "year"})
        st.plotly_chart(px.bar(history, x="year", y="deaths", color="deaths", color_continuous_scale="YlOrRd", title="Historical fatality trajectory"), use_container_width=True)
    with detail:
        st.markdown("### Score components")
        render_table(pd.DataFrame({"component": ["Fatal burden", "Severity", "Trend"], "points": [row.contrib_fatal_share, row.contrib_fatal_rate, row.contrib_trend], "weight": [weights["fatal_share"], weights["fatal_rate"], weights["trend"]]}).round(2), hide_index=True, use_container_width=True)
        st.caption("Score is an explainable weighted min-max index, not a probability of a crash.")

with pages[2]:
    st.subheader("ML & Explainability")
    if ml_bundle is None:
        message = (ml_metadata or {}).get("load_error", "ML model has not been trained yet. Run `python train_model.py` to enable this page.")
        st.warning(f"{message} Statistical risk intelligence remains available.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Model", ml_metadata.get("model_type", "saved classifier"))
        m2.metric("Temporal test", f"{ml_metadata['test_period'][0]}-{ml_metadata['test_period'][1]}")
        m3.metric("ROC-AUC / PR-AUC", f"{ml_metadata['metrics']['roc_auc']:.3f} / {ml_metadata['metrics']['pr_auc']:.3f}")
        st.caption("The model predicts crash fatality probability from observed crash characteristics. It describes model associations, not physical causality.")
        st.markdown("### Global model explanation")
        top_global = ml_global.head(15).sort_values("importance")
        st.plotly_chart(px.bar(top_global, x="importance", y="feature", orientation="h", title="Global SHAP feature importance"), use_container_width=True)
        selected_crash_station = st.selectbox("Station for local explanation", view.station.tolist(), key="ml_station")
        station_crash_indices = ml_crashes.index[ml_crashes.station == selected_crash_station].tolist()
        if station_crash_indices:
            idx = station_crash_indices[0]
            local, method = local_explanation(ml_bundle, ml_features, ml_features.index.get_loc(idx) if idx in ml_features.index else 0)
            probability = float(ml_crashes.loc[idx, "ml_fatality_probability"])
            st.metric("Predicted fatality probability for selected crash", f"{probability:.1%}")
            st.plotly_chart(px.bar(local.sort_values("contribution"), x="contribution", y="feature", orientation="h", title=f"Local contributors ({method})"), use_container_width=True)
        station_view = df[df.station == selected_crash_station].iloc[0]
        st.markdown(f"**Station ML severity risk:** {station_view.ml_severity_risk:.1f} &nbsp; **Composite decision risk:** {station_view.composite_decision_risk:.1f}", unsafe_allow_html=True)
        st.info("Interpretation: the model associates the displayed features with higher or lower predicted severity for this observation. SHAP does not prove that any feature causes fatality.")

with pages[2]:
    st.subheader("What changed? Structural breaks are signals, not proof")
    chosen = st.selectbox("Station", view.station.tolist(), key="break_station")
    br = breaks[breaks.station == chosen].iloc[0]
    ev = evidence[evidence.station == chosen].iloc[0].to_dict()
    hist = killed.loc[chosen].rename("deaths").reset_index().rename(columns={"index": "year"})
    st.plotly_chart(px.line(hist, x="year", y="deaths", markers=True, title=f"{chosen}: observed and estimated deaths"), use_container_width=True)
    change = br.get("change_pct", np.nan)
    st.markdown(f"**Signal:** {br.get('classification', 'No strong break')} | **Break year:** {br.get('break_year', 'n/a')} | **Pre/post change:** {change:.1f}%")
    st.write(f"**Evidence tier {ev.get('evidence_tier', 3)}:** {ev.get('evidence_label', 'Structural-break signal only')}. {ev.get('interpretation', '')}")
    if ev.get("controls"):
        render_table(pd.DataFrame([ev])[["controls", "target_post_change", "control_average_change", "differential_change", "pre_trend_compatibility"]].round(2), hide_index=True, use_container_width=True)
    st.warning("No intervention dates are present in the source data. A structural break cannot be attributed to a treatment here.")

with pages[3]:
    st.subheader("What appears to work locally?")
    st.caption("The current evidence base is a literature fallback. Future intervention logs can promote station-specific signals to higher tiers.")
    works = recommendations.groupby(["id", "intervention", "evidence_tier"], as_index=False).agg(potential_reduction=("potential_reduction", "sum"), cost_cr=("cost_cr", "sum"), source=("source", "first"))
    render_table(works.sort_values("potential_reduction", ascending=False).round(2), hide_index=True, use_container_width=True)
    st.markdown(f"{badge('LITERATURE')} effect sizes are discounted assumptions from the intervention knowledge base. They are not guaranteed outcomes.", unsafe_allow_html=True)

with pages[4]:
    st.subheader("Counterfactual Lab")
    chosen = st.selectbox("Station", view.station.tolist(), key="cf_station")
    base = float(baseline[chosen])
    opts = scenarios(chosen, base, recommendations)
    selected_ids = st.multiselect("Compare interventions", opts.id.tolist(), default=opts.id.tolist()[:3])
    shown = opts[opts.id.isin(selected_ids)]
    if not shown.empty:
        st.plotly_chart(px.bar(shown, x="intervention", y="counterfactual_fatalities", color="evidence_tier", title=f"{chosen}: baseline {base:.1f} modeled fatalities vs scenarios"), use_container_width=True)
        render_table(shown[["id", "intervention", "cost_cr", "potential_reduction", "percentage_reduction", "counterfactual_fatalities", "evidence_status", "confidence"]].round(2), hide_index=True, use_container_width=True)
        st.success(f"Best modeled option in this comparison: {shown.sort_values('counterfactual_fatalities').iloc[0].intervention}. Potential impact is modeled and non-causal.")

with pages[5]:
    st.subheader("Budget optimizer")
    if portfolio.empty:
        st.warning("No candidate packages fit this budget.")
    else:
        st.markdown(f"**Exact 0/1 dynamic programming** | spend **Rs {portfolio_summary['cost_cr']:.1f} Cr** | modeled potential impact **{portfolio_summary['potential_reduction']:.1f}**")
        render_table(portfolio[["station", "intervention", "cost_cr", "potential_reduction", "evidence_tier", "source"]].round(2), hide_index=True, use_container_width=True)
    frontier = efficient_frontier(recommendations[recommendations.station.isin(view.station)], [5, 10, 25, 50, 100])
    st.plotly_chart(px.line(frontier, x="cost_cr", y="potential_reduction", markers=True, title="Efficient frontier: spend vs modeled potential impact"), use_container_width=True)

with pages[6]:
    st.subheader("Action plans")
    chosen = st.selectbox("Station", view.station.tolist(), key="policy_station")
    row = df[df.station == chosen].iloc[0]
    option = scenarios(chosen, float(baseline[chosen]), recommendations).iloc[0].to_dict()
    ev = evidence[evidence.station == chosen].iloc[0].to_dict()
    st.code(station_brief(row, option, ev), language="text")
    st.markdown("### Portfolio actions")
    render_table(pd.DataFrame(portfolio_actions(portfolio)), hide_index=True, use_container_width=True)
    st.caption("Owner assignments are operational suggestions. Add implementation dates and outcome measures before claiming causal results.")

with pages[7]:
    st.subheader("Method, provenance, and limits")
    st.markdown("""
**Observed:** BTP station-wise crash counts (2018-2023), BTP city totals, KSP crash records, and BTP black spots.

**Estimated:** 2024-2025 killed values use fatal crashes multiplied by the observed 2021-2023 city killed/fatal ratio. They are flagged in the processed panel.

**Modeled:** OLS baseline forecasts, intervention effect arithmetic, and budget portfolios. Combined interventions use the conservative formula $1 - \\prod_i(1-e_i)$; the UI never calls these guaranteed lives saved.

**Quasi-experimental signal:** simple mean-shift detection plus nearest pre-period level/trend controls and a compatibility check. No treatment date exists, so the strongest available label is a structural-break or quasi-experimental signal, never proof of causality.

**Limits:** no station exposure denominator, administrative station boundaries, approximate centroids, incomplete KSP geocoding, no validated intervention log, and uncertain future outcomes. City-wide time patterns are not station-specific.

The learning-loop architecture is: **identify -> diagnose -> intervene -> measure -> evaluate -> learn**. Full closed-loop causal learning begins when implementation dates, treated locations, comparison locations, and post-intervention outcomes are collected.
""")
    st.download_button("Download validation summary", (DATA / "processed" / "validation_summary.txt").read_text(encoding="utf-8"), "validation_summary.txt")
