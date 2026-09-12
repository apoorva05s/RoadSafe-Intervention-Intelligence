"""Interpretable structural-break and matched-control evidence signals.

This module deliberately does not infer intervention effects: the repository
contains no validated intervention log. Results are signals for review.
"""
import numpy as np
import pandas as pd


def _slope(series):
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    return float(np.polyfit(series.index.astype(float), series.values.astype(float), 1)[0])


def structural_breaks(killed, min_pre=3, min_post=3):
    """Return the best mean-shift candidate per station, if one is notable."""
    rows = []
    for station, values in killed.iterrows():
        values = values.sort_index().dropna()
        candidates = []
        for break_year in values.index:
            pre = values[values.index < break_year]
            post = values[values.index >= break_year]
            if len(pre) < min_pre or len(post) < min_post:
                continue
            pre_mean, post_mean = pre.mean(), post.mean()
            change_pct = 100 * (post_mean - pre_mean) / pre_mean if pre_mean else np.nan
            candidates.append((abs(change_pct), break_year, pre, post, change_pct))
        if not candidates:
            rows.append({"station": station, "break_year": np.nan, "classification": "No strong break",
                         "confidence": "Low"})
            continue
        _, year, pre, post, change_pct = max(candidates, key=lambda item: item[0])
        magnitude = abs(change_pct)
        classification = "Strong structural break" if magnitude >= 35 else "Possible break" if magnitude >= 20 else "No strong break"
        confidence = "Moderate" if magnitude >= 35 else "Low"
        rows.append({"station": station, "break_year": int(year), "pre_break_mean": pre.mean(),
                     "post_break_mean": post.mean(), "pre_break_slope": _slope(pre),
                     "post_break_slope": _slope(post), "change_pct": change_pct,
                     "classification": classification, "confidence": confidence})
    return pd.DataFrame(rows)


def matched_signal(station, killed, breaks, n_controls=3):
    """Compare post-period change with similar stations using pre-break data only."""
    record = breaks.loc[breaks.station == station].iloc[0]
    if pd.isna(record.get("break_year")) or record["classification"] == "No strong break":
        return {"station": station, "evidence_tier": 3, "evidence_label": "Structural-break signal only",
                "pre_trend_compatibility": "Unavailable", "interpretation": "No strong break identified."}
    year = int(record["break_year"])
    target = killed.loc[station].dropna()
    pre_years = [y for y in target.index if y < year]
    post_years = [y for y in target.index if y >= year]
    if len(pre_years) < 3 or not post_years:
        return {"station": station, "evidence_tier": 3, "evidence_label": "Structural-break signal only",
                "pre_trend_compatibility": "Unavailable", "interpretation": "Insufficient history for comparison."}
    target_pre = target.loc[pre_years].mean()
    target_post = target.loc[post_years].mean()
    candidates = []
    for other in killed.index:
        if other == station:
            continue
        series = killed.loc[other]
        common_pre = series.reindex(pre_years).dropna()
        if len(common_pre) < 3:
            continue
        distance = abs(common_pre.mean() - target_pre) + abs(_slope(common_pre) - record["pre_break_slope"])
        candidates.append((distance, other, common_pre))
    controls = sorted(candidates)[:n_controls]
    if not controls:
        return {"station": station, "evidence_tier": 3, "evidence_label": "Structural-break signal only",
                "pre_trend_compatibility": "Unavailable", "interpretation": "No comparable controls found."}
    control_changes = []
    pre_slopes = []
    for _, name, common_pre in controls:
        series = killed.loc[name]
        common_post = series.reindex(post_years).dropna()
        if len(common_post):
            control_changes.append(common_post.mean() - common_pre.mean())
            pre_slopes.append(_slope(common_pre))
    if not control_changes:
        return {"station": station, "evidence_tier": 3, "evidence_label": "Structural-break signal only",
                "pre_trend_compatibility": "Unavailable", "interpretation": "Controls lack post-period data."}
    control_change = float(np.mean(control_changes))
    differential = float((target_post - target_pre) - control_change)
    slope_gap = abs(np.mean(pre_slopes) - record["pre_break_slope"])
    compatibility = "Good" if slope_gap <= 0.5 else "Moderate" if slope_gap <= 1.5 else "Poor"
    tier = 2 if compatibility in {"Good", "Moderate"} else 3
    return {"station": station, "break_year": year, "controls": ", ".join(name for _, name, _ in controls),
            "target_post_change": target_post - target_pre, "control_average_change": control_change,
            "differential_change": differential, "pre_trend_compatibility": compatibility,
            "evidence_tier": tier, "evidence_label": "Local quasi-experimental signal" if tier == 2 else "Structural-break signal only",
            "interpretation": "Change in trajectory, not proof of intervention causality."}


def build_evidence(killed):
    breaks = structural_breaks(killed)
    evidence = pd.DataFrame([matched_signal(station, killed, breaks) for station in killed.index])
    return breaks, evidence