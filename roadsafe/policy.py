"""Deterministic policy brief generation from computed, labeled values."""


def station_brief(row, scenario, evidence):
    intervention = scenario.get("intervention", "targeted safety action")
    tier = evidence.get("evidence_tier", 4)
    label = evidence.get("evidence_label", "Literature fallback")
    return f"""ROAD SAFETY ACTION BRIEF

Location: {row['station']}
Priority: {row['band'].upper()}

Situation
{row['why']}

Recommended action
{intervention}

Indicative budget
₹{float(scenario.get('cost_cr', 0)):.1f} Cr

Potential impact
{float(scenario.get('potential_reduction', 0)):.1f} fatalities potentially prevented over the modeled three-year horizon. This is a modeled estimate, not a guarantee.

Evidence
Tier {tier}: {label}. {evidence.get('interpretation', 'Local intervention outcome data is unavailable.')}

Implementation priority
Prioritize design and baseline measurement, then review quarterly.

Monitoring
Track fatal crashes, severe crashes, relevant violations, implementation status, and exposure measures when available.
"""


def portfolio_actions(portfolio):
    actions = []
    for _, item in portfolio.iterrows():
        actions.append({"phase": "Immediate (0–3 months)", "location": item.station,
                        "action": item.intervention, "owner": "BTP / BBMP", "cost_cr": item.cost_cr,
                        "metric": "Quarterly fatal and severe crashes", "evidence": f"Tier {item.evidence_tier}"})
    return actions