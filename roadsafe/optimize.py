"""Exact 0/1 portfolio selection for station-intervention packages."""
import pandas as pd


def optimize_portfolio(candidates, budget_cr):
    """Select a non-overlapping package portfolio with dynamic programming.

    Costs are in crore and discretized to 0.1 crore for a small, transparent
    exact search. Benefit is modeled potential impact, not observed lives saved.
    """
    items = candidates[candidates.cost_cr > 0].copy().reset_index(drop=True)
    scale = 10
    capacity = int(round(float(budget_cr) * scale))
    best = [(0.0, []) for _ in range(capacity + 1)]
    for idx, row in items.iterrows():
        cost = min(int(round(row.cost_cr * scale)), capacity + 1)
        if cost > capacity:
            continue
        for spent in range(capacity, cost - 1, -1):
            prior_benefit, prior = best[spent - cost]
            benefit = prior_benefit + float(row.potential_reduction)
            if benefit > best[spent][0]:
                best[spent] = (benefit, prior + [idx])
    _, selected = max(best, key=lambda value: value[0])
    portfolio = items.loc[selected].copy() if selected else items.iloc[0:0].copy()
    return portfolio, {"budget_cr": float(budget_cr), "cost_cr": float(portfolio.cost_cr.sum()),
                       "potential_reduction": float(portfolio.potential_reduction.sum()),
                       "method": "Exact 0/1 dynamic programming; modeled potential benefit"}


def efficient_frontier(candidates, budgets):
    rows = []
    for budget in budgets:
        _, summary = optimize_portfolio(candidates, budget)
        rows.append(summary)
    return pd.DataFrame(rows)