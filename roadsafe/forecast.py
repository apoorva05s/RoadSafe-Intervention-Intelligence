"""Linear-trend projection of deaths per station (2026-2028) and for the city.
OLS on annual points; 2020 and 2021 are down-weighted (COVID) by excluding them
from the fit when >=5 other points exist. Band = +-1.96 * residual std.
Fallback when <4 usable points: mean of available years carried forward, band = +-30%.
"""
import pandas as pd, numpy as np, pathlib

DATA = pathlib.Path(__file__).parent / "data"
HORIZON = [2026, 2027, 2028]
COVID = {2020, 2021}


def fit_project(series: pd.Series, horizon=HORIZON):
    s = series.dropna()
    use = s[[y not in COVID for y in s.index]] if (len(s) - len([y for y in s.index if y in COVID])) >= 5 else s
    if len(use) < 4:
        m = s.mean() if len(s) else np.nan
        return pd.DataFrame({"year": horizon, "fit": m, "lo": m * 0.7, "hi": m * 1.3, "method": "mean carried forward"})
    x = np.array(use.index, dtype=float); y = use.values.astype(float)
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    sd = resid.std(ddof=2) if len(y) > 2 else 0
    hx = np.array(horizon, dtype=float)
    fit = np.maximum(a + b * hx, 0)
    return pd.DataFrame({"year": horizon, "fit": fit, "lo": np.maximum(fit - 1.96 * sd, 0),
                         "hi": fit + 1.96 * sd, "method": "OLS linear trend"})


def station_forecasts(killed: pd.DataFrame, stations: pd.Series):
    out = []
    for st in stations:
        f = fit_project(killed.loc[st]); f["station"] = st; out.append(f)
    return pd.concat(out, ignore_index=True)


def city_forecast():
    c = pd.read_csv(DATA / "city_2007_2025.csv").set_index("year")["killed"]
    return fit_project(c[c.index >= 2012]), c   # post-2012 regime (Bengaluru deaths fell then rose)


def baseline_3yr(fc: pd.DataFrame):
    """Sum of projected deaths over the horizon per station."""
    return fc.groupby("station")["fit"].sum().rename("baseline_3yr")


if __name__ == "__main__":
    from risk import compute
    df, killed, total = compute()
    fc = station_forecasts(killed, df["station"])
    b = baseline_3yr(fc)
    print(b.sort_values(ascending=False).head(10).round(1))
    cf, hist = city_forecast()
    print(cf.round(0)); print("city baseline 2026-28:", round(cf["fit"].sum()))


def project(series: pd.Series, horizon_years=3, start_year=2018, exclude_covid=True, annual_change_pct=None, last_year=2025):
    """Parametrised projection used by the Prediction tab.
    - start_year: first year included in the fit window
    - exclude_covid: drop 2020-21 from the fit (if >=4 other points remain)
    - annual_change_pct: if given, overrides the fitted slope with a scenario: last observed value × (1+p)^k
    Returns (frame with year, fit, lo, hi, method, slope_per_year, base_value)."""
    s = series.dropna(); s = s[s.index >= start_year]
    horizon = list(range(last_year + 1, last_year + 1 + horizon_years))
    if annual_change_pct is not None:
        base = s[s.index >= last_year - 2].mean()  # 3-yr mean as the launch point
        fit = np.array([base * (1 + annual_change_pct / 100) ** (k + 1) for k in range(horizon_years)])
        return pd.DataFrame({"year": horizon, "fit": fit, "lo": fit * 0.85, "hi": fit * 1.15,
                             "method": f"scenario {annual_change_pct:+.0f}%/yr from 3-yr mean {base:.0f}"}), None, base
    use = s[[y not in COVID for y in s.index]] if (exclude_covid and (len(s) - sum(y in COVID for y in s.index)) >= 4) else s
    if len(use) < 3:
        m = s.mean()
        return pd.DataFrame({"year": horizon, "fit": m, "lo": m * .7, "hi": m * 1.3, "method": "mean carried forward (<3 pts)"}), 0.0, m
    x = np.array(use.index, float); y = use.values.astype(float)
    b, a = np.polyfit(x, y, 1); resid = y - (a + b * x); sd = resid.std(ddof=2) if len(y) > 2 else 0
    hx = np.array(horizon, float); fit = np.maximum(a + b * hx, 0)
    return pd.DataFrame({"year": horizon, "fit": fit, "lo": np.maximum(fit - 1.96 * sd, 0), "hi": fit + 1.96 * sd,
                         "method": f"OLS on {len(use)} pts {int(x.min())}–{int(x.max())}" + (" excl. 2020–21" if exclude_covid and len(use) < len(s) else "")}), b, a + b * x[-1]
