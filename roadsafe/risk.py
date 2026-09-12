"""Risk score 0-100 per BTP traffic station. Pure statistics, no ML.

Components (each min-max scaled to 0..1 across stations):
  fatal_share : station's share of city deaths, 2023-2025 mean   (burden)
  fatal_rate  : deaths per crash, 2023-2025 mean                 (severity)
  trend       : OLS slope of deaths 2018-2025 as % of station mean per year, clipped +-25%
Score = 100 * sum(w_c * z_c). Bands: >=75 Critical, 50-75 High, 25-50 Medium, <25 Low.
"""
import pandas as pd, numpy as np, pathlib

DATA = pathlib.Path(__file__).parent / "data"
YEARS = list(range(2018, 2026))
RECENT = [2023, 2024, 2025]
DEFAULT_W = {"fatal_share": 0.40, "fatal_rate": 0.30, "trend": 0.30}


def ols_slope_pct(y: pd.Series):
    """Slope per year as % of mean, on non-null points; needs >=4 points."""
    y = y.dropna()
    if len(y) < 4 or y.mean() == 0:
        return np.nan, len(y)
    x = np.array([int(i) for i in y.index], dtype=float)
    slope = np.polyfit(x, y.values.astype(float), 1)[0]
    return 100 * slope / y.mean(), len(y)


def minmax(s: pd.Series):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0


def band(score):
    return pd.cut(score, [-1, 25, 50, 75, 101], labels=["Low", "Medium", "High", "Critical"])


def compute(weights=None):
    w = dict(DEFAULT_W if weights is None else weights)
    tot = sum(w.values()); w = {k: v / tot for k, v in w.items()}

    wide = pd.read_csv(DATA / "stations_wide.csv")
    ll = pd.read_csv(DATA / "stations_latlng.csv")
    df = wide.merge(ll, on="station", how="left")

    killed = df[[f"killed_{y}" for y in YEARS]].copy(); killed.columns = YEARS; killed.index = df["station"]
    total = df[[f"total_{y}" for y in YEARS]].copy(); total.columns = YEARS; total.index = df["station"]

    df["killed_recent"] = killed[RECENT].mean(axis=1).values
    df["crashes_recent"] = total[RECENT].mean(axis=1).values
    df["fatal_share"] = df["killed_recent"] / df["killed_recent"].sum()
    df["fatal_rate"] = df["killed_recent"] / df["crashes_recent"]
    tr = killed.apply(lambda r: ols_slope_pct(r), axis=1, result_type="expand")
    df["trend_pct"], df["n_years"] = tr[0].values, tr[1].values
    df["trend_clipped"] = df["trend_pct"].clip(-25, 25)

    # stations with too few years: trend set to city median so they are not penalised/rewarded
    df["trend_filled"] = df["trend_clipped"].fillna(df["trend_clipped"].median())
    df["data_flag"] = np.where(df["n_years"] < 4, "new station (<4 yrs): trend = city median", "")

    df["z_fatal_share"] = minmax(df["fatal_share"])
    df["z_fatal_rate"] = minmax(df["fatal_rate"])
    df["z_trend"] = minmax(df["trend_filled"])
    for c in ("fatal_share", "fatal_rate", "trend"):
        df[f"contrib_{c}"] = 100 * w[c] * df[f"z_{c}"]
    df["score"] = df[[f"contrib_{c}" for c in w]].sum(axis=1).round(1)
    df["band"] = band(df["score"])
    df["rank"] = df["score"].rank(ascending=False, method="min").astype(int)

    # city reference values for the "why" sentences
    df.attrs["city_fatal_rate"] = df["killed_recent"].sum() / df["crashes_recent"].sum()
    df.attrs["weights"] = w
    return df.sort_values("score", ascending=False).reset_index(drop=True), killed, total


if __name__ == "__main__":
    df, killed, total = compute()
    pd.set_option("display.width", 200)
    print(df[["rank", "station", "zone", "killed_recent", "fatal_share", "fatal_rate", "trend_pct",
              "score", "band", "data_flag"]].head(15).round(3))
    print("city fatal rate:", round(df.attrs["city_fatal_rate"], 3))
    print(df["band"].value_counts())


def explain(row, df):
    """Plain-English reason for one station's score, with the reference values used."""
    n = len(df)
    rk = lambda col: int((df[col] > row[col]).sum() + 1)
    parts = []
    parts.append(f"{row['killed_recent']:.0f} deaths/yr (2023–25) = {100*row['fatal_share']:.1f}% of the city's toll, "
                 f"rank {rk('fatal_share')}/{n} → {row['contrib_fatal_share']:.0f} pts")
    parts.append(f"1 fatal crash in every {1/row['fatal_rate']:.0f} (city 1 in {1/df.attrs['city_fatal_rate']:.0f}), "
                 f"rank {rk('fatal_rate')}/{n} → {row['contrib_fatal_rate']:.0f} pts")
    tr = "n/a (new station, city median used)" if pd.isna(row['trend_pct']) else f"{row['trend_pct']:+.1f}%/yr since 2018"
    parts.append(f"trend {tr}, rank {rk('trend_filled')}/{n} → {row['contrib_trend']:.0f} pts")
    top = max([("burden", row['contrib_fatal_share']), ("severity", row['contrib_fatal_rate']), ("trend", row['contrib_trend'])], key=lambda x: x[1])[0]
    return f"Score {row['score']:.0f} ({row['band']}), driven mainly by {top}: " + "; ".join(parts) + "."


def reference_table(df):
    """Min / max / city value used for each component's min-max scaling."""
    return pd.DataFrame({
        "component": ["Share of city deaths", "Deaths per crash", "Trend %/yr (clipped ±25)"],
        "min (→ 0 pts)": [df["fatal_share"].min(), df["fatal_rate"].min(), df["trend_filled"].min()],
        "max (→ full weight)": [df["fatal_share"].max(), df["fatal_rate"].max(), df["trend_filled"].max()],
        "station at max": [df.loc[df["fatal_share"].idxmax(), "station"], df.loc[df["fatal_rate"].idxmax(), "station"], df.loc[df["trend_filled"].idxmax(), "station"]],
        "weight": [df.attrs["weights"]["fatal_share"], df.attrs["weights"]["fatal_rate"], df.attrs["weights"]["trend"]],
    })
