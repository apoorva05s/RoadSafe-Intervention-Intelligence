"""Intervention recommendation + impact estimate per station. Pure arithmetic.

lives_saved = baseline_3yr * attributable_share * effect_size * ramp
ramp = (0.5 + 1 + 1) / 3  -> year-1 partial rollout
Top-3 per station chosen by trigger strength (station z-score / share behind the rule).
Package total = sum(top-3) * OVERLAP (interventions share victims).
"""
import pandas as pd, numpy as np, pathlib

DATA = pathlib.Path(__file__).parent / "data"
RAMP = (0.5 + 1 + 1) / 3
OVERLAP = 0.85

prof = pd.read_csv(DATA / "city_profile.csv")
P = {(g, c): s for g, c, s in prof[["group", "category", "share"]].itertuples(index=False)}
RULES = pd.read_csv(DATA / "interventions.csv")
BS = pd.read_csv(DATA / "blackspots.csv")

# city-wide attributable shares (BTP 2023 / MoRTH 2024), applied proportionally per station
SHARES = {
    "share_overspeeding": 0.60,   # MoRTH attributes 87% to overspeeding; discounted to 60% as the enforceable part
    "share_pedestrian": P[("road_user", "Pedestrians")],
    "share_motorcyclist_nohelmet": P[("road_user", "Motorcyclists")] * (1 - P[("behaviour", "Correct helmet use")]),
    "share_night": P[("time_band", "18:00-22:00")] + P[("time_band", "22:00-02:00")],
    "share_blackspot": None,      # station-specific: black-spot deaths / station deaths 2021-23
    "share_hmv_impact": 0.21,     # lorries in 19% of motorcyclist and 24% of pedestrian deaths (BTP p.13)
    "share_hitrun": P[("crash_type", "Hit-and-run")],
    "share_dui": P[("violation", "Drunk driving")],
}


try:  # station-specific shares from KSP FIR crash records 2016-2023 (Bengaluru City)
    KSP = pd.read_csv(DATA / "ksp_station_profile.csv").set_index("station")
except FileNotFoundError:
    KSP = None


def ksp(row, col, default):
    if KSP is not None and row["station"] in KSP.index and pd.notna(KSP.loc[row["station"], col]):
        return float(KSP.loc[row["station"], col])
    return default


def triggers(row, df):
    """Trigger strength per rule for one station row (0..1-ish, comparable).
    Uses station-specific KSP shares where available; city BTP pattern otherwise."""
    bs_deaths = BS.loc[BS["station"] == row["station"], "fatalities_2021_23"].sum()
    killed_21_23 = np.nansum([row.get(f"killed_{y}", np.nan) for y in (2021, 2022, 2023)])
    bs_share = min(bs_deaths / killed_21_23, 0.8) if killed_21_23 > 0 else 0
    ped = ksp(row, "fatal_ped_share", 0.31)          # share of fatal crashes that hit a pedestrian
    hitrun = ksp(row, "fatal_hitrun_share", 0.30)
    highway = ksp(row, "fatal_highway_share", 0.25)  # NH / SH / arterial share of fatal crashes
    return {
        "fatal_rate": row["z_fatal_rate"] * (1 + highway),             # lethal AND on fast roads -> speed
        "ped_share": ped * 1.6,                                        # scaled so a 31% city-average ped share ~ 0.5
        "mc_share": 0.59 * (1 - ped) * 1.4,                            # motorcyclists dominate what is not pedestrian
        "night_share": 0.43,                                           # no station-level hour data: uniform
        "blackspot_deaths": bs_share * 1.5,
        "hmv_share": 0.21 * (1 + (row["fatal_rate"] > df.attrs["city_fatal_rate"])) * (1 + highway),
        "hitrun_share": hitrun * 1.5,
        "dui_share": 0.05 * (1 + 2 * (row["station"] in {"Adugodi", "Madivala", "HSR Layout", "Cubbon Park", "Ashokanagar", "Halasooru"})),
    }, bs_share


def recommend(df: pd.DataFrame, baseline: pd.Series, top_n=3):
    rows = []
    for _, r in df.iterrows():
        trig, bs_share = triggers(r, df)
        B = baseline.get(r["station"], np.nan)
        ped = ksp(r, "fatal_ped_share", SHARES["share_pedestrian"])
        hitrun = ksp(r, "fatal_hitrun_share", SHARES["share_hitrun"])
        for _, rule in RULES.iterrows():
            base = rule["attributable_base"]
            if base == "share_blackspot":
                share = bs_share
            elif base == "share_pedestrian":
                share = ped
            elif base == "share_hitrun":
                share = hitrun
            elif base == "share_motorcyclist_nohelmet":
                share = 0.59 * (1 - ped) / (1 - 0.31) * (1 - P[("behaviour", "Correct helmet use")])
            else:
                share = SHARES[base]
            saved = B * share * rule["effect_size"] * RAMP
            rows.append({
                "station": r["station"], "id": rule["id"], "intervention": rule["intervention"],
                "trigger": rule["trigger"], "trigger_strength": trig[rule["trigger"]],
                "attributable_share": share, "effect_size": rule["effect_size"],
                "baseline_3yr": B, "lives_saved_3yr": saved,
                "cost_cr": rule["cost_cr_per_station"],
                "cr_per_life": rule["cost_cr_per_station"] / saved if saved > 0 else np.nan,
                "source": rule["source"],
            })
    allr = pd.DataFrame(rows)
    allr["rank_trigger"] = allr.groupby("station")["trigger_strength"].rank(ascending=False, method="first")
    top = allr[allr["rank_trigger"] <= top_n].copy()
    top["rank_lives"] = top.groupby("station")["lives_saved_3yr"].rank(ascending=False, method="first").astype(int)
    pkg = top.groupby("station").agg(package_lives_saved=("lives_saved_3yr", "sum"),
                                    package_cost_cr=("cost_cr", "sum"))
    pkg["package_lives_saved"] *= OVERLAP
    return allr, top, pkg


if __name__ == "__main__":
    from risk import compute
    from forecast import station_forecasts, baseline_3yr
    df, killed, total = compute()
    fc = station_forecasts(killed, df["station"]); B = baseline_3yr(fc)
    allr, top, pkg = recommend(df, B)
    pd.set_option("display.width", 220)
    for s in ["K R Puram", "Yalahanka", "Cubbon Park"]:
        print(f"\n== {s}"); print(top[top.station == s][["id", "trigger_strength", "attributable_share", "lives_saved_3yr", "cr_per_life"]].round(2))
    crit = df[df["band"].isin(["Critical", "High"])]["station"]
    print("\nPackage over Critical+High stations:", round(pkg.loc[pkg.index.isin(crit), "package_lives_saved"].sum()),
          "lives 2026-28; cost", pkg.loc[pkg.index.isin(crit), "package_cost_cr"].sum(), "cr")
    print("best single intervention citywide:", allr.groupby("id")["lives_saved_3yr"].sum().sort_values(ascending=False).round(0).to_dict())
