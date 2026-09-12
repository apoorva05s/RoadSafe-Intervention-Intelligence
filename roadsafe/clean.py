"""Build the core per-station table from BTP station-wise CSVs (2018-2025).

Canonical station names = the 2023 BTP list. Older names are mapped via ALIASES.
Stations created after 2022 simply have NaN for earlier years.
Killed/Injured are not published for 2024-2025 -> estimated from fatal crashes
using the city-wide killed/fatal ratio (flagged in column `killed_est_years`).
"""
import pandas as pd, numpy as np, re, pathlib

RAW = pathlib.Path(__file__).parent / "data" / "raw"
OUT = pathlib.Path(__file__).parent / "data"

ALIASES = {
    "Ulsoor": "Halasooru", "F.Town": "Pulikeshinagar", "Y.Pura": "Yashawanthapura",
    "B.Pura": "Bytarayanapura", "K.R.Pura": "K R Puram", "K.Swamy Lyt.": "K S Layout",
    "Chikjala": "Chikkajala", "HuliMavu": "Hulimavu", "Malleswaram": "Malleshwaram",
    "Mico layout": "Micolayout", "RT Nagar": "R T Nagar", "White Field": "Whitefield",
    "Yelahanka": "Yalahanka", "Chamarajapet": "Chamarajpet",
    "Sheshadripuram/Chickpet": "Chickpet", "Devanahalli": "Int. Aiport",
}
DROP = re.compile(r"total", re.I)


def load(f):
    d = pd.read_csv(RAW / f, encoding="utf-8-sig")
    d.columns = [re.sub(r"\s+", " ", c).strip() for c in d.columns]
    d["Station"] = d["Station"].astype(str).str.strip().replace(ALIASES)
    d = d[~d["Station"].str.contains(DROP)]
    return d


def melt(d, year, cols):
    out = d[["Station"]].copy()
    out["year"] = year
    for k, c in cols.items():
        out[k] = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce").values
    return out


rows = []
s1 = load("s1_2018_2020.csv")
for y in (2018, 2019, 2020):
    rows.append(melt(s1, y, {"fatal": f"{y} - Fatal", "killed": f"{y} - Killed",
                             "injured": f"{y} - Injured", "total": f"{y} - Total Cases"}))
s2 = load("s2_2021_2022.csv")
for y in (2021, 2022):
    rows.append(melt(s2, y, {"fatal": f"{y} - Fatal", "killed": f"{y} - Killed",
                             "injured": f"{y} - Injured", "total": f"{y} - Total Cases"}))
s3 = load("s3_2023.csv")
rows.append(melt(s3, 2023, {"fatal": "2023 - Fatal Cases", "killed": "2023 - Killed People",
                            "injured": "2023 - Injured People", "total": "2023 - Total Cases"}))
s4 = load("s4_2024.csv")
rows.append(melt(s4, 2024, {"fatal": "2024-Fatal crashes", "total": "2024-Total crashes"}))
s5 = load("s5_2025.csv")
rows.append(melt(s5, 2025, {"fatal": "2025-Fatal crashes", "total": "2025-Total crashes"}))

long = pd.concat(rows, ignore_index=True)
# duplicates within a year (e.g. two rows mapped to Chickpet) -> sum
long = long.groupby(["Station", "year"], as_index=False).sum(min_count=1)

# --- estimate killed for 2024/2025 from fatal crashes using city ratio 2021-2023
city = pd.read_csv(RAW / "s6_city_2007_2025.csv", thousands=",")
city.columns = ["year", "fatal", "killed", "nonfatal", "total"]
ratio = (city.query("2021 <= year <= 2023")["killed"].sum()
         / city.query("2021 <= year <= 2023")["fatal"].sum())
est = long["killed"].isna() & long["fatal"].notna()
long.loc[est, "killed"] = (long.loc[est, "fatal"] * ratio).round(1)
long["killed_est"] = est

# --- zone / sub-division from the 2023 file (canonical)
meta = s3[["Zone", "Sub-division", "Station"]].drop_duplicates("Station")
meta = meta.rename(columns={"Zone": "zone", "Sub-division": "subdivision"})
# stations new in 2024/25 pick up meta from s4/s5
for s in (s4, s5):
    m = s[["Zone", "Sub-division", "Station"]].rename(columns={"Zone": "zone", "Sub-division": "subdivision"})
    meta = pd.concat([meta, m]).drop_duplicates("Station")

wide = long.pivot(index="Station", columns="year", values=["total", "fatal", "killed", "injured"])
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide.reset_index().merge(meta, on="Station", how="left").rename(columns={"Station": "station"})
wide["killed_est_years"] = "2024,2025"

OUT.mkdir(exist_ok=True)
long.to_csv(OUT / "stations_long.csv", index=False)
wide.to_csv(OUT / "stations_wide.csv", index=False)
city.to_csv(OUT / "city_2007_2025.csv", index=False)

# Publish a small, auditable processed layer for the application and downstream users.
PROCESSED = OUT / "processed"
PROCESSED.mkdir(exist_ok=True)
station_master = wide.rename(columns={"station": "station"}).copy()
station_master["lat"] = station_master["station"].map(ll_lookup := pd.read_csv(OUT / "stations_latlng.csv").set_index("station")["lat"])
station_master["lng"] = station_master["station"].map(pd.read_csv(OUT / "stations_latlng.csv").set_index("station")["lng"])
station_master.to_csv(PROCESSED / "station_master.csv", index=False)
long.rename(columns={"Station": "station", "total": "total_crashes", "fatal": "fatal_crashes", "killed": "killed"}).to_csv(PROCESSED / "station_year_panel.csv", index=False)
city.rename(columns={"year": "year", "fatal": "fatal_crashes", "killed": "killed", "total": "total_crashes"}).to_csv(PROCESSED / "city_master.csv", index=False)
pd.read_csv(OUT / "interventions.csv").assign(evidence_tier=4, evidence_status="Literature-derived estimate").to_csv(PROCESSED / "intervention_candidates.csv", index=False)
crash_ml = pd.read_csv(RAW / ".." / "ksp_points.csv") if (RAW / ".." / "ksp_points.csv").exists() else pd.read_csv(OUT / "ksp_points.csv")
crash_ml["fatal"] = crash_ml["severity"].astype(str).str.strip().str.casefold().eq("fatal").astype(int)
crash_ml["crash_id"] = range(len(crash_ml))
crash_ml.to_csv(PROCESSED / "crash_ml_dataset.csv", index=False)
pd.DataFrame({"field": ["station", "zone", "subdivision", "year", "total_crashes", "fatal_crashes", "killed", "killed_is_estimated", "lat", "lng"],
              "definition": ["BTP traffic station", "BTP zone", "BTP subdivision", "Calendar year", "Reported total crash cases", "Reported fatal crash cases", "People killed; 2024-25 estimated", "True when inferred from fatal crashes", "Station centroid latitude", "Station centroid longitude"],
              "status": ["Observed", "Observed", "Observed", "Observed", "Observed", "Observed", "Observed/Estimated", "Derived", "Approximate", "Approximate"]}).to_csv(PROCESSED / "data_dictionary.csv", index=False)
with open(PROCESSED / "validation_summary.txt", "w", encoding="utf-8") as fh:
    fh.write(f"station-year rows: {len(long)}\n")
    fh.write(f"duplicate station-year rows after aggregation: {long.duplicated(['Station', 'year']).sum()}\n")
    fh.write(f"estimated killed rows: {int(long['killed_est'].sum())}\n")
    fh.write(f"crash ML rows: {len(crash_ml)}; invalid coordinates: {int((~crash_ml.lat.between(-90, 90) | ~crash_ml.lng.between(-180, 180)).sum())}\n")
    fh.write("No intervention dates or exposure denominators are present in source data.\n")

if __name__ == "__main__":
    print("killed/fatal ratio 2021-23:", round(ratio, 3))
    print("stations:", len(wide), "| years per station:",
          long.groupby("Station")["year"].count().value_counts().sort_index().to_dict())
    print("missing meta:", wide[wide["zone"].isna()]["station"].tolist())
    chk = long.groupby("year")[["total", "fatal", "killed"]].sum().round(0)
    chk = chk.join(city.set_index("year")[["total", "fatal", "killed"]], rsuffix="_city")
    print(chk)
