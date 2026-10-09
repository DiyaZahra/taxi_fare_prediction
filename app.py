import os
import numpy as np, pandas as pd, joblib
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
AIRPORTS = [1, 132, 138]
SERVICES = ["Yellow", "Green", "Uber", "Lyft"]
BASE = ["est_distance", "avg_demand", "avg_speed", "is_weekend", "is_rush_hour",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "temperature_2m", "precipitation", "snowfall", "wind_speed_10m", "cloud_cover",
        "is_raining", "is_snowing", "is_airport_pu", "is_airport_do"]


def load_all():
    art = joblib.load(os.path.join(HERE, "model.joblib"))
    zones = pd.read_csv(os.path.join(HERE, "zones.csv"))
    pair = pd.read_parquet(os.path.join(HERE, "pair_distance.parquet"))
    dem = pd.read_parquet(os.path.join(HERE, "demand_avg.parquet"))
    spd = pd.read_parquet(os.path.join(HERE, "speed_avg.parquet"))
    wm = pd.read_csv(os.path.join(HERE, "weather_monthly.csv")).set_index("month")

    bmap = zones.set_index("LocationID")["Borough"].to_dict()
    pb = pair.assign(pu_b=pair["PULocationID"].map(bmap), do_b=pair["DOLocationID"].map(bmap))
    return {
        "model": art["model"], "scaler": art["scaler"], "columns": art["columns"],
        "zones": zones, "wm": wm, "bmap": bmap,
        "pair_map": pair.set_index(["PULocationID", "DOLocationID"])["est_distance"].to_dict(),
        "bor_map": pb.groupby(["pu_b", "do_b"])["est_distance"].median().to_dict(),
        "dist_global": float(pair["est_distance"].median()),
        "dem_map": dem.set_index(["PULocationID", "day_of_week", "hour_of_day"])["avg_demand"].to_dict(),
        "spd_map": spd.set_index(["PULocationID", "day_of_week", "hour"])["avg_speed"].to_dict(),
        "zone_spd": spd.groupby("PULocationID")["avg_speed"].mean().to_dict(),
        "spd_global": float(spd["avg_speed"].mean()),
    }


def get_distance(L, pu, do):
    # zone-pair median distance; fall back to borough-pair, then global median
    d = L["pair_map"].get((pu, do))
    if d is None:
        d = L["bor_map"].get((L["bmap"].get(pu), L["bmap"].get(do)), L["dist_global"])
    return float(d)


def predict_fares(L, pu, do, date, time, weather, temp):
    dow, hour, month = date.weekday(), time.hour, date.month
    dist = get_distance(L, pu, do)
    demand = float(L["dem_map"].get((pu, dow, hour), 0.0))
    speed = float(L["spd_map"].get((pu, dow, hour), L["zone_spd"].get(pu, L["spd_global"])))
    wm = L["wm"].loc[month]
    precip = {"Clear": 0.0, "Rain": 1.0, "Snow": 0.3}[weather]
    snow = 0.5 if weather == "Snow" else 0.0

    rows = []
    for s in SERVICES:
        r = {
            "est_distance": dist, "avg_demand": demand, "avg_speed": speed,
            "is_weekend": int(dow >= 5),
            "is_rush_hour": int(dow < 5 and (7 <= hour <= 9 or 16 <= hour <= 19)),
            "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
            "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
            "temperature_2m": temp, "precipitation": precip, "snowfall": snow,
            "wind_speed_10m": float(wm["wind_speed_10m"]), "cloud_cover": float(wm["cloud_cover"]),
            "is_raining": int(precip > 0.1), "is_snowing": int(snow > 0),
            "is_airport_pu": int(pu in AIRPORTS), "is_airport_do": int(do in AIRPORTS),
            "log_dist": np.log1p(dist), "dist_sq": dist ** 2,
        }
        for t in ["Green", "Lyft", "Uber"]:
            r[f"svc_{t}"] = int(s == t)
            r[f"dist_x_svc_{t}"] = dist * int(s == t)
            r[f"logdist_x_svc_{t}"] = np.log1p(dist) * int(s == t)
        rows.append(r)

    X = pd.DataFrame(rows).reindex(columns=L["columns"]).astype("float64")
    pred = np.expm1(L["model"].predict(L["scaler"].transform(X)))
    return pd.DataFrame({"Service": SERVICES, "Fare ($)": np.clip(pred, 3, None).round(2)}), dist


if __name__ == "__main__":
    st.set_page_config(page_title="NYC Fare Predictor", page_icon="🚕")
    L = st.cache_resource(load_all)()

    z = L["zones"]
    z = z[~z["LocationID"].isin([264, 265])]
    labels = (z["Zone"] + " (" + z["Borough"] + ")").tolist()
    ids = dict(zip(labels, z["LocationID"]))

    def idx(prefix):
        return next((i for i, l in enumerate(labels) if l.startswith(prefix)), 0)

    st.title("🚕 NYC Fare Predictor")
    st.write("Pick a pickup zone, a dropoff zone and a time to compare estimated fares "
             "for Yellow, Green, Uber and Lyft.")

    c1, c2 = st.columns(2)
    pu_l = c1.selectbox("Pickup", labels, index=idx("Times Sq"))
    do_l = c2.selectbox("Dropoff", labels, index=idx("JFK"))
    date = c1.date_input("Date")
    time = c2.time_input("Time", value=pd.Timestamp("18:00").time())
    weather = st.radio("Weather", ["Clear", "Rain", "Snow"], horizontal=True)
    month_temp = float(L["wm"].loc[date.month, "temperature_2m"])
    temp = st.slider("Temperature (°C)", -15.0, 40.0, round(month_temp, 1), key=f"t{date.month}")

    res, dist = predict_fares(L, ids[pu_l], ids[do_l], date, time, weather, temp)
    if L['bmap'].get(ids[pu_l]) == 'Manhattan':
        res = res[res['Service'] != 'Green']
        st.info('Green taxis cannot pick up in most of Manhattan, so Green is hidden for this pickup.')
    best = res.loc[res["Fare ($)"].idxmin()]
    st.metric("Cheapest option", f"{best['Service']}: ${best['Fare ($)']}")
    st.bar_chart(res.set_index("Service"))
    st.dataframe(res, hide_index=True)
    st.caption(f"Estimated distance: ~{dist:.1f} miles (median distance for this zone pair). "
               "Taxi fares are metered fares (excluding tip and tax); Uber/Lyft fares are "
               "base fares (excluding tolls and fees). Model trained on 2024 NYC trip data.")
