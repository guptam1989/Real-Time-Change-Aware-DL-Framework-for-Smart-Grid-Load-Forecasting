"""Held-out evaluation datasets (prepared after the method was frozen).

ENTSO-E country loads from Open Power System Data (OPSD, 2020-10-06 release,
hourly, 2015-01-01 to 2020-09-30): Germany, France, Italy, Spain, Great Britain.
EIA-930 balancing-authority demand ("Demand (MW) (Adjusted)", hourly,
2018-07-01 to 2021-06-30): ERCOT, CAISO, SPP.

Weather: hourly ERA5 reanalysis (Open-Meteo archive) averaged over three large
cities per region. Calendar features use local time. Output files
data/<name>.csv with columns time (UTC), local_time, load, temp, hum, wind.
"""
import os

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data")
OPSD_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
EIA_URL = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{}.csv"
EIA_PARTS = ["2018_Jul_Dec", "2019_Jan_Jun", "2019_Jul_Dec", "2020_Jan_Jun", "2020_Jul_Dec", "2021_Jan_Jun"]
METEO = "https://archive-api.open-meteo.com/v1/archive"

REGIONS = {
    # name: (source, column / BA code, time zone, [(lat, lon), ...])
    "DE": ("opsd", "DE_load_actual_entsoe_transparency", "Europe/Berlin",
           [(52.52, 13.405), (48.137, 11.575), (50.938, 6.960)]),          # Berlin, Munich, Cologne
    "FR": ("opsd", "FR_load_actual_entsoe_transparency", "Europe/Paris",
           [(48.857, 2.352), (45.764, 4.836), (43.297, 5.370)]),           # Paris, Lyon, Marseille
    "IT": ("opsd", "IT_load_actual_entsoe_transparency", "Europe/Rome",
           [(45.464, 9.190), (41.903, 12.496), (40.852, 14.268)]),         # Milan, Rome, Naples
    "ES": ("opsd", "ES_load_actual_entsoe_transparency", "Europe/Madrid",
           [(40.417, -3.704), (41.385, 2.173), (37.389, -5.984)]),         # Madrid, Barcelona, Seville
    "GB": ("opsd", "GB_GBN_load_actual_entsoe_transparency", "Europe/London",
           [(51.507, -0.128), (52.486, -1.890), (53.481, -2.242)]),        # London, Birmingham, Manchester
    "ERCOT": ("eia", "ERCO", "America/Chicago",
              [(32.777, -96.797), (29.760, -95.370), (30.267, -97.743)]),  # Dallas, Houston, Austin
    "CAISO": ("eia", "CISO", "America/Los_Angeles",
              [(34.052, -118.244), (37.775, -122.419), (38.582, -121.494)]),  # LA, San Francisco, Sacramento
    "SPP": ("eia", "SWPP", "America/Chicago",
            [(35.468, -97.516), (39.100, -94.579), (41.257, -95.935)]),    # Oklahoma City, Kansas City, Omaha
}


def _get(url, path):
    if not os.path.exists(path):
        with requests.get(url, stream=True, timeout=900) as r:
            r.raise_for_status()
            with open(path + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        os.replace(path + ".part", path)
    return path


def weather(coords, start, end, name):
    path = os.path.join(RAW, f"weather_{name}.csv")
    if not os.path.exists(path):
        frames = []
        for lat, lon in coords:
            r = requests.get(METEO, params=dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                                                hourly="temperature_2m,relative_humidity_2m,wind_speed_10m",
                                                timezone="GMT"), timeout=900)
            r.raise_for_status()
            frames.append(pd.DataFrame(r.json()["hourly"]))
        w = frames[0][["time"]].copy()
        for c in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]:
            w[c] = np.mean([f[c].values for f in frames], axis=0)
        w.to_csv(path, index=False)
    w = pd.read_csv(path, parse_dates=["time"])
    w.columns = ["time", "temp", "hum", "wind"]
    return w


def clean(s):
    """Hourly UTC series: implausible values removed, short gaps interpolated,
    long gaps filled with the value one week earlier."""
    s = s.astype(float)
    med = s.median()
    s[(s <= 0.2 * med) | (s > 3 * med)] = np.nan
    s = s.interpolate(limit=6, limit_area="inside")
    for lag in (168, 336, 504):
        s = s.fillna(s.shift(lag))
    return s.interpolate(limit_direction="both")


def opsd(col):
    path = _get(OPSD_URL, os.path.join(RAW, "opsd_time_series_60min_singleindex.csv"))
    df = pd.read_csv(path, usecols=["utc_timestamp", col])
    t = pd.to_datetime(df.utc_timestamp, utc=True).dt.tz_convert(None)
    s = pd.Series(df[col].values, index=t)
    s = s[(s.index >= "2015-01-01") & (s.index < "2020-10-01")]
    return s.resample("1h").mean()


def eia(ba):
    frames = []
    for part in EIA_PARTS:
        p = _get(EIA_URL.format(part), os.path.join(RAW, f"EIA930_BALANCE_{part}.csv"))
        df = pd.read_csv(p, usecols=["Balancing Authority", "UTC Time at End of Hour", "Demand (MW) (Adjusted)"],
                         thousands=",", low_memory=False)
        frames.append(df[df["Balancing Authority"] == ba])
    df = pd.concat(frames)
    t = pd.to_datetime(df["UTC Time at End of Hour"], format="%m/%d/%Y %I:%M:%S %p") - pd.Timedelta(hours=1)
    s = pd.Series(pd.to_numeric(df["Demand (MW) (Adjusted)"], errors="coerce").values, index=t).sort_index()
    s = s[~s.index.duplicated()]
    s = s[(s.index >= "2018-07-01") & (s.index < "2021-07-01")]
    return s.resample("1h").mean()


def prepare(name):
    src, key, tz, coords = REGIONS[name]
    raw = opsd(key) if src == "opsd" else eia(key)
    miss = float(raw.isna().mean())
    s = clean(raw)
    s.index.name = "time"
    d = s.rename("load").reset_index()
    d["local_time"] = d.time.dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
    w = weather(coords, str(d.time.min().date()), str(d.time.max().date()), name)
    d = d.merge(w, on="time", how="left")
    d[["temp", "hum", "wind"]] = d[["temp", "hum", "wind"]].interpolate(limit_direction="both")
    d.to_csv(os.path.join(OUT, f"ho_{name.lower()}.csv"), index=False)
    return d, miss


if __name__ == "__main__":
    import sys
    os.makedirs(RAW, exist_ok=True)
    for name in (sys.argv[1:] or REGIONS):
        d, miss = prepare(name)
        print(f"{name}: {len(d)} rows {d.time.min()} .. {d.time.max()}  raw missing {100 * miss:.2f}%  "
              f"load mean {d.load.mean():.0f} sd {d.load.std():.0f}  NaN {int(d.isna().sum().sum())}", flush=True)
