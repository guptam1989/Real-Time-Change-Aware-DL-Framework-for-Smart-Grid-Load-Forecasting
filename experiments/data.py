"""Download and prepare the real-world load datasets used in the paper.

RWD1: UCI Individual Household Electric Power Consumption (Sceaux, France),
      resampled to hourly active power, joined with ERA5 weather (Open-Meteo).
RWD2: PJM East (PJME) hourly grid load, joined with ERA5 weather for
      Philadelphia (Open-Meteo).

Each prepared dataset is saved as data/<name>.csv with columns:
    time, load, temp, hum, wind
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data")

UCI_URL = ("https://archive.ics.uci.edu/static/public/235/"
           "individual+household+electric+power+consumption.zip")
PJM_URLS = [
    "https://raw.githubusercontent.com/archd3sai/Hourly-Energy-Consumption-Prediction/master/PJME_hourly.csv",
    "https://raw.githubusercontent.com/aretegratatimferociter/CSC422-Energy-Consumption/main/data/raw/PJME_hourly.csv",
]
METEO = "https://archive-api.open-meteo.com/v1/archive"


def _get(url, path):
    if os.path.exists(path):
        return path
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def weather(lat, lon, start, end, tz, name):
    path = os.path.join(RAW, f"weather_{name}.csv")
    if not os.path.exists(path):
        params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                      hourly="temperature_2m,relative_humidity_2m,wind_speed_10m",
                      timezone=tz)
        r = requests.get(METEO, params=params, timeout=600)
        r.raise_for_status()
        h = r.json()["hourly"]
        pd.DataFrame(h).to_csv(path, index=False)
    w = pd.read_csv(path, parse_dates=["time"])
    w.columns = ["time", "temp", "hum", "wind"]
    return w


def uci_household():
    zpath = _get(UCI_URL, os.path.join(RAW, "uci_household.zip"))
    with zipfile.ZipFile(zpath) as z:
        name = [n for n in z.namelist() if n.endswith(".txt")][0]
        df = pd.read_csv(io.BytesIO(z.read(name)), sep=";", na_values="?",
                         usecols=["Date", "Time", "Global_active_power"],
                         low_memory=False)
    df["time"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    s = df.set_index("time")["Global_active_power"].astype(float)
    # Hourly mean active power (kW); short gaps (<= 3 h) linearly interpolated,
    # longer outages filled with the value one week earlier.
    s = s.resample("1h").mean()
    s = s.interpolate(limit=3)
    s = s.fillna(s.shift(168)).fillna(s.shift(336)).interpolate()
    load = s.rename("load").reset_index()
    w = weather(48.778, 2.290, str(load.time.min().date()), str(load.time.max().date()),
                "Europe/Paris", "sceaux")
    d = load.merge(w, on="time", how="left").interpolate(limit_direction="both")
    d.to_csv(os.path.join(OUT, "rwd1_uci_household.csv"), index=False)
    return d


def pjm_east(start="2015-01-01"):
    path = os.path.join(RAW, "PJME_hourly.csv")
    for u in PJM_URLS:
        try:
            _get(u, path)
            break
        except Exception as e:  # try next mirror
            print("mirror failed:", u, e)
    df = pd.read_csv(path, parse_dates=["Datetime"])
    df = df.groupby("Datetime", as_index=False)["PJME_MW"].mean()  # DST duplicates
    s = df.set_index("Datetime")["PJME_MW"].sort_index()
    s = s.resample("1h").mean().interpolate(limit=6)
    s = s[s.index >= start]
    load = s.rename("load").reset_index().rename(columns={"Datetime": "time"})
    w = weather(39.952, -75.165, str(load.time.min().date()), str(load.time.max().date()),
                "America/New_York", "philadelphia")
    d = load.merge(w, on="time", how="left").interpolate(limit_direction="both")
    d.to_csv(os.path.join(OUT, "rwd2_pjm_east.csv"), index=False)
    return d


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    for f in (uci_household, pjm_east):
        d = f()
        print(f.__name__, d.shape, d.time.min(), d.time.max())
        print(d.describe().T[["mean", "std", "min", "max"]])
        print("NaN:", int(d.isna().sum().sum()))
