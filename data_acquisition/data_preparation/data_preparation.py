import pandas as pd
import numpy as np

df = pd.read_json("output.json")

df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")

df["area_sqm"] = pd.to_numeric(df["area_sqm"], errors="coerce").astype("Int64")
df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("Int64")

df["availability_date"] = df["availability_date"].replace({
    "sofort": pd.Timestamp("today").normalize(),
    "nach Vereinbarung": pd.NaT
})
df["availability_date"] = pd.to_datetime(df["availability_date"], errors="coerce")

def parse_floor(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if "ug" in value:
        return -1
    if "eg" in value:
        return 0
    match = pd.Series(value).str.extract(r"(-?\d+)")
    try:
        return int(match[0][0])
    except:
        return np.nan

df["floor"] = df["floor"].apply(parse_floor).astype("Int64")

df.to_json("cleaned_data.json", orient="records", indent=2, date_format="iso")