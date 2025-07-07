import glob
import os
import pandas as pd
import numpy as np
import re

# === 1. Load data ===
folder_path = os.path.abspath("data_acquisition/data_preparation/data/")
all_files = glob.glob(os.path.join(folder_path, "*.json"))

df_list = [pd.read_json(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

# === 2. Type conversions ===
df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
df["area_sqm"] = pd.to_numeric(df["area_sqm"], errors="coerce").astype("Int64")
df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("Int64")

# === 3. Parse availability_date ===
df["availability_date"] = df["availability_date"].replace({
    "sofort": pd.Timestamp("today").normalize(),
    "nach Vereinbarung": pd.Timestamp("today").normalize() + pd.Timedelta(days=30)
})
df["availability_date"] = pd.to_datetime(df["availability_date"], errors="coerce")

# === 4. Floor parsing ===
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

# Add fallback: if title contains "haus" and floor is still missing, set to 0
def fill_floor_from_title(row):
    if pd.isna(row["floor"]) and isinstance(row.get("title"), str):
        if "haus" in row["title"].lower() or "villa" in row["title"].lower():
            return 0
    return row["floor"]

df["floor"] = df.apply(fill_floor_from_title, axis=1).astype("Int64")

# === 4b. Fill floor from description if pattern matches ===
def fill_floor_from_description(row):
    if pd.notna(row["floor"]):
        return row["floor"]
    desc = row.get("description", "")
    if not isinstance(desc, str):
        return row["floor"]

    desc = desc.lower()

    if "erdgeschoss" in desc or "eg" in desc:
        return 0
    if "untergeschoss" in desc or "ug" in desc:
        return -1
    if "studio" in desc:
        return 1
    match = re.search(r"\b(\d{1,2})\.\s*(og|obergeschoss|etage|stock)", desc)
    if match:
        return int(match.group(1))
    return row["floor"]

df["floor"] = df.apply(fill_floor_from_description, axis=1).astype("Int64")

# === 4c. If title contains "laden" and floor still missing, set to 0 ===
def fill_floor_from_laden(row):
    if pd.isna(row["floor"]) and isinstance(row.get("title"), str):
        if "laden" in row["title"].lower() or "büro" in row["title"].lower():
            return 0
    return row["floor"]

df["floor"] = df.apply(fill_floor_from_laden, axis=1).astype("Int64")

# === 5. Fill rooms from title ===
room_keywords = ["büro", "studio", "gewerbe", "lager", "hobbyraum", "hobby", "laden", "ladenfläche"]

def fill_room_from_title(row):
    if pd.isna(row["rooms"]) and isinstance(row.get("title"), str):
        lower_title = row["title"].lower()
        if any(word in lower_title for word in room_keywords):
            return 1.0
    return row["rooms"]

df["rooms"] = df.apply(fill_room_from_title, axis=1)

def fill_room_from_description(row):
    if pd.isna(row["rooms"]) and isinstance(row.get("description"), str):
        if "studio" in row["description"].lower():
            return 1.0
    return row["rooms"]

df["rooms"] = df.apply(fill_room_from_description, axis=1)

# === 6. Extract zip/city from description ===
def extract_zip(desc):
    if not isinstance(desc, str):
        return np.nan
    match = re.search(r"\b(\d{4})\b", desc)
    return match.group(1) if match else np.nan

def extract_city(desc):
    if not isinstance(desc, str):
        return np.nan
    match = re.search(r"in\s+((?:[A-ZÄÖÜ][\wäöüß\-\.]+(?:\s+|$)){1,3})", desc)
    if match:
        city = match.group(1).strip()
        city = re.sub(r"\s+(erwartet|liegt|befindet|steht|wurde|bietet)\b.*", "", city, flags=re.IGNORECASE)
        return city.strip()
    return np.nan

for col, extractor in [("zip_code", extract_zip), ("city", extract_city)]:
    df[col] = df[col].fillna(df["description"].apply(extractor))

df["zip_code"] = pd.to_numeric(df["zip_code"], errors="coerce").astype("Int64")

# === 7. Load postal code mapping ===
postal_file = os.path.abspath("data_acquisition/data_preparation/data/ch_postal_codes.csv")
postal_df = pd.read_csv(postal_file, sep=";", encoding="utf-8-sig")
postal_df.rename(columns=lambda x: x.strip().replace('\ufeff', ''), inplace=True)

postal_df["city_clean"] = postal_df["Ortschaftsname"].str.strip().str.lower()
postal_df["zip_code"] = pd.to_numeric(postal_df["PLZ"], errors="coerce")
postal_df["region"] = postal_df["Kantonskürzel"].str.strip().str.upper()

city_zip_region = postal_df.groupby("city_clean").agg({
    "zip_code": "first",
    "region": "first"
}).reset_index()

# === NEW: Try to extract city from description based on postal list ===
city_list = sorted(postal_df["Ortschaftsname"].dropna().unique(), key=lambda x: -len(x))

def extract_city_from_description(desc):
    if not isinstance(desc, str):
        return np.nan
    desc_lower = desc.lower()
    for city in city_list:
        if city.lower() in desc_lower:
            return city
    return np.nan

# Only fill if still missing
df["city"] = df["city"].fillna(df["description"].apply(extract_city_from_description))

# === Validate zip_code against official list ===
valid_zip_codes = set(postal_df["zip_code"].dropna().unique())
df.loc[~df["zip_code"].isin(valid_zip_codes), "zip_code"] = pd.NA

# === 8. Backfill missing zip_code and region from city ===
def fill_from_city(row):
    if pd.notna(row["city"]):
        city_key = row["city"].strip().lower()
        match = city_zip_region[city_zip_region["city_clean"] == city_key]
        if not match.empty:
            if pd.isna(row["zip_code"]):
                row["zip_code"] = int(match["zip_code"].values[0])
            if pd.isna(row["region"]):
                row["region"] = match["region"].values[0]
    return row

df = df.apply(fill_from_city, axis=1)
df["region"] = df["region"].str.upper().str.strip()
df["zip_code"] = pd.to_numeric(df["zip_code"], errors="coerce").astype("Int64")

# === 8b. Backfill missing city and region from zip_code ===
zip_city_region = postal_df.groupby("zip_code").agg({
    "city_clean": "first",
    "region": "first"
}).reset_index()

def fill_from_zip(row):
    if pd.notna(row["zip_code"]):
        match = zip_city_region[zip_city_region["zip_code"] == row["zip_code"]]
        if not match.empty:
            if pd.isna(row["city"]):
                row["city"] = match["city_clean"].values[0].title()
            if pd.isna(row["region"]):
                row["region"] = match["region"].values[0]
    return row

df = df.apply(fill_from_zip, axis=1)

# === Extract area_sqm from description if missing ===
def extract_area(desc):
    if not isinstance(desc, str):
        return np.nan
    match = re.search(r"(\d{2,4})\s*(m²|m2|quadratmeter)", desc.lower())
    if match:
        return int(match.group(1))
    return np.nan

df["area_sqm"] = df["area_sqm"].fillna(df["description"].apply(extract_area)).astype("Int64")

# === 9. Estimate missing prices ===
def estimate_price(row):
    if pd.notna(row["price"]):
        return row["price"]
    if pd.isna(row["area_sqm"]) or pd.isna(row["region"]):
        return np.nan
    lower = row["area_sqm"] - 10
    upper = row["area_sqm"] + 10
    similar = df[
        (df["region"] == row["region"]) &
        (df["area_sqm"].between(lower, upper)) &
        (df["price"].notna())
    ]
    if not similar.empty:
        return int(similar["price"].mean())
    return np.nan

df["price"] = df.apply(estimate_price, axis=1)

# === 10. Drop remaining missing attributes ===
df = df[df["price"].notna()]
df = df[df["address"].notna()]
df = df[df["rooms"].notna()]
df = df[df["floor"].notna()]
df = df[df["area_sqm"].notna()]

def is_rental(row):
    text = (str(row.get("title", "")) + " " + str(row.get("description", ""))).lower()
    
    if any(w in text for w in ["miete", "vermietet", "zur miete", "monatlich", "rental"]):
        return True
    if any(w in text for w in ["kaufen", "kauf", "verkauf", "zum kauf", "einmalig", "kaufpreis", "buy"]):
        return False
    
    if row["price"] < 20000:
        return True
    if row["price"] >= 20000:
        return False
    
    return np.nan

df["is_rental"] = df.apply(is_rental, axis=1)

# === 11. Save cleaned file ===
output_path = os.path.abspath("data_acquisition/data_preparation/cleaned_data/cleaned_data.json")
print(f"✅ Saving {len(df)} cleaned entries to: {output_path}")
df.to_json(output_path, orient="records", indent=2, date_format="iso")
