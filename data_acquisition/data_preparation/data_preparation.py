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
df["rooms"] = df["rooms"].astype(str).str.replace(r" ½", ".5", regex=True)
df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
df["area_sqm"] = df["area_sqm"].astype(str).str.extract(r"(\d{2,5})")[0]
df["area_sqm"] = pd.to_numeric(df["area_sqm"], errors="coerce").astype("Int64")
df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("Int64")

# === 3. Parse availability_date ===
df["availability_date"] = df["availability_date"].replace({
    "sofort": pd.Timestamp("today").normalize(),
    "nach Vereinbarung": pd.Timestamp("today").normalize() + pd.Timedelta(days=30),
    "immediately": pd.Timestamp("today").normalize(),
    "by arrangement": pd.Timestamp("today").normalize() + pd.Timedelta(days=30),
    "by agreement": pd.Timestamp("today").normalize() + pd.Timedelta(days=30)
})
df["availability_date"] = pd.to_datetime(df["availability_date"], errors="coerce")

# === 4. Floor parsing ===
def parse_floor(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    if any(x in value for x in ["ug", "basement"]):
        return -1
    if any(x in value for x in ["eg", "erdgeschoss", "ground floor"]):
        return 0
    match = pd.Series(value).str.extract(r"(-?\d+)")
    try:
        return int(match[0][0])
    except:
        return np.nan

df["floor"] = df["floor"].apply(parse_floor).astype("Int64")

# Fallback: title indicates "house"/"villa"
def fill_floor_from_title(row):
    if pd.isna(row["floor"]) and isinstance(row.get("title"), str):
        if any(x in row["title"].lower() for x in ["haus", "villa", "house", "villa"]):
            return 0
    return row["floor"]

df["floor"] = df.apply(fill_floor_from_title, axis=1).astype("Int64")

# Fill floor from description if pattern matches
def fill_floor_from_description(row):
    if pd.notna(row["floor"]):
        return row["floor"]
    desc = row.get("description", "")
    if not isinstance(desc, str):
        return row["floor"]

    desc = desc.lower()

    if any(x in desc for x in ["erdgeschoss", "eg", "ground floor"]):
        return 0
    if any(x in desc for x in ["untergeschoss", "ug", "basement"]):
        return -1
    if "studio" in desc:
        return 1
    match = re.search(r"\b(\d{1,2})\.\s*(og|obergeschoss|etage|stock|floor|level)", desc)
    if match:
        return int(match.group(1))
    return row["floor"]

df["floor"] = df.apply(fill_floor_from_description, axis=1).astype("Int64")

# Fallback: title contains "laden"/"store"/"office"
def fill_floor_from_laden(row):
    if pd.isna(row["floor"]) and isinstance(row.get("title"), str):
        if any(x in row["title"].lower() for x in ["laden", "büro", "store", "office"]):
            return 0
    return row["floor"]

df["floor"] = df.apply(fill_floor_from_laden, axis=1).astype("Int64")

# === 5. Fill rooms from title and description ===
room_keywords = [
    "büro", "studio", "gewerbe", "lager", "hobbyraum", "hobby", "laden", "ladenfläche",
    "office", "studio", "commercial", "storage", "shop", "retail", "workspace"
]

def fill_room_from_title(row):
    if pd.isna(row["rooms"]) and isinstance(row.get("title"), str):
        if any(word in row["title"].lower() for word in room_keywords):
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
        city = re.sub(r"\s+(erwartet|liegt|befindet|steht|wurde|bietet|is|offers|located|found)\b.*", "", city, flags=re.IGNORECASE)
        return city.strip()
    return np.nan

for col, extractor in [("zip_code", extract_zip), ("city", extract_city)]:
    df[col] = df[col].fillna(df["description"].apply(extractor))

df["zip_code"] = pd.to_numeric(df["zip_code"], errors="coerce").astype("Int64")

# === 7. Load postal code mapping ===
postal_file = os.path.abspath("data_acquisition/data_preparation/data/ch_postal_codes.csv")
postal_df = pd.read_csv(postal_file, sep=";", encoding="utf-8-sig")
postal_df.rename(columns=lambda x: x.strip().replace('\ufeff', ''), inplace=True)

postal_df["zip_code"] = pd.to_numeric(postal_df["PLZ"], errors="coerce")
postal_df.rename(columns={"Kantonskürzel": "region", "Ortschaftsname": "city"}, inplace=True)
postal_df["region"] = postal_df["region"].str.strip().str.upper()
postal_df["city_clean"] = postal_df["city"].str.strip().str.lower()

city_zip_region = postal_df.groupby("city_clean").agg({
    "zip_code": "first",
    "region": "first"
}).reset_index()

# === NEW: Try to extract city from description based on postal list ===
city_list = sorted(postal_df["city"].dropna().unique(), key=lambda x: -len(x))

def extract_city_from_description(desc):
    if not isinstance(desc, str):
        return np.nan
    desc_lower = desc.lower()
    for city in city_list:
        if city.lower() in desc_lower:
            return city
    return np.nan

df["city"] = df["city"].fillna(df["description"].apply(extract_city_from_description))

# === Validate zip_code against official list ===
valid_zip_codes = set(postal_df["zip_code"].dropna().unique())
df.loc[~df["zip_code"].isin(valid_zip_codes), "zip_code"] = pd.NA

df["region"] = df["region"].apply(lambda x: x if isinstance(x, str) and len(x.strip()) <= 2 else pd.NA)

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
    match = re.search(r"(\d{2,4})\s*(m²|m2|quadratmeter|square meters|sqm)", desc.lower())
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
df = df[df["availability_date"].notna()]
df = df[df["description"].notna()]

# === 10b. Classify rental or sale ===
def is_rental(row):
    text = (str(row.get("title", "")) + " " + str(row.get("description", ""))).lower()
    
    rental_keywords = ["miete", "vermietet", "zur miete", "monatlich", "rental", "rent", "per month"]
    sale_keywords = ["kaufen", "kauf", "verkauf", "zum kauf", "einmalig", "kaufpreis", "buy", "purchase", "one-time"]

    if any(w in text for w in rental_keywords):
        return True
    if any(w in text for w in sale_keywords):
        return False
    
    if row["price"] < 20000:
        return True
    if row["price"] >= 20000:
        return False
    
    return np.nan

df["is_rental"] = df.apply(is_rental, axis=1)

df["title"] = df.apply(
    lambda row: "Rental" if pd.isna(row["title"]) and row["is_rental"] is True
    else ("Property" if pd.isna(row["title"]) and row["is_rental"] is False
          else row["title"]),
    axis=1
)

# === 11. Save cleaned file ===
output_path = os.path.abspath("data_acquisition/data_preparation/cleaned_data/cleaned_data.json")
print(f"✅ Saving {len(df)} cleaned entries to: {output_path}")
df.to_json(output_path, orient="records", indent=2, date_format="iso")
