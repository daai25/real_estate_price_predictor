import pickle
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
import xgboost as xgb

kauf_model = xgb.XGBRegressor()
kauf_model.load_model(os.path.join(script_dir, "xgboost_kauf_model.json"))

# Modelle laden
miete_model = xgb.XGBRegressor()
miete_model.load_model(os.path.join(script_dir, "xgboost_miete_model.json"))
kauf_model = xgb.XGBRegressor()
kauf_model.load_model(os.path.join(script_dir, "xgboost_kauf_model.json"))


with open(os.path.join(script_dir, "kmeans.pkl"), "rb") as f:
    kmeans = pickle.load(f)

with open(os.path.join(script_dir, "ohe.pkl"), "rb") as f:
    ohe = pickle.load(f)

# Load feature names for both models
with open(os.path.join(script_dir, "xgboost_miete_feature_names.pkl"), "rb") as f:
    miete_feature_names = pickle.load(f)
with open(os.path.join(script_dir, "xgboost_kauf_feature_names.pkl"), "rb") as f:
    kauf_feature_names = pickle.load(f)

geolocator = Nominatim(user_agent="real_estate_predictor")

keywords = [
    'luxus', 'renoviert', 'neubau', 'altbau', 'hochwertig', 'modern',
    'barrierefrei', 'parkett', 'panoramafenster', 'einbauküche',
    'naturstein', 'quooker', 'wärmepumpe', 'photovoltaik', 'kamin',
    'begehbar', 'regendusche', 'grosszügig', 'hell', 'lichtdurchflutet', 'offen', 'bodenheizung',
    'terrasse', 'balkon', 'garten', 'loggia',
    'e-mobilität', 'aufzug', 'garage', 'tiefgarage', 'klimatisiert',
    'aktionspreis', 'rabatt', 'erstbezug', 'sommer-aktion',
    'ruhig', 'zentral', 'aussicht', 'rheinnähe', 'altstadt', 'autofrei'
]

def geocode(address):
    location = geolocator.geocode(address)
    if location:
        return location.longitude, location.latitude
    else:
        raise ValueError("Adresse konnte nicht geocodiert werden")

def predict_price(user_input: dict, mode: str):
    assert mode in ["miete", "kauf"], "Mode muss 'miete' oder 'kauf' sein"

    address_str = f"{user_input['address']}, {user_input['zip_code']} {user_input['city']}, {user_input['region']}"
    lon, lat = geocode(address_str)

    room_size = user_input['area_sqm'] / user_input['rooms']
    geo_cluster = kmeans.predict([[lon, lat]])[0]

    kw_features = {}
    for kw in keywords:
        kw_features[f'kw_{kw}'] = int(kw in user_input.get('title', '').lower() or kw in user_input.get('description', '').lower())

    df = pd.DataFrame([{
        'rooms': user_input['rooms'],
        'area_sqm': user_input['area_sqm'],
        'floor': user_input['floor'],
        'zip_code': user_input['zip_code'],
        'city': user_input['city'],
        'region': user_input['region'],
        'has_balcony': int(user_input['has_balcony']),
        'longitude': lon,
        'latitude': lat,
        'price_per_sqm': 0,
        'room_size': room_size,
        'geo_cluster': geo_cluster,
        **kw_features
    }])

    cat_cols = ['zip_code', 'city', 'region', 'geo_cluster']
    X_cat = ohe.transform(df[cat_cols])
    X_num = df.drop(columns=cat_cols).copy()

    # Combine and align features
    all_features = list(X_num.columns) + list(ohe.get_feature_names_out(cat_cols))
    X_final_df = pd.DataFrame(np.hstack([X_num.values, X_cat]), columns=all_features)

    if mode == "miete":
        X_final_df = X_final_df.reindex(columns=miete_feature_names, fill_value=0)
        log_pred = miete_model.predict(X_final_df.values)[0]
    else:
        X_final_df = X_final_df.reindex(columns=kauf_feature_names, fill_value=0)
        log_pred = kauf_model.predict(X_final_df.values)[0]

    pred = np.expm1(log_pred)
    return pred


if __name__ == "__main__":
    print("=== Real Estate Price Predictor ===")
    mode = input("Modus (miete/kauf): ").strip().lower()
    rooms = float(input("Anzahl Zimmer: "))
    area_sqm = float(input("Wohnfläche (m²): "))
    floor = int(input("Etage: "))
    zip_code = input("PLZ: ").strip()
    city = input("Stadt: ").strip()
    region = input("Kanton/Region: ").strip()
    address = input("Adresse (Strasse + Nr.): ").strip()
    balcony = input("Balkon (ja/nein): ").strip().lower() == "ja"
    title = input("Titel: ").strip()

    print("Beschreibung (mehrzeilig, mit 'END' abschliessen):")
    description_lines = []
    while True:
        line = input()
        if "END" in line.strip().upper():
            break
        description_lines.append(line)
    description = "\n".join(description_lines)

    user_input = {
        'rooms': rooms,
        'area_sqm': area_sqm,
        'floor': floor,
        'zip_code': zip_code,
        'city': city,
        'region': region,
        'address': address,
        'has_balcony': balcony,
        'title': title,
        'description': description
    }

    price = predict_price(user_input, mode=mode)
    print(f"\nGeschätzter {mode}-Preis: {price:,.2f} CHF")
