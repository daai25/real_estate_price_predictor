import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from database.select_data import get_all_properties


def load_data(is_rental: bool) -> pd.DataFrame:
    properties = get_all_properties()
    if not properties:
        raise ValueError("No property data found.")
    df = pd.DataFrame(properties)

    df = df[df["price"].notnull()]

    if is_rental:
        df = df[df["price"] < 10000]
    else:
        df = df[df["price"] > 80000]

    df["has_balcony"] = df.get("has_balcony", False).astype(int)
    df["is_rental"] = df.get("is_rental", True).astype(int)

    if "availability_date" in df.columns:
        df["availability_date"] = pd.to_datetime(df["availability_date"], errors="coerce")
        df["availability_days_from_now"] = (
            df["availability_date"] - datetime.now()
        ).dt.days.fillna(0)
    else:
        df["availability_days_from_now"] = 0

    df["log_area_sqm"] = np.log1p(df["area_sqm"])
    df["availability_soon"] = df["availability_days_from_now"] <= 30
    df["is_small"] = (df["area_sqm"] < 30).astype(int)
    df["is_large"] = (df["area_sqm"] > 80).astype(int)
    df["is_ground_floor"] = (df["floor"] <= 0).astype(int)
    df["is_top_floor"] = (df["floor"] > 4).astype(int)

    if "latitude" in df.columns and "longitude" in df.columns:
        df["lat_bin"] = pd.cut(df["latitude"], bins=10, labels=False)
        df["lon_bin"] = pd.cut(df["longitude"], bins=10, labels=False)

    drop_cols = ["title", "address", "availability_date", "description", "image_urls", "id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


def train_model(df: pd.DataFrame, model_path: str) -> Pipeline:
    X = df.drop(columns=["price"])
    y = df["price"]

    categorical = [col for col in ["city", "region"] if col in X.columns]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"📊 Model evaluation:")
    print(f"→ RMSE: {rmse:.2f} CHF")
    print(f"→ R²: {r2:.2f}")

    joblib.dump(pipeline, model_path)
    print(f"✅ Model saved to: {model_path}")
    return pipeline


def get_user_input(is_rental: bool) -> pd.DataFrame:
    print("\n📝 Enter property details:")
    city = input("City: ").strip()
    region = input("Region (2-letter code): ").strip()
    zip_code = int(input("ZIP Code: "))
    rooms = float(input("Rooms: "))
    floor = int(input("Floor: "))
    area_sqm = float(input("Area (sqm): "))
    has_balcony = input("Has balcony? (yes/no): ").strip().lower() == "yes"
    days_available = int(input("Days until available: "))
    address = input("Street Address (e.g., Bahnhofstrasse 10): ")

    # Convert address to lat/lon
    full_address = f"{address}, {zip_code} {city}, {region}, Switzerland"
    latitude, longitude = geocode_address(full_address)
    print(f"📍 Coordinates found: Latitude={latitude}, Longitude={longitude}")

    return pd.DataFrame([{
        "city": city,
        "region": region,
        "zip_code": zip_code,
        "rooms": rooms,
        "floor": floor,
        "area_sqm": area_sqm,
        "has_balcony": int(has_balcony),
        "is_rental": int(is_rental),
        "availability_days_from_now": days_available,
        "log_area_sqm": np.log1p(area_sqm),
        "availability_soon": int(days_available <= 30),
        "is_small": int(area_sqm < 30),
        "is_large": int(area_sqm > 80),
        "is_ground_floor": int(floor <= 0),
        "is_top_floor": int(floor > 4),
        "latitude": latitude,
        "longitude": longitude,
        "lat_bin": pd.cut([latitude], bins=10, labels=False)[0],
        "lon_bin": pd.cut([longitude], bins=10, labels=False)[0],
    }])
    
def geocode_address(address):
    try:
        geolocator = Nominatim(user_agent="property-price-predictor")
        location = geolocator.geocode(address, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            raise ValueError("Address could not be geocoded.")
    except GeocoderTimedOut:
        raise ValueError("Geocoding service timed out. Try again.")


def main():
    print("🏡 Linear Regression Price Prediction")
    print("Choose mode:\n1 - Rental\n2 - Purchase")
    choice = input("Enter 1 or 2: ").strip()

    if choice not in ["1", "2"]:
        print("Invalid mode.")
        return


    is_rental = (choice == "1")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_filename = "linear_model_rental.pkl" if is_rental else "linear_model_purchase.pkl"
    model_path = os.path.join(script_dir, model_filename)

    if os.path.exists(model_path):
        print("📦 Loading existing model...")
        model = joblib.load(model_path)
    else:
        print("📚 Training new model...")
        df = load_data(is_rental)
        model = train_model(df, model_path)

    while True:
        sample = get_user_input(is_rental)
        prediction = model.predict(sample)[0]
        print(f"\n💰 Predicted {'rental' if is_rental else 'purchase'} price: CHF {prediction:,.2f}")

        again = input("Predict another? (y/n): ").strip().lower()
        if again != "y":
            print("👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
