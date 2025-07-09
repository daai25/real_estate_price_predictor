

from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from database.select_data import get_all_properties


def load_data():
    properties = get_all_properties()
    if not properties:
        raise ValueError("No property data found.")

    df = pd.DataFrame(properties)
    df = df[df["price"].notnull()]

    if "availability_date" in df.columns:
        df["availability_date"] = pd.to_datetime(df["availability_date"], errors="coerce")
        df["availability_days_from_now"] = (df["availability_date"] - datetime.now()).dt.days
    else:
        df["availability_days_from_now"] = None

    df["has_balcony"] = df.get("has_balcony", False).astype(int)
    df["is_rental"] = df.get("is_rental", True).astype(int)
    df = df[df["price"] > 10000]

    drop_fields = ["title", "address", "availability_date", "description", "image_urls", "id"]
    df = df.drop(columns=[col for col in drop_fields if col in df.columns])
    
    df["price_per_sqm"] = df["price"] / df["area_sqm"]
    df["log_area_sqm"] = df["area_sqm"].apply(lambda x: np.log1p(x))
    df["availability_soon"] = df["availability_days_from_now"].apply(lambda x: int(x is not None and x <= 30))

    return df

def train_model(df):
    X = df.drop(columns=["price"])
    y = df["price"]

    categorical_features = [col for col in ["city", "region"] if col in X.columns]
    numerical_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ], remainder="passthrough")

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model evaluation:\nRMSE: {rmse:.2f} CHF\nR²: {r2:.2f}")

    return model

if __name__ == "__main__":
    try:
        df = load_data()
        model = train_model(df)

        print("\n🔮 Now enter new property data for prediction:")

        # Prompt the user for input
        city = input("City: ")
        region = input("Region: ")
        zip_code = int(input("ZIP Code: "))
        rooms = float(input("Number of rooms: "))
        floor = int(input("Floor: "))
        area_sqm = float(input("Area in sqm: "))
        has_balcony = int(input("Has balcony? (0 = No, 1 = Yes): "))
        is_rental = 0
        availability_days_from_now = int(input("Days until available: "))
        latitude = float(input("Latitude: "))
        longitude = float(input("Longitude: "))


        # Derived features
        price_per_sqm = 0  # Placeholder, not needed for prediction
        log_area_sqm = np.log1p(area_sqm)
        availability_soon = int(availability_days_from_now <= 30)

        # Build input sample as DataFrame
        sample = pd.DataFrame([{
            "zip_code": zip_code,
            "city": city,
            "region": region,
            "rooms": rooms,
            "floor": floor,
            "area_sqm": area_sqm,
            "has_balcony": has_balcony,
            "is_rental": is_rental,
            "availability_days_from_now": availability_days_from_now,
            "price_per_sqm": 0,
            "log_area_sqm": log_area_sqm,
            "availability_soon": availability_soon,
            "latitude": latitude,
            "longitude": longitude
        }])

        predicted_price = model.predict(sample)[0]
        print(f"\n💰 Predicted property price: CHF {predicted_price:.2f}")

    except Exception as e:
        print(f"Error: {e}")