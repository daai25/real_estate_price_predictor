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

# Add the parent directory to the module path to import project-specific database module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from database.select_data import get_all_properties

def augment_with_noise(df, condition_col, condition_val=1, n_aug=300):
    rare = df[df[condition_col] == condition_val]
    if rare.empty:
        return df

    augmented = rare.sample(n=min(n_aug, len(rare)), replace=True)
    numeric_cols = augmented.select_dtypes(include=np.number).columns
    noise = np.random.normal(loc=0.0, scale=0.03, size=augmented[numeric_cols].shape)
    augmented[numeric_cols] = augmented[numeric_cols] * (1 + noise)

    return pd.concat([df, augmented], ignore_index=True)

def load_data():
    # Load and preprocess property data from the database.
    properties = get_all_properties()
    if not properties:
        raise ValueError("No property data found.")

    # Convert property data to DataFrame
    df = pd.DataFrame(properties)

    # Drop entries with missing price values
    df = df[df["price"].notnull()]

    # Calculate days until availability (if column exists)
    if "availability_date" in df.columns:
        df["availability_date"] = pd.to_datetime(df["availability_date"], errors="coerce")
        df["availability_days_from_now"] = (df["availability_date"] - datetime.now()).dt.days
    else:
        df["availability_days_from_now"] = None

    # Encode boolean features as integers (0 or 1)
    df["has_balcony"] = df.get("has_balcony", False).astype(int)
    df["is_rental"] = df.get("is_rental", True).astype(int)

    # Remove outliers to then be used for the properties model
    df = df[df["price"] < 10000]

    # Drop unused or non-numeric columns
    drop_fields = ["title", "address", "availability_date", "description", "image_urls", "id"]
    df = df.drop(columns=[col for col in drop_fields if col in df.columns])

    # Create new features
    df["log_area_sqm"] = df["area_sqm"].apply(lambda x: np.log1p(x))  # log1p for stability with small areas
    df["availability_soon"] = df["availability_days_from_now"].apply(lambda x: int(x is not None and x <= 30))
    df["is_small"] = (df["area_sqm"] < 30).astype(int)
    df["is_large"] = (df["area_sqm"] > 80).astype(int)
    df["lat_bin"] = pd.cut(df["latitude"], bins=10, labels=False)  # Latitude binned into 10 groups
    df["lon_bin"] = pd.cut(df["longitude"], bins=10, labels=False)  # Longitude binned into 10 groups
    df["is_ground_floor"] = (df["floor"] <= 0).astype(int)
    df["is_top_floor"] = (df["floor"] > 4).astype(int)

    for _ in range(20):
        df = augment_with_noise(df, condition_col="is_top_floor", condition_val=1)
        df = augment_with_noise(df, condition_col="is_large", condition_val=1)
        df = augment_with_noise(df, condition_col="is_small", condition_val=1)
    
    return df


def train_model(df):
    # Train a Random Forest regression model and evaluate its performance.
    
    # Features
    X = df.drop(columns=["price"])
    
    # Target
    y = df["price"]

    # Identify categorical and numerical features
    categorical_features = [col for col in ["city", "region"] if col in X.columns]
    numerical_features = [col for col in X.columns if col not in categorical_features]

    # Create preprocessing pipeline: OneHotEncode categorical features
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ], remainder="passthrough")

    # Create a full pipeline with preprocessing and model
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Output model performance
    print(f"Model evaluation:\nRMSE: {rmse:.2f} CHF\nR²: {r2:.2f}")
    
    # Plot actual vs predicted prices
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.grid(True)
    plt.show()

    return model


if __name__ == "__main__":
    try:
        # Load and prepare dataset
        df = load_data()
        
        # Train and evaluate model
        model = train_model(df)

        print("\n🔮 Now enter new property data for prediction:")

        # Ask the user for property details
        city = input("City: ")
        region = input("Region: ")
        zip_code = int(input("ZIP Code: "))
        rooms = float(input("Number of rooms: "))
        floor = int(input("Floor: "))
        area_sqm = float(input("Area in sqm: "))
        has_balcony = int(input("Has balcony? (0 = No, 1 = Yes): "))
        is_rental = 1  # Always 1 (since predicting rental price)
        availability_days_from_now = int(input("Days until available: "))
        latitude = float(input("Latitude: "))
        longitude = float(input("Longitude: "))

        # Calculate derived features from input
        log_area_sqm = np.log1p(area_sqm)
        availability_soon = int(availability_days_from_now <= 30)
        is_small = int(area_sqm < 30)
        is_large = int(area_sqm > 80)
        is_ground_floor = int(floor <= 0)
        is_top_floor = int(floor > 4)
        lat_bin = pd.cut([latitude], bins=10, labels=False)[0]
        lon_bin = pd.cut([longitude], bins=10, labels=False)[0]

        # Construct a DataFrame for prediction
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
            "log_area_sqm": log_area_sqm,
            "availability_soon": availability_soon,
            "is_small": is_small,
            "is_large": is_large,
            "is_ground_floor": is_ground_floor,
            "is_top_floor": is_top_floor,
            "latitude": latitude,
            "longitude": longitude,
            "lat_bin": lat_bin,
            "lon_bin": lon_bin
        }])

        # Predict price using trained model
        predicted_price = model.predict(sample)[0]
        print(f"\n💰 Predicted rental price: CHF {predicted_price:.2f}")

    except Exception as e:
        # Catch and print any errors
        print(f"Error: {e}")
