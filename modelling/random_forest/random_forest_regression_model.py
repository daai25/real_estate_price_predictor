# Import standard and third-party libraries
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

# Import machine learning tools from scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Import geolocation library
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Add parent directory to path to allow importing the custom database module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from database.select_data import get_all_properties


# Function to augment rare cases in the dataset with small random noise (for class balancing)
def augment_with_noise(df, condition_col, condition_val=1, n_aug=300):
    rare = df[df[condition_col] == condition_val]
    if rare.empty:
        return df
    # Resample rare cases
    augmented = rare.sample(n=min(n_aug, len(rare)), replace=True)
    numeric_cols = augmented.select_dtypes(include=np.number).columns
    # Apply multiplicative noise
    noise = np.random.normal(loc=0.0, scale=0.03, size=augmented[numeric_cols].shape)
    augmented[numeric_cols] = augmented[numeric_cols] * (1 + noise)
    # Concatenate with original data
    return pd.concat([df, augmented], ignore_index=True)


# Function to convert a human-readable address into latitude and longitude
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


# Load and preprocess property data from the database
def load_data(is_rental_mode=True):
    properties = get_all_properties()
    if not properties:
        raise ValueError("No property data found.")

    df = pd.DataFrame(properties)
    df = df[df["price"].notnull()]  # Filter out rows with missing prices

    # Convert availability date to datetime and calculate days from now
    if "availability_date" in df.columns:
        df["availability_date"] = pd.to_datetime(df["availability_date"], errors="coerce")
        df["availability_days_from_now"] = (df["availability_date"] - datetime.now()).dt.days
    else:
        df["availability_days_from_now"] = None

    # Ensure boolean flags are integers
    df["has_balcony"] = df.get("has_balcony", False).astype(int)
    df["is_rental"] = df.get("is_rental", True).astype(int)

    # Remove outliers based on price, depending on rental vs. purchase
    if is_rental_mode:
        df = df[df["price"] < 10000]
    else:
        df = df[df["price"] > 80000]

    # Drop irrelevant or text-heavy fields
    drop_fields = ["title", "address", "availability_date", "description", "image_urls", "id"]
    df = df.drop(columns=[col for col in drop_fields if col in df.columns])

    # Feature engineering
    df["log_area_sqm"] = df["area_sqm"].apply(lambda x: np.log1p(x))  # log-transformed area
    df["availability_soon"] = df["availability_days_from_now"].apply(lambda x: int(x is not None and x <= 30))
    df["is_small"] = (df["area_sqm"] < 30).astype(int)
    df["is_large"] = (df["area_sqm"] > 80).astype(int)
    df["lat_bin"] = pd.cut(df["latitude"], bins=10, labels=False)
    df["lon_bin"] = pd.cut(df["longitude"], bins=10, labels=False)
    df["is_ground_floor"] = (df["floor"] <= 0).astype(int)
    df["is_top_floor"] = (df["floor"] > 4).astype(int)

    # Augment rare categories to help balance the model
    for _ in range(20):
        for col in ["is_top_floor", "is_large", "is_small"]:
            df = augment_with_noise(df, condition_col=col, condition_val=1)

    return df


# Function to train the regression model and visualize performance
def train_model(df):
    X = df.drop(columns=["price"])  # Features
    y = df["price"]  # Target variable

    # Select categorical columns to encode
    categorical_features = [col for col in ["city", "region"] if col in X.columns]

    # Create preprocessing pipeline with OneHotEncoding for categorical features
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ], remainder="passthrough")

    # Combine preprocessing and regression model in a pipeline
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model evaluation:\nRMSE: {rmse:.2f} CHF\nR²: {r2:.2f}")

    # Plot predicted vs. actual prices
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.grid(True)
    plt.show()

    return model


# Prompt user for new property details and prepare input DataFrame
def get_user_input(is_rental_mode):
    print("\n🔮 Enter new property data for prediction:")
    city = input("City: ")
    region = input("Region: ")
    zip_code = int(input("ZIP Code: "))
    rooms = float(input("Number of rooms: "))
    floor = int(input("Floor: "))
    area_sqm = float(input("Area in sqm: "))
    has_balcony = int(input("Has balcony? (0 = No, 1 = Yes): "))
    availability_days_from_now = int(input("Days until available: "))
    address = input("Street Address (e.g., Bahnhofstrasse 10): ")

    # Convert address to lat/lon
    full_address = f"{address}, {zip_code} {city}, {region}, Switzerland"
    latitude, longitude = geocode_address(full_address)
    print(f"📍 Coordinates found: Latitude={latitude}, Longitude={longitude}")

    # Derive engineered features for the new sample
    log_area_sqm = np.log1p(area_sqm)
    availability_soon = int(availability_days_from_now <= 30)
    is_small = int(area_sqm < 30)
    is_large = int(area_sqm > 80)
    is_ground_floor = int(floor <= 0)
    is_top_floor = int(floor > 4)
    lat_bin = pd.cut([latitude], bins=10, labels=False)[0]
    lon_bin = pd.cut([longitude], bins=10, labels=False)[0]

    # Return as a DataFrame with same structure as training data
    return pd.DataFrame([{
        "zip_code": zip_code,
        "city": city,
        "region": region,
        "rooms": rooms,
        "floor": floor,
        "area_sqm": area_sqm,
        "has_balcony": has_balcony,
        "is_rental": int(is_rental_mode),
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


# Main entry point of the program
if __name__ == "__main__":
    try:
        # User chooses prediction mode
        print("🏠 Property Price Prediction")
        print("Choose prediction mode:")
        print("1 = Rental price")
        print("2 = Purchase price")
        mode = input("Enter choice (1 or 2): ").strip()

        if mode not in ["1", "2"]:
            raise ValueError("Invalid mode selected. Please enter 1 or 2.")

        is_rental = (mode == "1")

        # Load data and train model
        df = load_data(is_rental_mode=is_rental)
        model = train_model(df)

        # Loop for multiple predictions
        while True:
            # Get new sample input and make a prediction
            sample = get_user_input(is_rental_mode=is_rental)
            predicted_price = model.predict(sample)[0]
            price_type = "rental" if is_rental else "purchase"
            print(f"\n💰 Predicted {price_type} price: CHF {predicted_price:.2f}")

            # Ask if user wants another prediction
            again = input("\n🔁 Do you want to make another prediction? (y/n): ").strip().lower()
            if again != "y":
                print("👋 Goodbye!")
                break

    except Exception as e:
        print(f"Error: {e}")
