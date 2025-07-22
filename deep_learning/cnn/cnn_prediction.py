import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image

import torch
from torchvision import models, transforms


# Import preprocessing functions from training script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cnn_training import preprocess_structured_data, load_feature_extractor

# === Configuration ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "rental": os.path.join(SCRIPT_DIR, "combined_price_model_rental.joblib"),
    "purchase": os.path.join(SCRIPT_DIR, "combined_price_model_purchase.joblib")
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def choose_mode():
    print("Select prediction mode:")
    print("1. Rental")
    print("2. Purchase")
    while True:
        mode_input = input("Enter 1 for Rental or 2 for Purchase: ").strip()
        if mode_input == "1":
            return "rental"
        elif mode_input == "2":
            return "purchase"
        else:
            print("Invalid input. Please enter 1 or 2.")

def load_model(mode):
    model_path = MODEL_PATHS[mode]
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        sys.exit(1)
    return joblib.load(model_path)

# === Image transform ===
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Functions ===

def geocode_address(address):
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
        geolocator = Nominatim(user_agent="property-price-predictor-cnn")
        location = geolocator.geocode(address, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            raise ValueError("Address could not be geocoded.")
    except GeocoderTimedOut:
        raise ValueError("Geocoding service timed out. Try again.")

def get_user_input():
    print("📥 Please enter the following features of the property:")
    city = input("City (e.g., Zurich): ")
    region = input("Region (e.g., Zurich): ")
    zip_code = input("ZIP Code: ")
    address = input("Street Address (e.g., Bahnhofstrasse 10): ")
    full_address = f"{address}, {zip_code} {city}, {region}, Switzerland"
    try:
        latitude, longitude = geocode_address(full_address)
        print(f"📍 Coordinates found: Latitude={latitude}, Longitude={longitude}")
    except Exception as e:
        print(f"❌ Geocoding failed: {e}")
        latitude, longitude = 0.0, 0.0

    data = {
        "zip_code": zip_code,
        "area_sqm": float(input("Area (m²): ")),
        "floor": int(input("Floor (0 = ground): ")),
        "rooms": float(input("Number of rooms: ")),
        "latitude": latitude,
        "longitude": longitude,
        "city": city,
        "region": region,
        "description_cluster": int(input("Description cluster ID (0–9, or 0 if unknown): ")),
        "has_balcony": int(input("Balcony? (1 = Yes, 0 = No): ")),
        "has_view": int(input("View? (1 = Yes, 0 = No): ")),
        "has_garden": int(input("Garden? (1 = Yes, 0 = No): ")),
        "has_parking": int(input("Parking? (1 = Yes, 0 = No): ")),
        "has_air_conditioning": int(input("Air conditioning? (1 = Yes, 0 = No): ")),
        "is_new": int(input("New building? (1 = Yes, 0 = No): ")),
        "availability_date": input("Available from (YYYY-MM-DD, blank = today): ") or str(datetime.now().date()),
        "is_rental": 1,
        "id": "user_input"
    }
    return pd.DataFrame([data])

def extract_image_features(image_path: str) -> np.ndarray:
    try:
        model = load_feature_extractor()
        image = Image.open(image_path).convert("RGB")
        image = image_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = model(image).cpu().numpy().flatten()
        return features
    except Exception as e:
        print(f"❌ Error extracting image features: {e}")
        return np.zeros(2048)

def attach_image_features(df: pd.DataFrame, image_path: str):
    print("🖼️ Extracting image features...")
    features = extract_image_features(image_path)
    for i in range(len(features)):
        df[f"img_feat_{i}"] = features[i]
    return df

# === Main ===
def main():

    mode = choose_mode()
    is_rental = (mode == "rental")
    df = get_user_input()
    df["is_rental"] = int(is_rental)

    # Optional image
    image_path = input("📷 Path to image file (.jpg, optional): ").strip()
    use_image = image_path.lower().endswith(".jpg") and os.path.exists(image_path)

    # Preprocess structured features (skip filtering by 'price' for prediction input)
    if 'price' in df.columns:
        df = preprocess_structured_data(df, is_rental_mode=is_rental)
    else:
        df = preprocess_structured_data(df.copy(), is_rental_mode=is_rental)

    # Attach image features if available
    if use_image:
        df = attach_image_features(df, image_path)
    else:
        print("⚠️ No image provided. Using zeros for image features.")
        for i in range(2048):
            df[f"img_feat_{i}"] = 0.0

    # Drop unused columns
    for col in ["id", "availability_date"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Load model and predict
    model = load_model(mode)
    print("🔮 Predicting price...")
    prediction = model.predict(df)[0]
    print(f"💰 Predicted price: CHF {prediction:,.2f}")

if __name__ == "__main__":
    main()
