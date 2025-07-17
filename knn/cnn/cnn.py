import os
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from typing import List

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import gdown

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from database.select_data import get_all_properties

# === CONFIGURATION ===
IMAGE_FOLDER = "downloaded_images"
GOOGLE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1Pn1qXZ0B1MHb-KuYoHqcJBrz10Bh7PPV"
def download_images_from_drive(folder_url: str, output_dir: str):
    """
    Downloads all images from a Google Drive folder using gdown.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"🔗 Downloading images from Google Drive folder: {folder_url}")
    # gdown supports folder download via 'gdown.download_folder'
    try:
        import gdown
        gdown.download_folder(folder_url, output=output_dir, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"Error downloading images from Google Drive: {e}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLUSTERS = 10


# === IMAGE MODEL SETUP ===
def load_feature_extractor():
    model = models.resnet50(weights="IMAGENET1K_V2")
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval().to(DEVICE)
    return model


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def download_image(url: str, output_path: str):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)


def process_image(img_path: str, model) -> np.ndarray:
    try:
        image = Image.open(img_path).convert("RGB")
        image = image_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = model(image).cpu().numpy().flatten()
        return features
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return np.zeros(2048)


def extract_image_features(df: pd.DataFrame, id_col="id", url_col="image_urls") -> pd.DataFrame:
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    model = load_feature_extractor()
    image_vectors = []
    ids = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid = row[id_col]
        urls = row[url_col]
        if not urls or not isinstance(urls, list):
            continue
        image_url = urls[0]
        local_path = os.path.join(IMAGE_FOLDER, f"{pid}.jpg")
        try:
            download_image(image_url, local_path)
            vec = process_image(local_path, model)
            image_vectors.append(vec)
            ids.append(pid)
        except Exception as e:
            print(f"Skipping ID {pid}: {e}")

    image_df = pd.DataFrame(image_vectors, index=ids)
    image_df.columns = [f"img_feat_{i}" for i in range(image_df.shape[1])]
    image_df[id_col] = image_df.index
    return image_df.reset_index(drop=True)


# === DESCRIPTION CLUSTERING ===
def generate_description_clusters(df: pd.DataFrame, n_clusters: int = N_CLUSTERS) -> pd.Series:
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X = tfidf.fit_transform(df["description"].fillna(""))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(X)


# === STRUCTURED PREPROCESSING ===
def preprocess_structured_data(df: pd.DataFrame, is_rental_mode: bool = True) -> pd.DataFrame:
    df = df[df["price"].notnull()]
    df["availability_date"] = pd.to_datetime(df.get("availability_date"), errors="coerce")
    df["availability_days_from_now"] = (df["availability_date"] - datetime.now()).dt.days
    df["availability_days_from_now"] = df["availability_days_from_now"].fillna(999)

    df["has_balcony"] = df.get("has_balcony", False).astype(int)
    df["is_rental"] = df.get("is_rental", True).astype(int)

    if is_rental_mode:
        df = df[df["price"] < 10000]
    else:
        df = df[df["price"] > 80000]

    df["log_area_sqm"] = np.log1p(df["area_sqm"])
    df["availability_soon"] = (df["availability_days_from_now"] <= 30).astype(int)
    df["is_small"] = (df["area_sqm"] < 30).astype(int)
    df["is_large"] = (df["area_sqm"] > 80).astype(int)
    df["is_ground_floor"] = (df["floor"] <= 0).astype(int)
    df["is_top_floor"] = (df["floor"] > 4).astype(int)

    for label in ["is_new", "has_view", "has_garden", "has_parking", "has_air_conditioning"]:
        if label in df.columns:
            df[label] = df[label].astype(int)
        else:
            df[label] = 0

    return df


# === TRAINING ===
def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    categorical = [col for col in ["city", "region", "description_cluster"] if col in X.columns]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=150, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\n🎯 Evaluation Metrics:")
    print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    print(f"MAE: CHF {mean_absolute_error(y_test, y_pred):,.2f}")
    return pipeline


# === MAIN ENTRY ===
if __name__ == "__main__":
    # Download all images from Google Drive folder if not already present
    if not os.path.exists(IMAGE_FOLDER) or not os.listdir(IMAGE_FOLDER):
        download_images_from_drive(GOOGLE_DRIVE_FOLDER_URL, IMAGE_FOLDER)

    print("📦 Loading property data...")
    df = pd.DataFrame(get_all_properties())
    df = preprocess_structured_data(df, is_rental_mode=True)

    print("🧠 Generating description clusters...")
    df["description_cluster"] = generate_description_clusters(df)

    print("🖼️ Loading image URLs from images table...")
    import psycopg2
    conn = psycopg2.connect(
        dbname="real_estate_price_predictor",
        user="postgres",
        password="postgres",
        host="localhost",
        port=5433
    )
    images_df = pd.read_sql("SELECT property_id, url FROM images", conn)
    conn.close()

    # Group URLs by property_id
    image_urls_map = images_df.groupby("property_id")['url'].apply(list)
    df["image_urls"] = df["id"].map(image_urls_map)
    df["image_urls"] = df["image_urls"].apply(lambda x: x if isinstance(x, list) else [])

    print("🖼️ Extracting image features...")
    image_features = extract_image_features(df)

    print("🔗 Merging all features...")
    full_df = pd.merge(df, image_features, on="id", how="inner")

    X = full_df.drop(columns=[
        "price", "id", "title", "address", "availability_date", 
        "description", "image_urls"
    ])
    y = full_df["price"]

    print("🚀 Training model...")
    model = train_model(X, y)

    import joblib
    joblib.dump(model, "combined_price_model.joblib")
    print("✅ Model saved as combined_price_model.joblib")
