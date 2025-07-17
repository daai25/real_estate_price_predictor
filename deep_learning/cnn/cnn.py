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
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
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


def verify_image_files_match_database(images_df: pd.DataFrame, image_folder: str):
    """
    Verifies that downloaded image filenames match (property_id, image_id) pairs in the database.
    Logs missing or unmatched images.
    """
    # Set of valid (property_id, image_id) tuples from DB
    valid_pairs = {
        (str(row['property_id']), str(row['id']))
        for _, row in images_df.iterrows()
    }

    # Set of (property_id, image_id) from downloaded files
    downloaded_pairs = set()
    for fname in os.listdir(image_folder):
        if fname.lower().endswith('.jpg'):
            name = fname[:-4]  # strip '.jpg'
            if '-' in name:
                pid, imgid = name.split('-', 1)
                downloaded_pairs.add((pid, imgid))

    # Compare
    missing_in_drive = valid_pairs - downloaded_pairs
    extra_in_drive = downloaded_pairs - valid_pairs

    print(f"✅ Total DB image records: {len(valid_pairs)}")
    print(f"📂 Total downloaded image files: {len(downloaded_pairs)}")

    if missing_in_drive:
        print(f"⚠️ {len(missing_in_drive)} image(s) missing in drive:")
        for pid, imgid in sorted(missing_in_drive)[:10]:  # show only first 10
            print(f"  - Missing: {pid}-{imgid}.jpg")

    if extra_in_drive:
        print(f"⚠️ {len(extra_in_drive)} extra image(s) in drive not in DB:")
        for pid, imgid in sorted(extra_in_drive)[:10]:  # show only first 10
            print(f"  - Unmatched: {pid}-{imgid}.jpg")

    if not missing_in_drive and not extra_in_drive:
        print("✅ All downloaded images match the database entries.")
        
        
        
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



# === DESCRIPTION CLUSTERING (from KNN script) ===
def vectorize_descriptions(df: pd.DataFrame, max_features: int = 1000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(df['description'].fillna(""))
    terms = vectorizer.get_feature_names_out()
    return X, vectorizer, terms


def perform_kmeans(X, n_clusters: int = 10, random_state: int = 42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X)
    return kmeans, cluster_labels



# === STRUCTURED PREPROCESSING (from Random Forest script) ===
def preprocess_structured_data(df: pd.DataFrame, is_rental_mode: bool = True) -> pd.DataFrame:
    df = df[df["price"].notnull()]
    if "availability_date" in df.columns:
        df["availability_date"] = pd.to_datetime(df["availability_date"], errors="coerce")
        df["availability_days_from_now"] = (df["availability_date"] - datetime.now()).dt.days
    else:
        df["availability_days_from_now"] = None

    df["has_balcony"] = df.get("has_balcony", False).astype(int)
    df["is_rental"] = df.get("is_rental", True).astype(int)

    if is_rental_mode:
        df = df[df["price"] < 10000]
    else:
        df = df[df["price"] > 80000]

    # Feature engineering
    df["log_area_sqm"] = df["area_sqm"].apply(lambda x: np.log1p(x))
    df["availability_soon"] = df["availability_days_from_now"].apply(lambda x: int(x is not None and x <= 30))
    df["is_small"] = (df["area_sqm"] < 30).astype(int)
    df["is_large"] = (df["area_sqm"] > 80).astype(int)

    # Latitude/longitude binning with robust fallback
    if "latitude" in df.columns:
        df["lat_bin"] = pd.cut(df["latitude"], bins=10, labels=False)
    else:
        df["lat_bin"] = np.nan
    if "longitude" in df.columns:
        df["lon_bin"] = pd.cut(df["longitude"], bins=10, labels=False)
    else:
        df["lon_bin"] = np.nan

    df["is_ground_floor"] = (df["floor"] <= 0).astype(int)
    df["is_top_floor"] = (df["floor"] > 4).astype(int)

    # Add missing engineered columns from RF script
    for label in ["is_new", "has_view", "has_garden", "has_parking", "has_air_conditioning"]:
        if label in df.columns:
            df[label] = df[label].astype(int)
        else:
            df[label] = 0

    return df



# === TRAINING (from Random Forest script) ===
def train_model(X: pd.DataFrame, y: pd.Series, save_path=None):
    categorical_features = [col for col in ["city", "region", "description_cluster"] if col in X.columns]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [10, 20],
        "regressor__min_samples_split": [2, 5],
        "regressor__min_samples_leaf": [1, 2]
    }

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=kf,
        verbose=1,
        n_jobs=-1
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    if save_path:
        import joblib
        joblib.dump(best_model, save_path)
        print(f"📦 Tuned model saved to {save_path}")

    y_pred = best_model.predict(X_test)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Model evaluation:\nMAE: CHF {mean_absolute_error(y_test, y_pred):,.2f}\nR²: {r2_score(y_test, y_pred):.2f}")
    return best_model


# === MAIN ENTRY ===

if __name__ == "__main__":
    # Download all images from Google Drive folder if not already present
    if not os.path.exists(IMAGE_FOLDER) or not os.listdir(IMAGE_FOLDER):
        download_images_from_drive(GOOGLE_DRIVE_FOLDER_URL, IMAGE_FOLDER)

    print("📦 Loading property data...")
    df = pd.DataFrame(get_all_properties())
    df = preprocess_structured_data(df, is_rental_mode=True)

    print("🧠 Generating description clusters using modular KMeans logic...")
    X_text, vectorizer, terms = vectorize_descriptions(df)
    _, description_clusters = perform_kmeans(X_text, n_clusters=N_CLUSTERS)
    df["description_cluster"] = description_clusters

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

    # Validate image files
    verify_image_files_match_database(images_df, IMAGE_FOLDER)
    
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

    print("🚀 Training model (Random Forest logic)...")
    model = train_model(X, y, save_path="combined_price_model.joblib")
    print("✅ Model saved as combined_price_model.joblib")
