# === DATA AUGMENTATION ===
def augment_with_noise(df, condition_col, condition_val=1, n_aug=300):
    rare = df[df[condition_col] == condition_val]
    if rare.empty:
        return df
    augmented = rare.sample(n=min(n_aug, len(rare)), replace=True)
    numeric_cols = augmented.select_dtypes(include=np.number).columns
    noise = np.random.normal(loc=0.0, scale=0.03, size=augmented[numeric_cols].shape)
    augmented[numeric_cols] = augmented[numeric_cols] * (1 + noise)
    return pd.concat([df, augmented], ignore_index=True)
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from tqdm import tqdm
from PIL import Image

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
import psycopg2

# === LOCAL IMPORT ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from database.select_data import get_all_properties


# === CONFIGURATION ===
IMAGE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'downloaded_images'))
GOOGLE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1Pn1qXZ0B1MHb-KuYoHqcJBrz10Bh7PPV"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLUSTERS = 10


# === IMAGE HANDLING ===
def download_images_from_drive(folder_url: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"🔗 Downloading images from Google Drive folder: {folder_url}")
    try:
        gdown.download_folder(folder_url, output=output_dir, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"Error downloading images from Google Drive: {e}")


def verify_image_files_match_database(images_df: pd.DataFrame, image_folder: str):
    valid_pairs = {
        (str(row['property_id']), str(row['id']))
        for _, row in images_df.iterrows()
    }

    downloaded_pairs = set()
    for fname in os.listdir(image_folder):
        if fname.lower().endswith('.jpg') and '-' in fname:
            pid, imgid = fname[:-4].split('-', 1)
            downloaded_pairs.add((pid, imgid))

    missing = valid_pairs - downloaded_pairs
    extra = downloaded_pairs - valid_pairs

    print(f"✅ DB image entries: {len(valid_pairs)}")
    print(f"📂 Downloaded image files: {len(downloaded_pairs)}")

    if missing:
        print(f"⚠️ {len(missing)} images missing from folder:")
        for pid, imgid in sorted(missing)[:10]:
            print(f"  - Missing: {pid}-{imgid}.jpg")
    if extra:
        print(f"⚠️ {len(extra)} extra images not in DB:")
        for pid, imgid in sorted(extra)[:10]:
            print(f"  - Extra: {pid}-{imgid}.jpg")
    if not missing and not extra:
        print("✅ All images match database entries.")


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


def process_image(img_path: str, model) -> np.ndarray:
    try:
        image = Image.open(img_path).convert("RGB")
        image = image_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return model(image).cpu().numpy().flatten()
    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")
        return np.zeros(2048)


def extract_image_features(df: pd.DataFrame, id_col="id") -> pd.DataFrame:
    model = load_feature_extractor()
    vectors, ids = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            pid_raw = row[id_col]
            pid = str(int(round(float(pid_raw))))
        except Exception as e:
            print(f"❌ Invalid ID format: {row[id_col]} ({e})")
            continue
        
        matches = [f for f in os.listdir(IMAGE_FOLDER) if f.startswith(f"{pid}-") and f.endswith('.jpg')]
        if matches:
            path = os.path.join(IMAGE_FOLDER, matches[0])
            vec = process_image(path, model)
            vectors.append(vec)
            ids.append(pid)
        else:
            print(f"⚠️ No image found for property ID {pid}")

    if not vectors:
        print("❌ No image features extracted.")
        return pd.DataFrame(columns=[f"img_feat_{i}" for i in range(2048)] + [id_col])

    image_df = pd.DataFrame(vectors, index=ids)
    image_df.columns = [f"img_feat_{i}" for i in range(image_df.shape[1])]
    image_df[id_col] = image_df.index
    return image_df.reset_index(drop=True)

def reduce_image_features(img_df: pd.DataFrame, n_components=100, id_col="id") -> pd.DataFrame:
    features = img_df.drop(columns=[id_col])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(features)
    reduced_df = pd.DataFrame(reduced, columns=[f"img_pca_{i}" for i in range(n_components)])
    reduced_df[id_col] = img_df[id_col].values
    return reduced_df

# === TEXT CLUSTERING ===
def vectorize_descriptions(df: pd.DataFrame, max_features=1000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(df['description'].fillna(""))
    return X, vectorizer, vectorizer.get_feature_names_out()


def perform_kmeans(X, n_clusters=10, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    return model, model.fit_predict(X)


# === STRUCTURED FEATURES ===
def preprocess_structured_data(df: pd.DataFrame, is_rental_mode=True) -> pd.DataFrame:
    # Only filter by 'price' if present (for training, not prediction)
    if "price" in df.columns:
        df = df[df["price"].notnull()]
        if is_rental_mode:
            df = df[df["price"] < 10000]
        else:
            df = df[df["price"] > 80000]

    df["availability_date"] = pd.to_datetime(df.get("availability_date"), errors="coerce")
    df["availability_days_from_now"] = (df["availability_date"] - datetime.now()).dt.days
    df["has_balcony"] = df.get("has_balcony", False).astype(int)
    df["is_rental"] = df.get("is_rental", True).astype(int)

    df["log_area_sqm"] = df["area_sqm"].apply(lambda x: np.log1p(x))
    df["availability_soon"] = df["availability_days_from_now"].apply(lambda x: int(x is not None and x <= 30))
    df["is_small"] = (df["area_sqm"] < 30).astype(int)
    df["is_large"] = (df["area_sqm"] > 80).astype(int)
    df["lat_bin"] = pd.cut(df.get("latitude", pd.Series([np.nan]*len(df))), bins=10, labels=False)
    df["lon_bin"] = pd.cut(df.get("longitude", pd.Series([np.nan]*len(df))), bins=10, labels=False)
    df["is_ground_floor"] = (df["floor"] <= 0).astype(int)
    df["is_top_floor"] = (df["floor"] > 4).astype(int)

    for col in ["is_new", "has_view", "has_garden", "has_parking", "has_air_conditioning"]:
        df[col] = df.get(col, False).astype(int)

    # Augment rare categories to help balance the model
    for _ in range(20):
        for col in ["is_top_floor", "is_large", "is_small"]:
            df = augment_with_noise(df, condition_col=col, condition_val=1)

    return df


# === MODEL TRAINING ===
def train_model(X: pd.DataFrame, y: pd.Series, save_path=None):
    cat_features = [col for col in ["city", "region", "description_cluster"] if col in X.columns]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [100],
        "model__max_depth": [10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }

    grid = GridSearchCV(
        pipeline, param_grid, scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1, verbose=1
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    if save_path:
        import joblib
        joblib.dump(best, save_path)
        print(f"✅ Model saved to {save_path}")

    y_pred = best.predict(X_test)
    print(f"Best params: {grid.best_params_}")
    print(f"MAE: CHF {mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.2f}")
    return best


# === MAIN ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-images", action="store_true", help="Download images and exit")
    args = parser.parse_args()

    if args.download_images:
        download_images_from_drive(GOOGLE_DRIVE_FOLDER_URL, IMAGE_FOLDER)
        print("✅ Images downloaded.")
        sys.exit(0)

    if not os.path.exists(IMAGE_FOLDER) or not os.listdir(IMAGE_FOLDER):
        download_images_from_drive(GOOGLE_DRIVE_FOLDER_URL, IMAGE_FOLDER)

    print("📦 Loading property data...")
    df = pd.DataFrame(get_all_properties())

    script_dir = os.path.dirname(os.path.abspath(__file__))
    for mode, mode_name, model_filename in [
        (True, "rental", "combined_price_model_rental.joblib"),
        (False, "purchase", "combined_price_model_purchase.joblib")
    ]:
        print(f"\n=== Training {mode_name} model ===")
        df_mode = preprocess_structured_data(df.copy(), is_rental_mode=mode)
        print(f"Shape after preprocessing for {mode_name}: {df_mode.shape}")
        if df_mode.shape[0] == 0:
            print(f"❌ No data available for {mode_name} mode after filtering. Skipping model training.")
            continue

        print("🧠 Generating description clusters...")
        X_text, _, _ = vectorize_descriptions(df_mode)
        _, clusters = perform_kmeans(X_text, n_clusters=N_CLUSTERS)
        df_mode["description_cluster"] = clusters

        print("🖼️ Verifying image matches...")
        conn = psycopg2.connect(
            dbname="real_estate_price_predictor",
            user="postgres", password="postgres",
            host="localhost", port=5433
        )
        images_df = pd.read_sql("SELECT id, property_id FROM images", conn)
        conn.close()
        verify_image_files_match_database(images_df, IMAGE_FOLDER)

        print("🖼️ Extracting image features...")
        img_df = extract_image_features(df_mode)
        print(f"Shape of image features for {mode_name}: {img_df.shape}")

        # ✅ Korrekte ID-Vorbereitung für sauberes Merge
        df_mode["id"] = df_mode["id"].astype(float).round().astype(int).astype(str)
        img_df["id"] = img_df["id"].astype(str)

        img_df = reduce_image_features(img_df, n_components=100)

        print("🔗 Merging all features...")
        before_merge = df_mode.shape[0]
        full_df = pd.merge(df_mode, img_df, on="id", how="inner")
        after_merge = full_df.shape[0]
        print(f"Shape after merging for {mode_name}: {full_df.shape}")
        print(f"ℹ️ {before_merge - after_merge} entries lost during merge due to missing images.")

        if full_df.shape[0] == 0:
            print(f"❌ No data available for {mode_name} mode after merging with image features. Skipping model training.")
            continue

        X = full_df.drop(columns=[
            "price", "id", "title", "address", "availability_date", "description"
        ])
        y = full_df["price"]

        print(f"🚀 Training {mode_name} model...")
        save_path = os.path.join(script_dir, model_filename)
        train_model(X, y, save_path=save_path)
        print(f"✅ {mode_name.capitalize()} model saved as {save_path}")
