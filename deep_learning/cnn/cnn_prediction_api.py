from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import torch
from PIL import Image
from torchvision import transforms
from cnn_training import preprocess_structured_data, load_feature_extractor

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths - use absolute paths
MODEL_PATHS = {
    "rental": os.path.join(SCRIPT_DIR, "combined_price_model_rental.joblib"),
    "purchase": os.path.join(SCRIPT_DIR, "combined_price_model_purchase.joblib")
}

# Image processing
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_image_features(image_file):
    """Extract features from a single image file"""
    model = load_feature_extractor()
    image = Image.open(image_file).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model(image).cpu().numpy().flatten()
    return features

def extract_multiple_image_features(image_files):
    """Extract and average features from multiple image files"""
    if not image_files:
        return np.zeros(2048)
    
    model = load_feature_extractor()
    all_features = []
    
    for image_file in image_files:
        try:
            image = Image.open(image_file).convert("RGB")
            image = image_transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                features = model(image).cpu().numpy().flatten()
                all_features.append(features)
        except Exception as e:
            print(f"Error processing image: {e}")
            continue
    
    if not all_features:
        return np.zeros(2048)
    
    # Average all image features
    averaged_features = np.mean(all_features, axis=0)
    return averaged_features

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Debug: Print incoming request data
        print("=== Incoming Request Debug ===")
        print("Form data keys:", list(request.form.keys()))
        print("Files:", list(request.files.keys()))
        print("Form data:", dict(request.form))
        
        # Parse JSON data
        data = request.form.to_dict()
        
        # Validate required fields
        required_fields = ["mode", "area", "floor", "rooms", "lat", "lon", "zip_code", "city", "region"]
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            print("ERROR:", error_msg)
            return jsonify({"error": error_msg}), 400
        
        features = {
            "zip_code": data["zip_code"],
            "area_sqm": float(data["area"]),
            "floor": int(data["floor"]),
            "rooms": float(data["rooms"]),
            "latitude": float(data["lat"]),
            "longitude": float(data["lon"]),
            "city": data["city"],
            "region": data["region"],
            "description_cluster": int(data.get("description_cluster", 0)),
            "has_balcony": int(data.get("has_balcony", 0)),
            "has_view": int(data.get("has_view", 0)),
            "has_garden": int(data.get("has_garden", 0)),
            "has_parking": int(data.get("has_parking", 0)),
            "has_air_conditioning": int(data.get("has_air_conditioning", 0)),
            "is_new": int(data.get("is_new", 0)),
            "availability_date": data.get("availability_date", "2025-01-01"),
            "is_rental": 1 if data["mode"] == "rental" else 0,
            "id": "user_input"
        }
        
        print("Parsed features:", features)

        df = pd.DataFrame([features])
        print("DataFrame before preprocessing:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample values: {df.iloc[0].to_dict()}")
        
        # Use the mode directly instead of the is_rental field
        is_rental_mode = (data["mode"] == "rental")
        df = preprocess_structured_data(df.copy(), is_rental_mode=is_rental_mode)
        
        print(f"Processing in {'rental' if is_rental_mode else 'purchase'} mode")
        print("DataFrame after preprocessing:")
        print(f"Shape: {df.shape}")
        print(f"New columns: {[col for col in df.columns if col not in features.keys()]}")

        # Image feature handling - support multiple images
        image_files = request.files.getlist("images")  # Get multiple images
        if not image_files:
            # Fallback to single image for backward compatibility
            single_image = request.files.get("image")
            if single_image:
                image_files = [single_image]
        
        if image_files:
            img_features = extract_multiple_image_features(image_files)
        else:
            img_features = np.zeros(2048)

        # Apply PCA to image features to match training (100 components)
        # For now, we'll use a simple dimensionality reduction by taking every 20th feature
        # This is a quick fix - ideally you'd load the actual PCA transformer used during training
        img_pca_features = img_features[::20][:100]  # Take every 20th feature, limit to 100
        
        # If we don't have enough features, pad with zeros
        if len(img_pca_features) < 100:
            img_pca_features = np.pad(img_pca_features, (0, 100 - len(img_pca_features)), 'constant')

        # Create PCA image features DataFrame to match model expectations
        img_features_dict = {f"img_pca_{i}": [img_pca_features[i]] for i in range(100)}
        img_df = pd.DataFrame(img_features_dict)
        
        print(f"Original image features shape: {img_features.shape}")
        print(f"PCA features shape: {img_pca_features.shape}")
        print(f"Image DataFrame shape: {img_df.shape}")
        
        # Concatenate all features at once for better performance
        df = pd.concat([df, img_df], axis=1)

        # Remove unused columns (but keep is_rental as the model expects it)
        df.drop(columns=["id", "availability_date"], inplace=True, errors="ignore")

        print("Final DataFrame shape:", df.shape)
        print("DataFrame columns:", list(df.columns)[:10], "... (showing first 10)")
        print(f"All columns: {list(df.columns)}")
        print(f"is_rental value: {df['is_rental'].iloc[0] if 'is_rental' in df.columns else 'NOT FOUND'}")

        model_path = MODEL_PATHS[data["mode"]]
        print(f"Loading model from: {model_path}")
        print(f"Mode: {data['mode']}")
        print(f"Model file exists: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        print(f"Model loaded successfully for {data['mode']} mode")
        print(f"Model type: {type(model)}")
        
        # Debug: show some sample data going into the model
        print("Sample of features going to model:")
        print("Feature columns:", list(df.columns))
        print("Shape:", df.shape)
        print("First few values:")
        for col in df.columns[:10]:  # Show first 10 columns
            print(f"  {col}: {df[col].iloc[0]}")
        
        # Check if all values are the same (indicating a problem)
        print("\nChecking for variance in features:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:
            val = df[col].iloc[0]
            print(f"  {col}: {val}")
        
        # Check some key features that should vary
        print("\nKey feature values:")
        key_features = ['area_sqm', 'rooms', 'floor', 'is_rental', 'is_large', 'is_small']
        for feat in key_features:
            if feat in df.columns:
                print(f"  {feat}: {df[feat].iloc[0]}")
        
        prediction = model.predict(df)
        price = float(prediction[0])
        
        print(f"Raw prediction: {prediction}")
        print("Prediction successful:", price)
        return jsonify({"prediction": round(price, 2)})

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print("EXCEPTION:", error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({"error": error_msg}), 400

if __name__ == "__main__":
    # Verify model files exist before starting the server
    print("=== Model File Check ===")
    for mode, path in MODEL_PATHS.items():
        if os.path.exists(path):
            print(f"✓ {mode} model found: {path}")
        else:
            print(f"✗ {mode} model NOT found: {path}")
    
    print(f"Script directory: {SCRIPT_DIR}")
    print("=== Starting Flask Server ===")
    app.run(debug=True)
