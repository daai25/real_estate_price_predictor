import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# === Robust Import of load_data ===
import importlib.util

script_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'modelling', 'random_forest', 'random_forest_regression_model.py'))

spec = importlib.util.spec_from_file_location("random_forest_regression_model", import_path)
rf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf_module)
load_data = rf_module.load_data


def evaluate_xgb(model_path_json, is_rental=True):
    # Load XGBoost model
    model = xgb.XGBRegressor()
    print(f"📦 Loading model from {model_path_json}")
    model.load_model(model_path_json)

    plot_name = "rental" if is_rental else "purchase"

    # Load preprocessor
    print("🔄 Loading encoders (kmeans + ohe)...")
    model_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'modelling', 'xgboost'))
    with open(os.path.join(model_dir, 'kmeans.pkl'), 'rb') as f:
        kmeans = pickle.load(f)
    with open(os.path.join(model_dir, 'ohe.pkl'), 'rb') as f:
        ohe = pickle.load(f)

    # Load and prepare data
    df = load_data(is_rental_mode=is_rental)
    X = df.drop(columns=["price"])
    y = df["price"]

    # Apply same preprocessing
    X['room_size'] = X['area_sqm'] / X['rooms']
    X['price_per_sqm'] = y / X['area_sqm']
    X['geo_cluster'] = kmeans.predict(X[['longitude', 'latitude']].values)

    cat_cols = ['zip_code', 'city', 'region', 'geo_cluster']
    X_cat = ohe.transform(X[cat_cols])
    X_num = X.drop(columns=cat_cols).copy()

    # Align features to match training
    feature_names_file = os.path.join(model_dir, 'xgboost_miete_feature_names.pkl' if is_rental else 'xgboost_kauf_feature_names.pkl')
    with open(feature_names_file, 'rb') as f:
        feature_names = pickle.load(f)

    # Convert to DataFrame for reindexing
    X_final_df = pd.DataFrame(np.hstack([X_num.values, X_cat]), columns=list(X_num.columns) + list(ohe.get_feature_names_out(cat_cols)))
    X_final_df = X_final_df.reindex(columns=feature_names, fill_value=0)
    X_train, X_test, y_train, y_test = train_test_split(X_final_df.values, y, test_size=0.2, random_state=42)

    # Predict
    log_pred = model.predict(X_test)
    y_pred = np.expm1(log_pred)
    y_true = np.expm1(np.log1p(y_test))

    # Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n📊 Evaluation Metrics ({plot_name.title()}):")
    print(f"RMSE: CHF {rmse:,.2f}")
    print(f"MAE : CHF {mae:,.2f}")
    print(f"R²  : {r2:.4f}")

    # Plot: Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "--", color="gray")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"📈 Actual vs. Predicted Prices ({plot_name.title()})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f"{plot_name}_actual_vs_predicted.png"))
    plt.close()

    # Plot: Residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, edgecolor="black")
    plt.title(f"📉 Residuals Distribution ({plot_name.title()})")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f"{plot_name}_residuals_distribution.png"))
    plt.close()

    # Feature importances (only if feature names are available)
    try:
        xgb.plot_importance(model, max_num_features=15)
        plt.title(f"🔍 Top Feature Importances ({plot_name.title()})")
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, f"{plot_name}_feature_importances.png"))
        plt.close()
    except Exception as e:
        print("⚠️ Feature importances could not be plotted:", e)


if __name__ == "__main__":
    # Model files are stored in the modelling/xgboost/ directory
    model_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'modelling', 'xgboost'))
    model_file_rental = os.path.join(model_dir, 'xgboost_miete_model.json')
    model_file_purchase = os.path.join(model_dir, 'xgboost_kauf_model.json')

    evaluate_xgb(model_file_purchase, is_rental=False)
    evaluate_xgb(model_file_rental, is_rental=True)
