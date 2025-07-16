import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modelling', 'random_forest')))


import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Robust import of load_data from random_forest_regression_model.py
import importlib.util
# Go up two directories to project root, then into modelling/random_forest
import_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'modelling', 'random_forest', 'random_forest_regression_model.py'))
spec = importlib.util.spec_from_file_location("random_forest_regression_model", import_path)
rf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf_module)
load_data = rf_module.load_data


def evaluate_model(model_path, is_rental=True):
    # Load model
    print(f"📦 Loading model from {model_path}")
    model = joblib.load(model_path)
    
    plot_name = "rental" if is_rental else "purchase"

    # Load and prepare data
    df = load_data(is_rental_mode=is_rental)
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Evaluation Metrics:")
    print(f"RMSE: CHF {rmse:.2f}")
    print(f"MAE : CHF {mae:.2f}")
    print(f"R²  : {r2:.4f}")


    # Plot: Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "--", color="gray")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("📈 Actual vs. Predicted Prices")
    plt.grid(True)
    plt.tight_layout()
    plot_path1 = os.path.join(os.path.dirname(__file__), plot_name + "_actual_vs_predicted.png")
    plt.savefig(plot_path1)
    plt.close()


    # Plot: Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, edgecolor="black")
    plt.title("📉 Residuals Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.grid(True)
    plt.tight_layout()
    plot_path2 = os.path.join(os.path.dirname(__file__), plot_name + "_residuals_distribution.png")
    plt.savefig(plot_path2)
    plt.close()

    # Feature importances (only for tree-based models like RandomForest)
    try:
        regressor = model.named_steps["regressor"]
        importances = regressor.feature_importances_
        preprocessor = model.named_steps["preprocessing"]
        cat_features = preprocessor.transformers_[0][2]
        onehot_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features)
        passthrough_len = len(importances) - len(onehot_names)
        passthrough_names = [f"feature_{i}" for i in range(passthrough_len)]
        feature_names = list(onehot_names) + passthrough_names

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"][:15][::-1], importance_df["Importance"][:15][::-1])
        plt.xlabel("Importance")
        plt.title("🔍 Top 15 Feature Importances")
        plt.tight_layout()
        plot_path3 = os.path.join(os.path.dirname(__file__), plot_name + "_feature_importances.png")
        plt.savefig(plot_path3)
        plt.close()

    except Exception as e:
        print("⚠️ Feature importances could not be extracted:", e)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two directories to project root, then into modelling/random_forest
    model_file_rental = os.path.abspath(os.path.join(script_dir, '..', '..', 'modelling', 'random_forest', 'random_forest_model_rental.joblib'))
    model_file_purchase = os.path.abspath(os.path.join(script_dir, '..', '..', 'modelling', 'random_forest', 'random_forest_model_purchase.joblib'))
    evaluate_model(model_file_purchase, False)
    evaluate_model(model_file_rental, True)
