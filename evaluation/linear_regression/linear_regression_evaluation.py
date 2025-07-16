import importlib
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Local import of your load_data function
import_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'modelling', 'random_forest', 'random_forest_regression_model.py'))
spec = importlib.util.spec_from_file_location("random_forest_regression_model", import_path)
rf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf_module)
load_data = rf_module.load_data

def evaluate_linear_model(model_path, is_rental=True):
    model_type = "rental" if is_rental else "purchase"
    print(f"📦 Loading Linear Regression model for {model_type} from: {model_path}")

    model = joblib.load(model_path)

    # Load and prepare data
    df = load_data(is_rental)
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 Evaluation Results:")
    print(f"→ RMSE : CHF {rmse:.2f}")
    print(f"→ MAE  : CHF {mae:.2f}")
    print(f"→ R²   : {r2:.4f}")

    # Save plots in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "--", color="gray")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("📈 Actual vs. Predicted Prices")
    plt.grid(True)
    plt.tight_layout()
    plot1_path = os.path.join(script_dir, f"{model_type}_actual_vs_predicted.png")
    plt.savefig(plot1_path)
    plt.close()

    # Plot 2: Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, edgecolor="black")
    plt.title("📉 Residuals Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.grid(True)
    plt.tight_layout()
    plot2_path = os.path.join(script_dir, f"{model_type}_residuals_distribution.png")
    plt.savefig(plot2_path)
    plt.close()

    # Plot 3: Feature Importances (Linear Regression Coefficients)
    try:
        regressor = model.named_steps["regressor"]
        preprocessor = model.named_steps["preprocessor"]
        categorical = preprocessor.transformers_[0][2]
        onehot = preprocessor.named_transformers_["cat"]
        onehot_feature_names = onehot.get_feature_names_out(categorical)
        
        # Combine with passthrough features
        num_passthrough = len(regressor.coef_) - len(onehot_feature_names)
        passthrough_names = [f"feature_{i}" for i in range(num_passthrough)]
        feature_names = np.concatenate([onehot_feature_names, passthrough_names])

        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": regressor.coef_
        }).sort_values("Coefficient", key=abs, ascending=False)

        # Plot top 15 by absolute value
        top_features = coef_df.head(15)
        plt.figure(figsize=(10, 6))
        plt.barh(top_features["Feature"][::-1], top_features["Coefficient"][::-1])
        plt.xlabel("Coefficient")
        plt.title("📌 Top 15 Feature Importances (Linear Coefficients)")
        plt.tight_layout()
        plt.savefig(os.path.join(script_dir, f"{model_type}_feature_importances.png"))
        plt.close()

    except Exception as e:
        print("⚠️ Could not extract feature importances:", e)

if __name__ == "__main__":
    # Model files are stored in the same directory as the training script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_dir = os.path.join(project_root, 'modelling', 'linear regression')
    model_rental_path = os.path.join(model_dir, "linear_model_rental.pkl")
    model_purchase_path = os.path.join(model_dir, "linear_model_purchase.pkl")

    evaluate_linear_model(model_purchase_path, is_rental=False)
    evaluate_linear_model(model_rental_path, is_rental=True)
