# Standard library
from datetime import datetime
from typing import Tuple, Any, List, Optional
import os

# Third-party libraries
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Project-specific
from database.select_data import get_all_properties

# Constants
RANDOM_SEED = 0x0
PRICE_MIN_THRESHOLD = 500
PRICE_MAX_THRESHOLD = 10000
TEST_SIZE = 0.2
CV_FOLDS = 5
FIGURE_SIZE = (10, 5)
DPI = 100


class RegressionPreprocessor:
    """Handles data preprocessing for regression analysis."""

    @staticmethod
    def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable from the input DataFrame.

        Args:
            df: Input DataFrame containing property data

        Returns:
            Tuple containing features (X) and target variable (y)
        """
        # Filter out unreasonable prices
        df = df[(df['price'] >= PRICE_MIN_THRESHOLD) & (df['price'] <= PRICE_MAX_THRESHOLD)]

        X = df.drop(columns=["price"])
        y = df["price"]

        return X, y

    @staticmethod
    def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
        """
        Create a preprocessor for categorical and numerical features.

        Args:
            X: Feature DataFrame

        Returns:
            ColumnTransformer for data preprocessing
        """
        categorical_features = [col for col in ["city", "region"] if col in X.columns]
        numerical_features = [col for col in X.columns if col not in categorical_features]

        return ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features), ("num", "passthrough", numerical_features)], remainder="drop"
            )


def format_currency(x: float, pos: Any) -> str:
    """Format number as currency string."""
    return f"CHF {int(x):,}"


class RegressionModel:
    """Handles linear regression modeling and evaluation."""

    def __init__(self):
        self.model = None
        self.preprocessor = None

    def train(self, df: pd.DataFrame) -> None:
        """
        Train the regression model.

        Args:
            df: Input DataFrame containing property data
        """
        X, y = RegressionPreprocessor.prepare_features(df)
        self.preprocessor = RegressionPreprocessor.create_preprocessor(X)

        pipeline = Pipeline(
            [("preprocessing", self.preprocessor), ("regressor", LinearRegression())]
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
            )

        # Train model and calculate metrics
        self.model = pipeline.fit(X_train, y_train)
        self._evaluate_model(X_train, X_test, y_train, y_test)


    def _evaluate_model(self, X_train, X_test, y_train, y_test) -> None:
        """Calculate and print model evaluation metrics."""
        cv_scores = cross_val_score(
            self.model, X_train, y_train, cv=CV_FOLDS, scoring='neg_mean_squared_error'
            )
        cv_rmse = np.sqrt(-cv_scores.mean())

        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Model Evaluation Metrics:")
        print(f"Cross-validation RMSE: {cv_rmse:.2f} CHF")
        print(f"Test RMSE: {rmse:.2f} CHF")
        print(f"R²: {r2:.2f}")

    def visualize_predictions(self, X_train, X_test, y_train, y_test) -> None:
        """Create visualization of model predictions."""
        y_hat_train = self.model.predict(X_train)
        y_hat_test = self.model.predict(X_test)

        plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
        plt.scatter(y_train, y_hat_train, color='blue', label='Train', alpha=0.5)
        plt.scatter(y_test, y_hat_test, color='red', label='Test', alpha=0.5)

        y_range = [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())]
        plt.plot(
            y_range, y_range, color='black', linestyle='--', label='Ideal Prediction'
            )

        plt.gca().xaxis.set_major_formatter(FuncFormatter(format_currency))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_currency))
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Linear Regression Predictions')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_model(self, model, model_name) -> None:
        """Save the trained model and preprocessor to disk."""
        save_name = f"{model_name}.pkl"
        joblib.dump(model, save_name)
        joblib.dump(self.preprocessor, f"{model_name}_preprocessor.pkl")
        # Verify save file
        if not (os.path.exists(save_name) and os.path.exists(f"{model_name}_preprocessor.pkl")):
            raise FileNotFoundError("Model or preprocessor file not found after saving.")
        print("Model and preprocessor saved successfully.")




class PropertyDataLoader:
    """Handles loading and preprocessing of property data."""

    # Constants
    PRICE_THRESHOLD = 6500
    SMALL_AREA_THRESHOLD = 30
    LARGE_AREA_THRESHOLD = 80
    GROUND_FLOOR_THRESHOLD = 0
    TOP_FLOOR_THRESHOLD = 4
    AVAILABILITY_THRESHOLD_DAYS = 14
    LOCATION_BINS = 10

    CAT_FIELDS = ["city", "region"]

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """Initialize the PropertyDataLoader."""
        self.df: Optional[pd.DataFrame] = df

    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess property data.

        Returns:
            pd.DataFrame: Processed property data
        """
        if self.df is None:  # Changed from "if not self.df"
            properties = self._get_raw_data()
            self.df = pd.DataFrame(properties)

        self._clean_data()
        self._process_availability()
        self._process_boolean_fields()
        self._filter_and_drop_fields()
        self._create_derived_features()

        return self.df

    def _get_raw_data(self) -> List[dict]:
        """Fetch raw property data from the source."""
        properties = get_all_properties()
        if not properties:
            raise ValueError("No property data found.")
        return properties

    def _clean_data(self) -> None:
        """Clean the raw data by removing null prices."""
        if "price" in self.df.columns:  # Only clean price if it exists
            self.df = self.df[self.df["price"].notnull()]
            self.df = self.df[self.df["price"] < self.PRICE_THRESHOLD]

    def _process_availability(self) -> None:
        """Process availability dates and calculate days from now."""
        if "availability_date" in self.df.columns:
            self.df["availability_date"] = pd.to_datetime(
                self.df["availability_date"], errors="coerce"
                )
            # Set a default value of 0 for NaN values
            self.df["availability_days_from_now"] = (
                (self.df["availability_date"] - pd.Timestamp.now()).dt.days
            ).fillna(0)
        else:
            # Initialize with default value if column doesn't exist
            self.df["availability_days_from_now"] = 0

    def _process_boolean_fields(self) -> None:
        """Convert boolean fields to integers."""
        self.df["has_balcony"] = self.df.get("has_balcony", False).astype(int)
        self.df["is_rental"] = self.df.get("is_rental", True).astype(int)

    def _filter_and_drop_fields(self) -> None:
        """Remove unnecessary columns."""
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns
        columns_to_keep = list(numerical_columns) + self.CAT_FIELDS
        # Remove id column if it exists to avoid model dependency on it
        if 'id' in columns_to_keep:
            columns_to_keep.remove('id')
        self.df = self.df.loc[:, columns_to_keep]

    def _create_derived_features(self) -> None:
        """Create derived features from existing data."""
        # Area-based features
        if "area_sqm" in self.df.columns:
            self.df["log_area_sqm"] = self.df["area_sqm"].apply(np.log1p)
            self.df["is_small"] = (self.df["area_sqm"] < self.SMALL_AREA_THRESHOLD).astype(int)
            self.df["is_large"] = (self.df["area_sqm"] > self.LARGE_AREA_THRESHOLD).astype(int)

        # Location features
        if all(col in self.df.columns for col in ["latitude", "longitude"]):
            self.df["lat_bin"] = pd.cut(self.df["latitude"], bins=self.LOCATION_BINS, labels=False)
            self.df["lon_bin"] = pd.cut(self.df["longitude"], bins=self.LOCATION_BINS, labels=False)

        # Floor-related features
        if "floor" in self.df.columns:
            self.df["is_ground_floor"] = (self.df["floor"] <= self.GROUND_FLOOR_THRESHOLD).astype(int)
            self.df["is_top_floor"] = (self.df["floor"] > self.TOP_FLOOR_THRESHOLD).astype(int)

        # Availability feature
        self.df["availability_soon"] = (
            self.df["availability_days_from_now"].apply(
                lambda x: int(x <= self.AVAILABILITY_THRESHOLD_DAYS)
            )
        )

def request_input():
    """
    Prompts the user for property information and converts the input into a structured
    DataFrame object suitable for further processing or analysis.

    This function collects various details about a property through a series of input prompts,
    validates basic types through input casting, and organizes them into a single structured
    pandas DataFrame. The captured information includes geographic, spatial, and usage data.

    Returns:
        pandas.DataFrame: A DataFrame containing the structured property information from user input.
    """
    try:
        city = input("City: ").strip()

        if city and city.isalpha():
            print(f"Valid city: {city}")
        else:
            print("Invalid input: please enter a city name with letters only.")
        region = input("Region: ").strip()
        if region and region.isalpha() and len(region) == 2:
            print(f"Valid region: {region}")
        else:
            print("Invalid input: please enter a region name with two letters only.")
        zip_code = int(input("Zip Code: "))
        if 1000 <= zip_code <= 9999:
            print(f"Valid zip code: {zip_code}")
        else:
            print("Invalid input: please enter a zip code with four digits only.")
        rooms = float(input("Number of Rooms: "))
        if rooms > 0:
            print(f"Valid number of rooms: {rooms}")
        else:
            print("Invalid input: please enter a positive number for rooms.")
        floor = int(input("Floor: "))
        if floor >= 0:
            print(f"Valid floor: {floor}")
        else:
            print("Invalid input: please enter a non-negative integer for floor.")
        area_sqm = float(input("Area in sqm: "))
        if area_sqm > 0:
            print(f"Valid area: {area_sqm} sqm")
        else:
            print("Invalid input: please enter a positive number for area in sqm.")
        has_balcony = input("Has Balcony (yes/no): ").strip().lower() == 'yes'
        valid_input_balcony = ["yes", "no"]
        if has_balcony in valid_input_balcony:
            print("Balcony: Valid input")
        else:
            print("Invalid input: please enter 'yes' or 'no' for balcony.")
        is_rental = input("Is Rental (yes/no): ").strip().lower() == 'yes'
        valid_input_rental = ["yes", "no"]
        if is_rental in valid_input_rental:
            print("Rental: Valid input")
        else:
            print("Invalid input: please enter 'yes' or 'no' for rental.")
        availability_days_from_now = int(input("Availability Days from Now: "))
        if availability_days_from_now >= 0:
            print(f"Valid availability days: {availability_days_from_now}")
        else:
            print("Invalid input: please enter a non-negative integer for availability days.")
        latitude = float(input("Latitude: "))
        if -90 <= latitude <= 90:
            print(f"Valid latitude: {latitude}")
        else:
            print("Invalid input: please enter a latitude between -90 and 90 degrees.")
        longitude = float(input("Longitude: "))
        if -180 <= longitude <= 180:
            print(f"Valid longitude: {longitude}")
        else:
            print("Invalid input: please enter a longitude between -180 and 180 degrees.")
    except ValueError:
        print("Invalid input. Please enter the correct data types.")
    try:
        new_property_data = {
            "city": city,
            "region": region,
            "zip_code": zip_code,
            "rooms": rooms,
            "floor": floor,
            "area_sqm": area_sqm,
            "has_balcony": has_balcony,
            "is_rental": is_rental,
            "availability_days_from_now": availability_days_from_now,
            "latitude": latitude,
            "longitude": longitude
            }
        new_df = pd.DataFrame([new_property_data])
        return new_df
    except ValueError:
        print("Error creating DataFrame from input data. Please check the input values.")
        return pd.DataFrame()
if __name__ == "__main__":
    model_name = "linear_regression_model"
    # Load data
    df = PropertyDataLoader().load_data()

    # Initialize and train the regression model
    # If model exists, it will be loaded; otherwise, a new model will be trained
    if os.path.exists(f"{model_name}.pkl") and os.path.exists(f"{model_name}_preprocessor.pkl"):
        print("Loading existing model...")
        regression_model = joblib.load(f"{model_name}.pkl")
        regression_model.preprocessor = joblib.load(f"{model_name}_preprocessor.pkl")
    else:
        print("Training new model...")
        regression_model = RegressionModel()
        regression_model.train(df)
        regression_model.save_model(regression_model, "linear_regression_model")
    print("Now enter new property data for prediction:")

    properties_df = request_input()

    property_loader = PropertyDataLoader(properties_df)
    processed_df = property_loader.load_data()  # Get the processed DataFrame

    # Predict price using the processed DataFrame
    predicted_price = regression_model.model.predict(processed_df)
    print("\033[91mPredicted Price: CHF \033[0m", f"\033[92m{predicted_price[0]:.0f}\033[0m")
