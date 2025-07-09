from typing import Tuple, Any
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from database.select_data import get_all_properties
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Optional
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
        self._visualize_predictions(X_train, X_test, y_train, y_test)

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

    def _visualize_predictions(self, X_train, X_test, y_train, y_test) -> None:
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

if __name__ == "__main__":
    # Load data
    df = PropertyDataLoader().load_data()

    # Initialize and train the regression model
    regression_model = RegressionModel()
    regression_model.train(df)

    print("Now enter new property data for prediction:")

    # Example input for prediction (replace with actual input logic)
    new_property_data = {
        "city": "Zurich",
        "region": "Zürich",
        "zip_code": 8001,
        "rooms": 3.5,
        "floor": 2,
        "area_sqm": 80,
        "has_balcony": True,
        "is_rental": True,
        "availability_days_from_now": 30,
        "latitude": 47.3769,
        "longitude": 8.5417
    }

    # Convert to DataFrame for prediction
    new_df = pd.DataFrame([new_property_data])
    property_loader = PropertyDataLoader(new_df)
    processed_df = property_loader.load_data()  # Get the processed DataFrame

    # Predict price using the processed DataFrame
    predicted_price = regression_model.model.predict(processed_df)
    print(f"Predicted Price: CHF {int(predicted_price[0]):,}")