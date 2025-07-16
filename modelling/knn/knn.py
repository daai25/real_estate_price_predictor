import sqlite3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

# Machine Learning
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import seaborn as sns

# Project-specific
from database.select_data import get_all_properties, update_property_category

# === Configuration ===
K = 5
STD_THRESHOLD = 1.0
ID_COLUMN = "id"
TABLE_NAME = "properties"
CATEGORY_COLUMN = "category_flag"

# === Feature Definitions ===
numerical_features = ['price', 'rooms', 'area_sqm', 'floor', 'latitude', 'longitude']
boolean_features = ['has_balcony']
categorical_features = ['address', 'zip_code', 'city', 'region']
FEATURE_COLUMNS = numerical_features + boolean_features

# === Data Loader ===
class PropertyDataLoader:
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df

    def load_data(self) -> pd.DataFrame:
        if self.df is None:
            properties = get_all_properties()
            if not properties:
                raise ValueError("No property data found.")
            self.df = pd.DataFrame(properties)
        return self.df

# === Preprocessing Function ===
def preprocess_real_estate_data(df: pd.DataFrame) -> np.ndarray:
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('bool', 'passthrough', boolean_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    return preprocessor.fit_transform(df)

# === Load and Prepare Data ===
data = PropertyDataLoader()
df_data = data.load_data()
X = preprocess_real_estate_data(df_data)

# === Unsupervised KNN for Category Flagging ===
knn = NearestNeighbors(n_neighbors=K)
knn.fit(X)
distances, indices = knn.kneighbors(X)

output: Dict[str, Dict[str, bool]] = {}
category_flags = []
mean_distances = []
similarity_counts = []

for idx, row in df_data.iterrows():
    prop_id = row[ID_COLUMN]
    neighbor_idxs = indices[idx]
    neighbor_values = df_data.iloc[neighbor_idxs][FEATURE_COLUMNS]

    feature_flags = {}
    similarity_count = 0

    for feature in FEATURE_COLUMNS:
        val = row[feature]
        neighbor_mean = neighbor_values[feature].mean()
        neighbor_std = neighbor_values[feature].std()

        if neighbor_std == 0:
            is_similar = val == neighbor_mean
        else:
            is_similar = abs(val - neighbor_mean) <= STD_THRESHOLD * neighbor_std

        feature_flags[feature] = is_similar
        similarity_count += int(is_similar)

    output[prop_id] = feature_flags
    category_flags.append((all(feature_flags.values()), prop_id))
    mean_distances.append(distances[idx].mean())
    similarity_counts.append(similarity_count)

# Append additional metrics to dataframe
df_data["mean_knn_distance"] = mean_distances
df_data["num_similar_features"] = similarity_counts
df_data[CATEGORY_COLUMN] = [flag[0] for flag in category_flags]

y = df_data[CATEGORY_COLUMN].values

# === Train/Test Split ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# === Supervised KNN Evaluation ===
accuracies = []
k_values = range(1, 21)

for k in k_values:
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# === Graphs ===
# Accuracy vs K
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN Model Performance vs. Number of Neighbors')
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_vs_k.png")
plt.show()

# Confusion Matrix
y_pred_final = model.predict(scaler.transform(X_test))
cm = confusion_matrix(y_test, y_pred_final)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# PCA Plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=20)
plt.title("PCA Projection of Properties by Category")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_projection.png")
plt.show()

# Feature Distribution: Histograms
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_data[feature], kde=True)
    plt.title(f"Histogram: {feature}")
    plt.tight_layout()
    plt.savefig(f"hist_{feature}.png")
    plt.close()

# Feature Distribution: Boxplots
for feature in numerical_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_data[feature])
    plt.title(f"Boxplot: {feature}")
    plt.tight_layout()
    plt.savefig(f"box_{feature}.png")
    plt.close()

# Spatial Distribution
plt.figure(figsize=(8, 6))
plt.scatter(df_data['longitude'], df_data['latitude'], c=y, cmap='coolwarm', alpha=0.6)
plt.title("Spatial Distribution by Category")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("spatial_distribution.png")
plt.show()

# Elbow Method for Clustering
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.tight_layout()
plt.savefig("elbow_plot.png")
plt.show()

# === Export ===
df_data.to_csv("property_classification_output.csv", index=False)
print("Exported results to property_classification_output.csv")
# === Optional: Update SQLite ===
# try:
#     conn = sqlite3.connect("your_database_file.db")  # Replace with actual path
#     cursor = conn.cursor()
#     for _, row in df_data.iterrows():
#         cursor.execute(f"UPDATE {TABLE_NAME} SET {CATEGORY_COLUMN} = ? WHERE {ID_COLUMN} = ?", (int(row[CATEGORY_COLUMN]), row[ID_COLUMN]))
#     conn.commit()
#     conn.close()
#     print("Database updated with category flags.")
# except Exception as e:
#     print("Failed to update database:", e)