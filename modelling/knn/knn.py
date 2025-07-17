import sqlite3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from deep_translator import GoogleTranslator

# Machine Learning
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns

# Project-specific
from database.select_data import get_all_properties

def translate_terms(terms: List[str], source_lang: str = "auto", target_lang: str = "en") -> List[str]:
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    return [translator.translate(term) for term in terms]

# === Configuration ===
K = 5
ID_COLUMN = "id"
TABLE_NAME = "properties"
CATEGORY_COLUMN = "description_cluster"

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

# === Load and Preprocess ===
data = PropertyDataLoader()
df_data = data.load_data()

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df_data['description'])

# === KMeans Clustering for Keyword-Based Categories ===
cluster_range = range(2, 30)  # Try clusters from 2 to 10

terms = vectorizer.get_feature_names_out()

for n_clusters in cluster_range:
    print(f"\n=== KMeans with {n_clusters} Clusters ===")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 2 * n_clusters), squeeze=False)
    axes = axes.flatten()  # Convert 2D array to 1D for easier indexing

    for i in range(n_clusters):
        top_indices = order_centroids[i, :10]
        top_terms = [terms[ind] for ind in top_indices]
        translated = translate_terms(top_terms)
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'terms': translated,
            'values': [1] * len(translated)
        })
        
        # Plot using the DataFrame
        sns.barplot(
            data=plot_data,
            x='values',
            y='terms',
            ax=axes[i],
            orient='h',
            palette="Blues_d"
        )
        axes[i].set_title(f"Cluster {i} Top Terms")
        axes[i].set_xlim(0, 1.1)
        axes[i].get_xaxis().set_visible(False)

    plt.suptitle(f"Top Cluster Terms (Translated) — {n_clusters} Clusters", fontsize=14)
    plt.tight_layout()
    plt.show()
    # === Dimensionality Reduction ===
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X.toarray())  # must convert sparse to dense

    # === Plot the clustered points ===
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=cluster_labels,
        cmap='tab10',
        alpha=0.7
    )
    plt.colorbar(scatter, ticks=range(len(set(cluster_labels))))
    plt.title(f"KMeans Clustering with {n_clusters} Clusters (PCA Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()
# === Export ===
df_data.to_csv("property_description_clusters.csv", index=False)
print("Exported results to property_description_clusters.csv")