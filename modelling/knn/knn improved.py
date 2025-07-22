import sqlite3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# Machine Learning
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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

# === Semantic Embedding ===
model = SentenceTransformer('all-MiniLM-L6-v2')
X = model.encode(df_data['description'].tolist(), show_progress_bar=True)

# === Evaluate Clustering Quality ===
elbow_inertias = []
silhouette_scores = []
cluster_range = range(2, 30)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    inertia = kmeans.inertia_
    elbow_inertias.append(inertia)
    silhouette = silhouette_score(X, labels)
    silhouette_scores.append(silhouette)
    print(f"K={n_clusters}, Inertia={inertia:.2f}, Silhouette={silhouette:.4f}")

# === Plot Elbow Method ===
plt.figure(figsize=(8, 4))
plt.plot(cluster_range, elbow_inertias, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot Silhouette Scores ===
plt.figure(figsize=(8, 4))
plt.plot(cluster_range, silhouette_scores, 'go-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by Number of Clusters')
plt.grid(True)
plt.tight_layout()
plt.show()

# === KMeans Clustering and Visualization ===
from sklearn.feature_extraction.text import TfidfVectorizer

for n_clusters in cluster_range:
    print(f"\n=== KMeans with {n_clusters} Clusters ===")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # PCA Projection
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

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

    # Show top terms using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df_data['description'])
    terms = vectorizer.get_feature_names_out()

    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    fig, axes = plt.subplots(min(n_clusters, 6), 1, figsize=(10, 2.5 * min(n_clusters, 6)), squeeze=False)
    axes = axes.flatten()
    for i in range(min(n_clusters, 6)):
        top_indices = order_centroids[i, :10]
        top_terms = [terms[ind] for ind in top_indices]
        translated = translate_terms(top_terms)

        plot_data = pd.DataFrame({
            'terms': translated,
            'values': [1] * len(translated)
        })

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
    plt.subplots_adjust(top=0.95)
    plt.show()
    cluster_term_table = []

    for i in range(n_clusters):
        top_indices = order_centroids[i, :10]
        top_terms = [terms[ind] for ind in top_indices]
        translated = translate_terms(top_terms)
        cluster_term_table.append({
            "Cluster": f"Cluster {i}",
            "Top Terms": ", ".join(translated)
        })

    # Convert to DataFrame for clean viewing/export
    df_clusters = pd.DataFrame(cluster_term_table)
    print(df_clusters.to_string(index=False))

    # Optional: export to CSV
    df_clusters.to_csv("cluster_top_terms.csv", index=False)
# === Export ===
df_data.to_csv("property_description_clusters.csv", index=False)
print("Exported results to property_description_clusters.csv")