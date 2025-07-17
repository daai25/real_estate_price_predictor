# explore_clusters.py

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from database.select_data import get_all_properties


def translate_terms(terms: List[str], source_lang: str = "auto", target_lang: str = "en") -> List[str]:
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    return [translator.translate(term) for term in terms]

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

# Load and vectorize
df_data = PropertyDataLoader().load_data()
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df_data['description'])
terms = vectorizer.get_feature_names_out()

# Try multiple cluster values

# --- Compute clustering metric (silhouette score) for each n_clusters ---
from sklearn.metrics import silhouette_score

cluster_range = range(2, 30)
silhouette_scores = []

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    score = silhouette_score(X, cluster_labels)
    silhouette_scores.append(score)

# --- Plot silhouette score for each n_clusters ---
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Silhouette Score for KMeans Clustering')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()
