# cluster_property_descriptions.py

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

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


def cluster_property_descriptions(n_clusters: int = 5, output_csv: str = "property_description_clusters.csv"):
    loader = PropertyDataLoader()
    df = loader.load_data()

    # Semantic Embedding
    model = SentenceTransformer('all-MiniLM-L6-v2')
    X = model.encode(df['description'].tolist(), show_progress_bar=True)

    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    df["description_cluster"] = labels

    # TF-IDF for term inspection
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    # Plot top terms per cluster
    fig, axes = plt.subplots(min(n_clusters, 6), 1, figsize=(10, 2.5 * min(n_clusters, 6)), squeeze=False)
    axes = axes.flatten()

    for i in range(min(n_clusters, 6)):
        top_indices = order_centroids[i, :10]
        top_terms = [terms[ind] for ind in top_indices]
        translated = translate_terms(top_terms)
        sns.barplot(
            data=pd.DataFrame({'terms': translated, 'values': [1] * len(translated)}),
            x='values', y='terms', ax=axes[i],
            orient='h', palette='Blues_d'
        )
        axes[i].set_title(f"Cluster {i} Top Terms")
        axes[i].set_xlim(0, 1.1)
        axes[i].get_xaxis().set_visible(False)

    # Export
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_csv)
    df.to_csv(output_path, index=False)
    print(f"Exported clustered data to: {output_path}")


if __name__ == "__main__":
    cluster_property_descriptions(n_clusters=5)
