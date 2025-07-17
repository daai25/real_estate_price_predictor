import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from database.select_data import get_all_properties


def translate_terms(terms: List[str], source_lang: str = "auto", target_lang: str = "en") -> List[str]:
    """Translate a list of terms using Google Translate."""
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    return [translator.translate(term) for term in terms]


class PropertyDataLoader:
    """Loads property data either from a provided DataFrame or from the database."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df

    def load_data(self) -> pd.DataFrame:
        if self.df is None:
            properties = get_all_properties()
            if not properties:
                raise ValueError("No property data found.")
            self.df = pd.DataFrame(properties)
        return self.df


def vectorize_descriptions(df: pd.DataFrame, max_features: int = 1000) -> Tuple[pd.DataFrame, any, List[str]]:
    """Vectorize the 'description' column using TF-IDF."""
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(df['description'])
    terms = vectorizer.get_feature_names_out()
    return X, vectorizer, terms


def perform_kmeans(X, n_clusters: int = 5, random_state: int = 42) -> Tuple[KMeans, List[int]]:
    """Fit KMeans clustering and return model and labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X)
    return kmeans, cluster_labels



def cluster_property_descriptions(n_clusters: int = 5, output_csv: str = "property_description_clusters.csv"):
    """Main function to perform the clustering pipeline and save results."""
    # Load and vectorize
    loader = PropertyDataLoader()
    df = loader.load_data()
    X, vectorizer, terms = vectorize_descriptions(df)

    # Cluster
    kmeans, labels = perform_kmeans(X, n_clusters)
    df["description_cluster"] = labels

    # Export to same folder as script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_csv)
    df.to_csv(output_path, index=False)
    print(f"Exported results to {output_path}")


if __name__ == "__main__":
    cluster_property_descriptions(n_clusters=5, output_csv="property_description_clusters.csv")
