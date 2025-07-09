import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set up plotting
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data(json_path):
    with open(os.path.abspath(json_path), "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def main():
    # Load cleaned data
    df = load_data("data_acquisition/data_preparation/cleaned_data/cleaned_data.json")
    print("✅ Data loaded successfully.")
    print(f"Shape: {df.shape}")
    print("🔎 Columns:\n", df.columns)

    # Basic info
    print("\n📌 Data types:\n", df.dtypes)
    print("\n📉 Summary stats:\n", df.describe(include='all'))

    # Check for missing values
    print("\n🚩 Missing values:\n", df.isnull().sum())

    # Convert to datetime
    df['availability_date'] = pd.to_datetime(df['availability_date'], errors='coerce')

    # Filter out suspicious values
    df = df[df['price'] >= 500]
    df = df[df['price'] <= 10000]

    # Add derived feature: price per sqm
    df['price_per_sqm'] = df['price'] / df['area_sqm']

    # Plot distributions
    sns.histplot(df['price'], bins=30, kde=True)
    plt.xlim(500, df['price'].max())
    plt.title("Price Distribution")
    plt.xlabel("Price (CHF)")
    plt.savefig("eda_price_distribution.png")
    plt.clf()

    sns.histplot(df['area_sqm'], bins=30, kde=True)
    plt.title("Area Distribution")
    plt.xlabel("Area (sqm)")
    plt.savefig("eda_area_distribution.png")
    plt.clf()

    sns.histplot(df['rooms'], bins=20, kde=False)
    plt.title("Room Count Distribution")
    plt.xlabel("Rooms")
    plt.savefig("eda_rooms_distribution.png")
    plt.clf()

    sns.histplot(df['price_per_sqm'], bins=40, kde=True)
    plt.title("Price per m² Distribution")
    plt.xlabel("CHF per m²")
    plt.savefig("eda_price_per_sqm.png")
    plt.clf()

    # Correlation heatmap
    numeric_cols = ['price', 'rooms', 'floor', 'area_sqm', 'price_per_sqm']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("eda_correlation_heatmap.png")
    plt.clf()

    # Price per region
    if 'region' in df.columns:
        region_avg = df.groupby("region")["price"].mean().sort_values()
        region_avg.plot(kind="barh", title="Average Price by Region")
        plt.xlabel("Avg Price (CHF)")
        plt.tight_layout()
        plt.savefig("eda_avg_price_by_region.png")
        plt.clf()

    # Description length vs. price
    df["desc_length"] = df["description"].apply(lambda x: len(x) if isinstance(x, str) else 0)
    sns.scatterplot(data=df, x="desc_length", y="price")
    plt.title("Description Length vs. Price")
    plt.xlabel("Description Length (chars)")
    plt.ylabel("Price (CHF)")
    plt.savefig("eda_description_length_vs_price.png")
    plt.clf()

    if 'region' in df.columns:
        fig = px.scatter_3d(
            df,
            x='region',
            y='area_sqm',
            z='price_per_sqm',
            color='region',
            size='price_per_sqm',
            title='3D Plot: Price per m² by Canton and Area',
            labels={'region': 'Canton', 'area_sqm': 'Area (sqm)', 'price_per_sqm': 'CHF/m²'}
        )
        fig.write_html("eda_3d_price_per_sqm_by_region.html")
        print("🧭 3D plot saved as 'eda_3d_price_per_sqm_by_region.html'")
        
    print("📊 EDA plots saved successfully.")

if __name__ == "__main__":
    main()
