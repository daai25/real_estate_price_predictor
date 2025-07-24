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
    # Go up two directory levels (from eda/properties) to reach project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    full_path = os.path.join(project_root, json_path)
    with open(full_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def main():
    # Load cleaned data with the same relative path, but now from project root
    df = load_data("data_acquisition/data_preparation/cleaned_data/cleaned_data.json")
    print("✅ Data loaded successfully.")
    # Rest of the code remains the same...
    
    # ... (rest of the main function remains unchanged)

if __name__ == "__main__":
    main()