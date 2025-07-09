import psycopg2
import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import xgboost as xgb
import matplotlib.pyplot as plt

# DB-Verbindung
conn = psycopg2.connect(
    dbname="real_estate_price_predictor",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433
)

df = pd.read_sql("SELECT * FROM properties", conn)
conn.close()

# Vorverarbeitung
exclude_keywords = [
    "gewerbe", "lager", "hobby", "büro", "praxis",
    "industrie", "atelier", "geschäftslokal", "werkstatt"
]
pattern = re.compile(r"|".join(exclude_keywords), re.IGNORECASE)
df = df[~df['title'].fillna('').str.contains(pattern)]
df = df[~df['description'].fillna('').str.contains(pattern)]

df = df[(df['price'] > 200) & (df['price'] < 10000000)]
df = df[(df['area_sqm'] > 10) & (df['area_sqm'] < 1000)]

df = df.dropna(subset=['rooms', 'area_sqm', 'floor', 'zip_code', 'city', 'region', 'has_balcony', 'longitude', 'latitude'])

df['price_per_sqm'] = df['price'] / df['area_sqm']
df['room_size'] = df['area_sqm'] / df['rooms']

coords = df[['longitude', 'latitude']].values
kmeans = KMeans(n_clusters=10, random_state=42)
df['geo_cluster'] = kmeans.fit_predict(coords)

# Keywords
keywords = [
    'luxus', 'renoviert', 'neubau', 'altbau', 'hochwertig', 'modern',
    'barrierefrei', 'parkett', 'panoramafenster', 'einbauküche',
    'naturstein', 'quooker', 'wärmepumpe', 'photovoltaik', 'kamin',
    'begehbar', 'regendusche', 'grosszügig', 'hell', 'lichtdurchflutet', 'offen', 'bodenheizung',
    'terrasse', 'balkon', 'garten', 'loggia',
    'e-mobilität', 'aufzug', 'garage', 'tiefgarage', 'klimatisiert',
    'aktionspreis', 'rabatt', 'erstbezug', 'sommer-aktion',
    'ruhig', 'zentral', 'aussicht', 'rheinnähe', 'altstadt', 'autofrei'
]
for kw in keywords:
    df[f'kw_{kw}'] = df['description'].fillna('').str.contains(kw, case=False, regex=False).astype(int)

miete_df = df[df['price'] <= 20000].copy()
kauf_df = df[df['price'] > 20000].copy()

# Augmentationsfunktion
def augment(df, n_copies=3):
    augmented = []
    for _, row in df.iterrows():
        for _ in range(n_copies):
            noise = row.copy()
            noise['rooms'] += np.random.normal(0, 0.1)
            noise['area_sqm'] += np.random.normal(0, 2.0)
            noise['floor'] += np.random.choice([-1, 0, 1])
            noise['price'] *= np.random.uniform(0.97, 1.03)
            noise['price_per_sqm'] = noise['price'] / noise['area_sqm']
            noise['room_size'] = noise['area_sqm'] / noise['rooms']
            augmented.append(noise)
    return pd.DataFrame(augmented)

miete_aug = augment(miete_df, n_copies=3)
miete_df = pd.concat([miete_df, miete_aug], ignore_index=True)

kauf_aug = augment(kauf_df, n_copies=3)
kauf_df = pd.concat([kauf_df, kauf_aug], ignore_index=True)

# Trainingsfunktion
def train_model(data, name):
    print(f"\n--- Training Modell mit Augmentation & log1p: {name} ---")

    X = data[['rooms', 'area_sqm', 'floor', 'zip_code', 'city', 'region', 'has_balcony', 'longitude', 'latitude',
              'price_per_sqm', 'room_size', 'geo_cluster'] + [f'kw_{kw}' for kw in keywords]]
    y = np.log1p(data['price'])

    cat_cols = ['zip_code', 'city', 'region', 'geo_cluster']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = ohe.fit_transform(X[cat_cols])
    cat_feature_names = ohe.get_feature_names_out(cat_cols)

    X_num = X.drop(columns=cat_cols).copy()
    X_num['has_balcony'] = X_num['has_balcony'].astype(int)

    X_final = np.hstack([X_num.values, X_cat])

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    model = xgb.XGBRegressor(random_state=42)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, cv=kf, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Params for {name}: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test)
    y_test_exp = np.expm1(y_test)
    y_pred_exp = np.expm1(y_pred)

    mae = mean_absolute_error(y_test_exp, y_pred_exp)
    r2 = r2_score(y_test_exp, y_pred_exp)

    print(f"{name} MAE: {mae:.2f}")
    print(f"{name} R²: {r2:.2f}")

    # Feature Importances
    xgb.plot_importance(best_model, max_num_features=10, importance_type='weight')
    plt.title(f"Feature Importance: {name}")
    plt.tight_layout()
    plt.show()

    # NEW: Scatterplot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_exp, y_pred_exp, alpha=0.5)
    plt.plot([y_test_exp.min(), y_test_exp.max()], [y_test_exp.min(), y_test_exp.max()], 'r--')
    plt.xlabel("Actual Price (CHF)")
    plt.ylabel("Predicted Price (CHF)")
    plt.title(f"Actual vs Predicted Prices: {name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    best_model.save_model(f"xgboost_{name.lower()}_model.json")
    print(f"Best Modell {name} gespeichert als xgboost_{name.lower()}_model.json")

    if name == "Miete":
        with open("kmeans.pkl", "wb") as f:
            pickle.dump(kmeans, f)
        with open("ohe.pkl", "wb") as f:
            pickle.dump(ohe, f)

train_model(miete_df, "Miete")
train_model(kauf_df, "Kauf")