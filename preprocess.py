import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():

    df = pd.read_csv("C:/30 days/Python-ML/p22-House Recommandation/data/housing_csv.csv")

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=["tier", "furnishing"])

    # ❌ REMOVE price (not a feature)
    if "price" in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=["price"])

    # Feature columns
    feature_columns = df_encoded.columns

    scaler = StandardScaler()
    df_encoded[feature_columns] = scaler.fit_transform(df_encoded[feature_columns])

    return df, df_encoded, scaler