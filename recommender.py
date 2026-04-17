import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend_houses(user_input, df, df_encoded, scaler):

    # ✅ STEP 1: STRICT FILTER
    filtered_df = df[
        (df['tier'] == user_input['tier']) &
        (df['bedrooms'] >= user_input['bedrooms']) &
        (df['bathrooms'] >= user_input['bathrooms']) &
        (df['parking'] >= user_input['parking'])
    ]

    # ❌ NO MATCH CASE
    if filtered_df.empty:
        return "❌ No house available according to your expectations"

    # ✅ STEP 2: FILTER ENCODED DATA
    filtered_encoded = df_encoded.loc[filtered_df.index]

    # ✅ STEP 3: CREATE USER VECTOR
    user_df = pd.DataFrame([user_input])

    # Ensure same feature columns ONLY
    feature_columns = filtered_encoded.columns
    user_df = user_df.reindex(columns=feature_columns, fill_value=0)

    # Scale user input
    user_scaled = scaler.transform(user_df)

    # ✅ STEP 4: SIMILARITY
    similarity = cosine_similarity(user_scaled, filtered_encoded)[0]

    filtered_df = filtered_df.copy()
    filtered_df["similarity"] = similarity

    # ✅ STEP 5: RETURN TOP RESULTS
    return filtered_df.sort_values(by="similarity", ascending=False).head(5)