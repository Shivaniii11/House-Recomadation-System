import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.preprocess import load_and_preprocess
from src.recommender import recommend_houses

st.title("🏠 House Recommendation System")

df, df_encoded, scaler = load_and_preprocess()

area = st.slider("Area", 500, 5000)
bedrooms = st.selectbox("Bedrooms", [1,2,3,4,5])
bathrooms = st.selectbox("Bathrooms", [1,2,3,4])
parking = st.selectbox("Parking", [0,1,2,3])
house_age = st.slider("House Age", 0, 20)
location = st.selectbox("Location", ["Tier-1", "Tier-2", "Tier-3"])
furnishing = st.selectbox("Furnishing", ["Furnished", "Semi", "Unfurnished"])

if st.button("Recommend Houses"):

    user_input = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking": parking,
        "house_age": house_age,

        # ✅ REQUIRED FOR FILTER
        "tier": location,

        # encoded features
        "tier_Tier-1": 1 if location == "Tier-1" else 0,
        "tier_Tier-2": 1 if location == "Tier-2" else 0,
        "tier_Tier-3": 1 if location == "Tier-3" else 0,

        "furnishing_Furnished": 1 if furnishing == "Furnished" else 0,
        "furnishing_Semi": 1 if furnishing == "Semi" else 0,
        "furnishing_Unfurnished": 1 if furnishing == "Unfurnished" else 0,
    }

    result = recommend_houses(user_input, df, df_encoded, scaler)

    # ✅ HANDLE STRING + DATAFRAME
    if isinstance(result, str):
        st.error(result)
    else:
        st.dataframe(result)