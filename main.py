from src.preprocess import load_and_preprocess
from src.recommender import recommend_houses

# Load data
df, df_encoded, scaler = load_and_preprocess()

# User input
user_input = {
    "area": float(input("Enter area: ")),
    "bedrooms": int(input("Bedrooms: ")),
    "bathrooms": int(input("Bathrooms: ")),
    "parking": int(input("Parking: ")),
    "house_age": int(input("House age: ")),
    "location_Tier-1": 1 if input("Location Tier-1? (y/n): ") == "y" else 0,
    "location_Tier-2": 1 if input("Location Tier-2? (y/n): ") == "y" else 0,
    "location_Tier-3": 1 if input("Location Tier-3? (y/n): ") == "y" else 0,
    "furnishing_Furnished": 1 if input("Furnished? (y/n): ") == "y" else 0,
    "furnishing_Semi": 1 if input("Semi furnished? (y/n): ") == "y" else 0,
    "furnishing_Unfurnished": 1 if input("Unfurnished? (y/n): ") == "y" else 0,
}

# Get recommendations
result = recommend_houses(user_input, df, df_encoded, scaler)

print("\nTop Recommended Houses:\n")
print(result)