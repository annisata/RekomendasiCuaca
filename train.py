import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("seattle-weather.csv")

# Encode target
le = LabelEncoder()
df["weather_encoded"] = le.fit_transform(df["weather"])

X = df[["temp_max", "temp_min", "wind", "precipitation"]]
y = df["weather_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model + encoder
with open("model.pkl", "wb") as f:
    pickle.dump((model, le), f)

print("Model saved: model.pkl")
