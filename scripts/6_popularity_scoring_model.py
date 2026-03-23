import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/news_with_proxy_scores.csv")

X = df[["emotion","readability","diversity","length","subjectivity"]]
y = df["popularity_proxy"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "models/popularity_model.pkl")

print("Model saved")