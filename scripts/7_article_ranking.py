import pandas as pd
import joblib

model = joblib.load("models/popularity_model.pkl")

df = pd.read_csv("data/news_with_proxy_scores.csv")

X = df[["emotion","readability","diversity","length","subjectivity"]]

df["predicted_popularity"] = model.predict(X)

df = df.sort_values(by="predicted_popularity", ascending=False)

df.to_csv("data/ranked_news_articles.csv", index=False)

print(df[["title","predicted_popularity"]].head(10))