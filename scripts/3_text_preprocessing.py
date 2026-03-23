import pandas as pd

df = pd.read_csv("data/cleaned_news_data.csv")

df["word_count"] = df["text"].apply(lambda x: len(x.split()))

print(df.head())