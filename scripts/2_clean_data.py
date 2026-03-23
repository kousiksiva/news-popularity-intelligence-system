import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from utils.text_utils import clean_text, combine_text

df = pd.read_csv("data/News_dataset.csv")

# FIX column names
df.columns = df.columns.str.lower()

# remove nulls
df = df.dropna(subset=["title", "description"])

# combine
df["text"] = df.apply(
    lambda x: combine_text(x["title"], x["description"]), axis=1
)

# clean
df["text"] = df["text"].apply(clean_text)

df.to_csv("data/cleaned_news_data.csv", index=False)

print("Cleaned data saved")