import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from utils.scoring_utils import *

df = pd.read_csv("data/cleaned_news_data.csv")

df["emotion"] = df["text"].apply(emotion_score)
df["subjectivity"] = df["text"].apply(subjectivity_score)
df["readability"] = df["text"].apply(readability_score)
df["diversity"] = df["text"].apply(lexical_diversity)
df["length"] = df["text"].apply(length_score)

df["popularity_proxy"] = (
    0.25*df["emotion"] +
    0.2*df["readability"]/100 +
    0.15*df["diversity"] +
    0.2*df["length"]/100 +
    0.2*df["subjectivity"]
)

df.to_csv("data/news_with_proxy_scores.csv", index=False)

print("Proxy signals done")