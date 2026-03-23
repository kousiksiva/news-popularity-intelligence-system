import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

def extract_features():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    
    df = pd.read_csv("data/cleaned_news_data.csv")
    texts = df['full_text'].astype(str).tolist()
    embeddings = []

    print("🚀 Extracting 768-D Transformer Embeddings...")
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_token = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_token)

    np.save("data/news_embeddings.npy", np.vstack(embeddings))
    print("✅ Step 2 Complete: 768-D Features saved.")

if __name__ == "__main__":
    extract_features()