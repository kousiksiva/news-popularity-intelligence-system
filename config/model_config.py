import os

# 📂 Project Root Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 📊 Data Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA = os.path.join(DATA_DIR, 'News_dataset.csv')
CLEANED_DATA = os.path.join(DATA_DIR, 'cleaned_news_data.csv')
PROXIED_DATA = os.path.join(DATA_DIR, 'news_with_proxy_scores.csv')
RANKED_DATA = os.path.join(DATA_DIR, 'ranked_news_articles.csv')

# 🤖 Model Paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
POPULARITY_MODEL_PATH = os.path.join(MODEL_DIR, 'popularity_model.pkl')

# ⚙️ Model Hyperparameters
# Adjust these based on your specific News Popularity logic
POPULARITY_THRESHOLD = 0.75  # Articles above this score are "Trending"
TEST_SIZE = 0.2              # 20% of data for testing
RANDOM_STATE = 42            # For reproducible results

# 📝 NLP Settings
LANGUAGE = 'en'
FEATURES_TO_USE = ['sentiment', 'subjectivity', 'word_count', 'avg_sentence_len']

print("✅ Configuration loaded successfully.")