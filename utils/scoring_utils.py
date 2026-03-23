from textblob import TextBlob
import textstat

def emotion_score(text):
    return abs(TextBlob(text).sentiment.polarity)

def subjectivity_score(text):
    return TextBlob(text).sentiment.subjectivity

def readability_score(text):
    return textstat.flesch_reading_ease(text)

def lexical_diversity(text):
    words = text.split()
    return len(set(words)) / len(words) if len(words) > 0 else 0

def length_score(text):
    return len(text.split())