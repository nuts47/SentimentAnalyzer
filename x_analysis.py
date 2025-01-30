import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt

file_path = "/Users/ayaan/VSC Projects/x analysis/kaggle_RC_2019-05.csv"  

try:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print("File loaded successfully!")
except UnicodeDecodeError:
    print("Encoding error encountered. Trying cp1252 encoding...")
    df = pd.read_csv(file_path, encoding='cp1252')

print("Columns in the dataset:")
print(df.columns)

print("\nFirst few rows of the dataset:")
print(df.head())

text_column = 'body' 

def clean_text(text):
    if not isinstance(text, str):
        return ""  

    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"\@\w+|\#\w+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if len(word) > 2]
    return " ".join(tokens)

df['cleaned_text'] = df[text_column].apply(clean_text)

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)

output_file = "processed_reddit_comments.csv"
df.to_csv(output_file, index=False)
print(f"Processed dataset saved to {output_file}")

sentiment_counts = df['sentiment'].value_counts()
sentiment_counts.plot(kind='bar', title="Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of comments")
plt.show()