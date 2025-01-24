import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt

def read_text(file_path):
    try:
        with open(file_path, encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return ""
    except UnicodeDecodeError:
        print(f"Error: Unable to decode file '{file_path}' with UTF-8 encoding.")
        return ""

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(cleaned_text)
    lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_words)

def tokenize_words(text):
    return word_tokenize(text, "english")

def remove_stopwords(words):
    final_words = []
    for word in words:
        if word not in stopwords.words("english"):
            final_words.append(word)
    return final_words

def extract_emotion_words(emotion_file_path):
    emotion_words = []
    try:
        with open(emotion_file_path, 'r') as file:
            for line in file:
                word, emotion, score = line.strip().split('\t')
                if int(score) == 1:
                    emotion_words.append((word, emotion))
    except FileNotFoundError:
        print(f"Error: Emotion file '{emotion_file_path}' not found.")
    return emotion_words

def extract_emotion_list(words, emotion_words):
    emotion_list = []
    for word in words:
        for emotion_word, emotion in emotion_words:
            if word == emotion_word:
                emotion_list.append(emotion)
                break
    return emotion_list

def perform_sentiment_analysis(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    neg = score["neg"]
    pos = score["pos"]
    if neg > pos:
        return "Negative Sentiment"
    elif pos > neg:
        return "Positive Sentiment"
    else:
        return "Neutral Sentiment"

def visualize_sentiment(sentiment_counts):
    labels = sentiment_counts.keys()
    values = sentiment_counts.values()

    plt.bar(labels, values)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Analysis')

    plt.show()

def main():
    text = read_text("read.txt")
    if not text:
        return  # Exit if file reading failed

    cleaned_text = clean_text(text)
    tokenized_words = tokenize_words(cleaned_text)
    final_words = remove_stopwords(tokenized_words)

    emotion_words = extract_emotion_words("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
    if not emotion_words:
        return  # Exit if emotion file reading failed

    emotion_list = extract_emotion_list(final_words, emotion_words)
    emotion_counts = Counter(emotion_list)
    sentiment = perform_sentiment_analysis(cleaned_text)
    print("Sentiment:", sentiment)
    visualize_sentiment(emotion_counts)

if __name__ == "__main__":
    main()