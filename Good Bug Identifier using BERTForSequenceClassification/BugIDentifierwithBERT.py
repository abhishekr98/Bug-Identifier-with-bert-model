import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Load a pre-trained BERT model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"  # Using a sentiment model as a base
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Pipeline for text classification
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


# Function to check if the bug title is good or bad based on sentiment
def check_bug_title(title):
    sentiment = classifier(title)[0]
    is_good = sentiment['label'] != 'NEGATIVE'

    return {
        'title': title,
        'is_good': is_good,
        'sentiment_label': sentiment['label'],
        'sentiment_score': sentiment['score']
    }


# Test the function with user input
user_title = input("Enter a bug title: ")
result = check_bug_title(user_title)

# Display the result
print("\nClassification Result:")
print(f"Title: {result['title']}")
print(f"Is it a good title? {'Yes' if result['is_good'] else 'No'}")
print(f"Sentiment label: {result['sentiment_label']}")
print(f"Sentiment score: {result['sentiment_score']:.2f}")
