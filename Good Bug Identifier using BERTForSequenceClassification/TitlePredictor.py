import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import re

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('./bug_model')
tokenizer = BertTokenizer.from_pretrained('./bug_model')

# Set the model to evaluation mode
model.eval()

# Adjust pandas settings to avoid truncation
pd.set_option('display.max_colwidth', None)  # Show full content of each column
pd.set_option('display.max_rows', None)  # Display all rows in the DataFrame
pd.set_option('display.max_columns', None)  # Display all columns

# Function to classify titles into problem, action, and location
def classify_title(title):
    problem_patterns = ["crash", "error", "issue", "problem", "failure", "clip"]
    action_patterns = ["clicking", "submitting", "loading", "accessing", "entering"]
    location_patterns = ["page", "screen", "tab", "section", "field", "pos"]

    problem = any(re.search(pattern, title, re.IGNORECASE) for pattern in problem_patterns)
    action = any(re.search(pattern, title, re.IGNORECASE) for pattern in action_patterns)
    location = any(re.search(pattern, title, re.IGNORECASE) for pattern in location_patterns)

    return pd.Series([problem, action, location], index=['problem', 'action', 'location'])


# Function to predict if a bug title is good or bad
def predict_good_or_bad(title):
    # Tokenize the input title
    inputs = tokenizer(title, return_tensors="pt", padding=True, truncation=True)

    # Perform inference (prediction)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class (good or bad)
    prediction = torch.argmax(outputs.logits, dim=1).item()

    return prediction


# Function to test new titles
def test_custom_titles(titles):
    results = []
    for title in titles:
        classified = classify_title(title)

        # Predict if the title is good or bad
        prediction = predict_good_or_bad(title)

        results.append({
            'title': title,
            'problem_present': classified['problem'],
            'action_present': classified['action'],
            'location_present': classified['location'],
            'is_good': 'good' if prediction == 1 else 'bad'
        })
    return pd.DataFrame(results)


# Get custom titles from the user
user_titles = []
print("Enter bug titles (type 'done' when finished):")
while True:
    title = input("Title: ")
    if title.lower() == 'done':
        break
    user_titles.append(title)

# Process the custom titles
results_df = test_custom_titles(user_titles)

# Display the results
print("\nPAL Classification Results:")
print(results_df)
