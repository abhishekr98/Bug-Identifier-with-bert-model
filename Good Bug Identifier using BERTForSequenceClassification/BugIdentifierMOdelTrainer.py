import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split

# Load the CSV file containing bug titles
df = pd.read_csv('bug_titles.csv')

# Prepare the data
X = df['title'].tolist()
y = df['isgood'].tolist()

# Ensure labels are in integer format (0 for bad, 1 for good)
y = [1 if label == 'good' else 0 for label in y]

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=True)
inputs = tokenizer(X, padding=True, truncation=True, return_tensors="pt")

# Convert labels to tensor
labels = torch.tensor(y, dtype=torch.long)

# Create Dataset class
class BugDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

# Split the dataset into training and evaluation datasets
X_train, X_val, y_train, y_val = train_test_split(inputs['input_ids'], y, test_size=0.2, random_state=42)

# Create the training and evaluation datasets
train_dataset = BugDataset(
    input_ids=X_train,
    attention_mask=inputs['attention_mask'][:len(X_train)],
    labels=torch.tensor(y_train, dtype=torch.long)
)

eval_dataset = BugDataset(
    input_ids=X_val,
    attention_mask=inputs['attention_mask'][len(X_train):],
    labels=torch.tensor(y_val, dtype=torch.long)
)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
)

# Initialize the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Include the evaluation dataset
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./bug_model')
tokenizer.save_pretrained('./bug_model')

print("Model training complete and saved.")
