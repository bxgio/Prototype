import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import os

# Create the models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

portfolio_df = pd.read_csv('data/portfolio_intent.csv')
home_df = pd.read_csv('data/home_intent.csv')
project_df = pd.read_csv('data/project_intent.csv')
back_df = pd.read_csv('data/back_intent.csv')

# Load additional datasets
services_df = pd.read_csv('data/services_intent.csv')
about_df = pd.read_csv('data/about_intent.csv')

# Combine all dataframes
df = pd.concat([portfolio_df, home_df, project_df, back_df, services_df, about_df], ignore_index=True)


# Encode labels
label_encoder = LabelEncoder()
df['intent'] = label_encoder.fit_transform(df['intent'])

# Split data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['intent'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer and linear SVM pipeline
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
classifier = SVC(kernel='linear', C=1.0, probability=True)

pipeline = make_pipeline(vectorizer, classifier)

# Train the SVM model
pipeline.fit(train_texts, train_labels)

# Predict and evaluate
train_predictions = pipeline.predict(train_texts)
test_predictions = pipeline.predict(test_texts)

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Save the model, vectorizer, and label encoder using pickle
with open('models/intent_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

np.save('models/classes.npy', label_encoder.classes_)
