import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Step 1: Load and Preprocess Data
data = pd.read_csv("ques.csv", encoding="latin-1")

# Combine Subtopics and Self-learning Topics into one column
data['Topics'] = data['Subtopics'].astype(str) + ' , ' + data['Self-learning Topics'].astype(str)

# Drop unused columns
data.drop(['Subtopics', 'Self-learning Topics', 'Topic'], axis=1, inplace=True)

# Prepare features and target
train_x = data['Topics'].values
train_y = data['Unit'].astype(int).values  # Make sure 'Unit' column exists with correct name

# Step 2: Train the Model
model = make_pipeline(CountVectorizer(stop_words='english'), MultinomialNB())
model.fit(train_x, train_y)

# Step 3: Test the Model with Example Questions
test_questions = [
    "Discuss Phone rootkits",
    "Write short note on PGP",
    "Write short note on Digital signature",
    "Explain inference",
    "What is SOAP web service",
    "Discuss various types of P-Boxes",
    "Write a short note on SAML assertion",
    "Discuss various types of authentication tokens",
    "Describe working of S/MIME",
    "Explain DOS attack",
]

predicted_units = model.predict(test_questions)

# Print Predictions
print("Predicted Units:\n")
for question, unit in zip(test_questions, predicted_units):
    print(f"Question: {question}\nPredicted Unit: {unit}\n")

# Step 4: Save the Trained Model for Later Use in Web App
with open("unit_predictor.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'unit_predictor.pkl'")
