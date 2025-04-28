from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
CORS(app)

model = None  # Global model

# ===== 1. Process syllabus text to extract module + topics only =====
@app.route('/process-syllabus', methods=['POST'])
def process_syllabus():
    try:
        raw_text = request.json.get('text', '')
        raw_modules = re.split(r'\n(?=\d\s)', raw_text.strip())
        data = []

        for module in raw_modules:
            lines = module.strip().splitlines()
            if not lines:
                continue

            # Filter out irrelevant lines
            filtered_lines = [
                line for line in lines
                if not re.search(r'\b(Hrs\.?|CO\b|Ref\s*No\.?)', line, re.IGNORECASE)
            ]

            if not filtered_lines:
                continue

            module_no = filtered_lines[0].split()[0].strip()
            full_text = " ".join(filtered_lines)

            # Extract topics and self-learning topics
            if 'learning topics:' in full_text.lower():
                parts = re.split(r'learning topics:', full_text, flags=re.IGNORECASE)
                subtopics = parts[0].strip()
                self_learning = parts[1].strip().split('\n')[0].strip()
            else:
                subtopics = full_text.strip()
                self_learning = ""

            topics = subtopics + " , " + self_learning
            data.append({
                "Module No.": module_no,
                "Topic": topics
            })

        df = pd.DataFrame(data)
        df.rename(columns={"Module No.": "Unit"}, inplace=True)
        df.to_csv("module_data.csv", index=False)

        return jsonify({"message": "CSV created successfully with Unit and Topic!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== 2. Train model from the generated CSV =====
@app.route('/train-model', methods=['POST'])
def train_model():
    global model
    try:
        df = pd.read_csv("module_data.csv")
        X = df['Topic'].astype(str)
        y = df['Unit'].astype(str)

        model = make_pipeline(CountVectorizer(stop_words='english'), MultinomialNB())
        model.fit(X, y)

        return jsonify({"message": "Model trained successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== 3. Predict module number for user questions =====
@app.route('/predict', methods=['POST'])
def predict():
    global model
    try:
        if model is None:
            return jsonify({"error": "Model not trained yet."}), 400

        questions = request.json.get("questions", [])
        predictions = model.predict(questions)

        result = [{"question": q, "predicted_unit": p} for q, p in zip(questions, predictions)]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "Unit Predictor API is running!"

if __name__ == '__main__':
    app.run(debug=True)
