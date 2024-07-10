from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load intent model and tokenizer
with open('models/intent_model.pkl', 'rb') as f:
    intent_model = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        text = data['text']
        
        # Tokenize the input text
        vectorizer = intent_model.named_steps['tfidfvectorizer']
        classifier = intent_model.named_steps['svc']
        inputs = vectorizer.transform([text])
        
        # Get model outputs
        outputs = classifier.predict(inputs)
        
        # Get predicted class and intent
        predicted_class = outputs[0]
        intent = label_encoder.inverse_transform([predicted_class])[0]
        
        # Log the output logits for debugging
        print(f"Text: {text}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Intent: {intent}")
        
        return jsonify({'intent': intent})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
