import json
import numpy as np
from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import random
import os

app = Flask(__name__)

# Load the model and tokenizer
model = TFBertForSequenceClassification.from_pretrained('./model')
tokenizer = BertTokenizer.from_pretrained('./model')

# Define emotion labels and messages
emotions = ["anxiety", "depression", "lonely", "normal", "happy"]
messages = {
    "anxiety": [
        "Kamu memiliki kekuatan untuk mengatasi semua rintangan.",
        "Tetap tenang dan fokus pada hal-hal yang bisa kamu kendalikan.",
        "Ingatlah untuk bernafas dan mengambil waktu sejenak untuk dirimu sendiri."
    ],
    "depression": [
        "Kamu berharga dan penting.",
        "Hari ini mungkin sulit, tapi besok bisa lebih baik.",
        "Jangan ragu untuk mencari dukungan, kamu tidak sendiri."
    ],
    "lonely": [
        "Hubungi teman atau keluargamu, mereka peduli padamu.",
        "Cobalah untuk terlibat dalam kegiatan sosial atau komunitas.",
        "Ingatlah bahwa perasaan kesepian ini sementara dan bisa berubah."
    ],
    "normal": [
        "Lanjutkan hari dengan semangat positif.",
        "Kamu melakukan yang terbaik, teruskan!",
        "Nikmati momen-momen kecil dalam hidupmu."
    ],
    "happy": [
        "Sebarkan kebahagiaan kepada orang di sekitarmu.",
        "Nikmati setiap detik dari kebahagiaan ini.",
        "Teruskan melakukan hal-hal yang membuatmu bahagia."
    ]
}

# Load activities from CSV
activities = pd.read_csv(os.path.join(os.path.dirname(__file__), 'activity.csv'))
# print(activities.head())  # Tambahkan ini untuk cek kolom

confidence_threshold = 25.0  # Define a threshold for confidence

def analyze_text(text):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    probabilities = tf.nn.softmax(outputs.logits, axis=-1)
    scores = probabilities.numpy()[0]
    print("Scores:", scores)  # Debugging line to check scores
    max_index = np.argmax(scores)
    dominant_emotion = emotions[max_index]
    confidence = scores[max_index] * 100
    print("Dominant Emotion:", dominant_emotion)  # Debugging line to check dominant emotion
    print("Confidence:", confidence)  # Debugging line to check confidence
    return dominant_emotion, confidence

def get_random_saran():
    random_activity = activities.sample(n=1)
    description = random_activity.iloc[0]['description']  # Pastikan kolom ini sesuai dengan kolom di CSV
    return description

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    data = request.json
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    emotion, confidence = analyze_text(text)
    message = random.choice(messages[emotion])
    saran = get_random_saran()
    
    if confidence < confidence_threshold:
        result = {
            "Mental State": "Uncertain",
            "Message": "The model is not confident about the emotion.",
            "Saran": "The model is not confident about the emotion."
        }
    else:
        result = {
            "Mental State": emotion.capitalize(),
            "Message": message,
            "Saran": saran
        }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
