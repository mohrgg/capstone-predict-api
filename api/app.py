import json
import numpy as np
from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import random
import os
import uuid

app = Flask(__name__)

# Load the model and tokenizer
model = TFBertForSequenceClassification.from_pretrained('./model')
tokenizer = BertTokenizer.from_pretrained('./model')

# Define emotion labels and messages
emotions = ["anxiety", "depression", "happy", "lonely", "neutral"]
positive_emotions = ["happy", "neutral"]
negative_emotions = ["anxiety", "depression", "lonely"]

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
    "neutral": [
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

# In-memory storage for tweets
tweets = []

def analyze_text(text):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    probabilities = tf.nn.softmax(outputs.logits, axis=-1)
    scores = probabilities.numpy()[0]
    print("Scores:", scores)  # Debugging line to check scores

    # Calculate the weighted average score for each class
    positive_score = np.sum([scores[emotions.index(emotion)] for emotion in positive_emotions]) / len(positive_emotions)
    negative_score = np.sum([scores[emotions.index(emotion)] for emotion in negative_emotions]) / len(negative_emotions)
    print("Positive Score:", positive_score)  # Debugging line to check positive score
    print("Negative Score:", negative_score)  # Debugging line to check negative score

    if positive_score >= negative_score:
        dominant_class = "positive"
    else:
        dominant_class = "negative"

    print("Dominant Class:", dominant_class)  # Debugging line to check dominant class

    if dominant_class == "positive":
        relevant_emotions = positive_emotions
    else:
        relevant_emotions = negative_emotions

    max_index = np.argmax([scores[emotions.index(emotion)] for emotion in relevant_emotions])
    dominant_emotion = relevant_emotions[max_index]

    print("Dominant Emotion:", dominant_emotion)  # Debugging line to check dominant emotion

    return dominant_emotion

def get_random_saran():
    random_activity = activities.sample(n=1)
    description = random_activity.iloc[0]['description']
    return description

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    data = request.form
    text = data.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    emotion = analyze_text(text)
    message = random.choice(messages[emotion])
    saran = get_random_saran()
    
    result = {
        "Mental State": emotion.capitalize(),
        "Message": message,
        "Saran": saran
    }
    
    return jsonify(result)

@app.route('/save-tweet', methods=['POST'])
def save_tweet():
    data = request.form
    user_id = data.get('user_id')
    text = data.get('text')
    if not user_id or not text:
        return jsonify({"error": "User ID and text are required"}), 400
    
    detail_id = str(uuid.uuid4())
    emotion = analyze_text(text)
    tweet = {
        "user_id": user_id,
        "detail_id": detail_id,
        "text": text,
        "mental_state": emotion.capitalize()
    }
    tweets.append(tweet)
    
    return jsonify({"message": "Tweet saved successfully", "detail_id": detail_id})

@app.route('/get-tweets', methods=['GET'])
def get_tweets():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400
    
    user_tweets = [tweet for tweet in tweets if tweet['user_id'] == user_id]
    
    return jsonify(user_tweets)

@app.route('/get-tweet-detail', methods=['GET'])
def get_tweet_detail():
    detail_id = request.args.get('detail_id')
    if not detail_id:
        return jsonify({"error": "Detail ID is required"}), 400
    
    tweet = next((tweet for tweet in tweets if tweet['detail_id'] == detail_id), None)
    
    if not tweet:
        return jsonify({"error": "Tweet not found"}), 404
    
    return jsonify(tweet)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
