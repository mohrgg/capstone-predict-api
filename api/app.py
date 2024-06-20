import json
import numpy as np
from flask import Flask, request, jsonify
from transformers import BertTokenizer, TFBertForSequenceClassification, TFBertModel
import tensorflow as tf
import pandas as pd
import random
import os
import uuid
from auth import auth, token_required
from google.cloud import firestore
import requests

app = Flask(__name__)

# Register the auth blueprint
app.register_blueprint(auth, url_prefix='/auth')

# Load the model and tokenizer from the local directory
local_model_dir = 'local_model'
model = tf.keras.models.load_model(local_model_dir, custom_objects={"TFBertForSequenceClassification": TFBertForSequenceClassification, "TFBertModel": TFBertModel})
tokenizer = BertTokenizer.from_pretrained(local_model_dir)

# Define emotion labels and messages
emotions = ["anxiety", "depression", "happy", "lonely", "neutral"]
positive_emotions = ["happy", "neutral"]
negative_emotions = ["anxiety", "depression", "lonely"]

messages = {
    "anxiety": [
        "You have the strength to overcome all obstacles.",
        "Stay calm and focus on things you can control.",
        "Remember to breathe and take a moment for yourself.",
        "You are capable of handling this situation, one step at a time.",
        "Don't let fear control you; you are stronger than you think.",
        "It's okay to feel anxious, but don't let it define you.",
        "Reach out to someone you trust and talk about your worries.",
        "Engage in an activity that helps you relax and unwind.",
        "Remind yourself of past challenges you've overcome.",
        "Take it easy on yourself; you are doing the best you can."
    ],
    "depression": [
        "You are valuable and important.",
        "Today might be tough, but tomorrow can be better.",
        "Don't hesitate to seek support; you are not alone.",
        "Allow yourself to feel and express your emotions.",
        "Remember that your feelings are valid and deserve attention.",
        "Take small steps towards self-care and healing.",
        "You are not defined by your struggles; you are more than that.",
        "Reach out to loved ones and let them know how you're feeling.",
        "Find a creative outlet to express your thoughts and feelings.",
        "Be kind to yourself; recovery is a journey, not a destination."
    ],
    "lonely": [
        "Reach out to friends or family; they care about you.",
        "Try to get involved in social activities or communities.",
        "Remember that this feeling of loneliness is temporary and can change.",
        "Spend time doing things you enjoy, even if you do them alone.",
        "Consider joining a club or group that interests you.",
        "Engage in volunteer work to connect with others and make a difference.",
        "Don't be afraid to initiate conversations with new people.",
        "Reflect on past times when you felt connected and happy.",
        "Practice self-compassion and be gentle with yourself.",
        "Remember that building new relationships takes time and effort."
    ],
    "neutral": [
        "Continue your day with a positive attitude.",
        "You are doing your best; keep it up!",
        "Enjoy the small moments in your life.",
        "Find joy in the little things that make you happy.",
        "Take a moment to appreciate the calm and balance in your day.",
        "Stay open to new experiences that might bring you joy.",
        "Reflect on your accomplishments, no matter how small.",
        "Maintain a healthy routine that keeps you balanced.",
        "Keep an open mind and embrace the present moment.",
        "Remember that it's okay to have neutral days; they are part of life."
    ],
    "happy": [
        "Spread happiness to those around you.",
        "Enjoy every second of this happiness.",
        "Keep doing the things that make you happy.",
        "Share your joy with others and create positive memories.",
        "Celebrate your achievements and successes.",
        "Take time to appreciate the good things in your life.",
        "Engage in activities that bring you fulfillment and joy.",
        "Express gratitude for the happiness you feel.",
        "Cherish the moments that make you smile.",
        "Continue to pursue what makes you truly happy."
    ]
}

# Download the activity.csv file
activity_csv_url = "https://storage.googleapis.com/happyoumodelbucket/activity.csv"
activity_csv_path = 'activity.csv'
response = requests.get(activity_csv_url)
with open(activity_csv_path, 'wb') as file:
    file.write(response.content)

# Load activities from CSV
activities = pd.read_csv(activity_csv_path)

# Initialize Firestore DB
db = firestore.Client(project="capstone-api-426212")

def analyze_text(text):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding='max_length', max_length=53)
    outputs = model([inputs['input_ids'], inputs['attention_mask']])
    if hasattr(outputs, 'logits'):
        probabilities = tf.nn.softmax(outputs.logits, axis=-1)
    else:
        probabilities = tf.nn.softmax(outputs, axis=-1)
    scores = probabilities.numpy()[0]

    dominant_emotions = [emotions[i] for i, score in enumerate(scores) if score > 0.5]
    if len(dominant_emotions) == 0:
        dominant_emotion = emotions[np.argmax(scores)]
    elif len(dominant_emotions) == 1:
        dominant_emotion = dominant_emotions[0]
    else:
        dominant_emotion = dominant_emotions[np.argmax([scores[emotions.index(emotion)] for emotion in dominant_emotions])]
    
    return dominant_emotion

def get_random_saran(emotion):
    # Filter activities based on the emotion activity_id
    emotion_activity_id_range = {
        "depression": range(201, 211),
        "anxiety": range(301, 311),
        "lonely": range(401, 411),
        "neutral": range(501, 511),
        "happy": range(601, 611)
    }
    filtered_activities = activities[activities['activity_id'].isin(emotion_activity_id_range[emotion])]
    random_activity = filtered_activities.sample(n=1)
    description = random_activity.iloc[0]['description']
    return description

@app.route('/save-tweet', methods=['POST'])
@token_required
def save_tweet(current_user):
    text = request.form.get('text')
    if not text:
        return jsonify({"error": "Text is required"}), 400

    tweet_id = str(uuid.uuid4())
    emotion = analyze_text(text)
    message = random.choice(messages[emotion])
    saran = get_random_saran(emotion)

    tweet_data = {
        "tweet_id": tweet_id,
        "user_id": current_user,
        "text": text,
        "mental_state": emotion,
        "message": message,
        "saran": saran
    }

    db.collection('tweets').document(tweet_id).set(tweet_data)

    return jsonify({
        "tweet_id": tweet_id,
        "text": text,
        "status": "Tweet saved successfully"
    }), 200

@app.route('/get-tweet-byuserid', methods=['GET'])
def get_tweet_byuserid():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    tweets_ref = db.collection('tweets').where('user_id', '==', user_id).stream()
    tweets = [tweet.to_dict() for tweet in tweets_ref]

    return jsonify({"tweets": tweets}), 200

@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    text = request.form.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    emotion = analyze_text(text)
    message = random.choice(messages[emotion])
    saran = get_random_saran(emotion)
    
    result = {
        "Mental State": emotion.capitalize(),
        "Message": message,
        "Saran": saran
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
