from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from google.cloud import firestore
import uuid
import jwt
import datetime
from functools import wraps

auth = Blueprint('auth', __name__)

# Initialize Firestore DB
db = firestore.Client(project="capstone-api-426212")

# Secret key for JWT
SECRET_KEY = "YOUR_SECRET_KEY"

@auth.route('/register', methods=['POST'])
def register():
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')

    if not name or not email or not password:
        return jsonify({"error": "Name, email, and password are required"}), 400

    user_id = str(uuid.uuid4())
    hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)

    token = jwt.encode({
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, SECRET_KEY, algorithm='HS256')

    user_data = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "password": hashed_password,
        "token": token
    }

    db.collection('users').document(user_id).set(user_data)

    return jsonify({
        "message": "User registered successfully",
        "data": {
            "user_id": user_id,
            "name": name,
            "email": email
        },
        "token": token
    }), 200

@auth.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    users_ref = db.collection('users')
    query = users_ref.where('email', '==', email).stream()

    user = None
    for doc in query:
        user = doc.to_dict()
        break

    if user and check_password_hash(user['password'], password):
        token = user.get('token')
        if not token:
            token = jwt.encode({
                'user_id': user['user_id'],
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }, SECRET_KEY, algorithm='HS256')
            # Update the user's token in Firestore
            db.collection('users').document(user['user_id']).update({"token": token})

        return jsonify({
            "message": "Login successful",
            "data": {
                "user_id": user['user_id'],
                "name": user['name'],
                "email": user['email']
            },
            "token": token
        }), 200
    else:
        return jsonify({"error": "Invalid email or password"}), 401

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"error": "Token is missing!"}), 401
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user = data['user_id']
        except:
            return jsonify({"error": "Token is invalid!"}), 401
        return f(current_user, *args, **kwargs)
    return decorated
