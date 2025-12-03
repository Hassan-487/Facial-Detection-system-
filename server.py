
import flask
from flask import request, jsonify
import sqlite3
import numpy as np
from cryptography.fernet import Fernet
import base64
import os
import logging
from deepface import DeepFace
import cv2
from flask_cors import CORS
app = flask.Flask(__name__)
CORS(app)

DATABASE_FILE = 'secure_attendance.db'
ENCRYPTION_KEY = b'gA5N_bF-a-Y_Y0h-b_iP-q_zT-a-K_lP-x-i_cE-o_E='
FERNET_SUITE = Fernet(ENCRYPTION_KEY)
MODEL_NAME = "Facenet512"  # Using Facenet512 for better accuracy
DETECTOR_BACKEND = "opencv"
RECOGNITION_THRESHOLD = 0.6  

# This dictionary will hold {user_id: {"name": str, "embedding": np.array}}
known_embeddings = {}

logging.basicConfig(level=logging.INFO)

def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    # User table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encrypted_embedding BLOB NOT NULL
        )
    ''')
    # Log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()
    logging.info("Database initialized.")

def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def encrypt_data(data):
    return FERNET_SUITE.encrypt(data)

def decrypt_data(encrypted_data):
    return FERNET_SUITE.decrypt(encrypted_data)

# --- In-Memory Cache Loader ---
def load_known_embeddings():
    global known_embeddings
    known_embeddings.clear()# to avoid again getting data from db
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, encrypted_embedding FROM users")
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        try:
            user_id = row['id']
            name = row['name']
            decrypted_bytes = decrypt_data(row['encrypted_embedding'])
            embedding = np.frombuffer(decrypted_bytes, dtype=np.float32)
            
            known_embeddings[user_id] = {
                "name": name,
                "embedding": embedding
            }
        except Exception as e:
            logging.error(f"Failed to load embedding for user {row['id']} ({row['name']}): {e}")
            
    logging.info(f"Loaded {len(known_embeddings)} known embeddings into memory.")

# --- Helper: base64 to image ---
def image_from_base64(base64_string):
    try:
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        # Client sends cropped BGR, DeepFace needs RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logging.error(f"Image decode error: {e}")
        return None

# --- Helper: Get Embedding ---
def get_embedding(img):
    
    try:
        embeddings = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND
        )
        
        if len(embeddings) == 0:
            return None, "No face detected in the image."
        if len(embeddings) > 1:
            return None, "Multiple faces detected. Please ensure only one face is visible."
            
        embedding = np.array(embeddings[0]["embedding"], dtype=np.float32)
        return embedding, None

    except Exception as e:
        # This catches errors from enforce_detection=True if no face is found
        if "Face could not be detected" in str(e):
            return None, "No face detected in the image."
        logging.error(f"DeepFace.represent error: {e}")
        return None, f"Internal error during face representation: {e}"

# --- Enrollment ---
@app.route('/enroll', methods=['POST'])
def enroll_user():
    data = request.get_json()
    if not data or 'name' not in data or 'image' not in data:
        return jsonify({'status':'error','message':'Missing name or image'}), 400

    name = data['name']
    img = image_from_base64(data['image'])
    if img is None:
        return jsonify({'status':'error','message':'Invalid image data'}), 400

    embedding, error_msg = get_embedding(img)
    if error_msg:
        return jsonify({'status':'error','message': error_msg}), 400

    try:
        # Encrypt embedding
        embedding_bytes = embedding.tobytes()
        encrypted_embedding = encrypt_data(embedding_bytes)

        # Store in DB
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, encrypted_embedding) VALUES (?,?)", (name, encrypted_embedding))
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # --- IMPORTANT: Add to in-memory cache ---
        known_embeddings[user_id] = {
            "name": name,
            "embedding": embedding
        }

        logging.info(f"Enrolled user: {name} (ID: {user_id})")
        return jsonify({'status':'success','message':f'User {name} enrolled successfully'}), 201

    except Exception as e:
        logging.error(f"Enrollment DB error: {e}")
        return jsonify({'status':'error','message':f'Internal server error: {e}'}), 500

# --- Attendance ---
@app.route('/attend', methods=['POST'])
def mark_attendance():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'status':'error','message':'Missing image'}), 400

    img = image_from_base64(data['image'])
    if img is None:
        return jsonify({'status':'error','message':'Invalid image'}), 400

    unknown_embedding, error_msg = get_embedding(img)
    if error_msg:
        return jsonify({'status':'error','message': error_msg}), 400

    # --- Fast In-Memory Comparison ---
    # No DB query here, iterating over the pre-loaded cache
    best_match_name = "Unknown"
    best_match_id = None
    best_similarity = 0.0

    for user_id, data in known_embeddings.items():
        known_embedding = data["embedding"]
        name = data["name"]
        
        # Compare embeddings (cosine similarity)
        similarity = np.dot(unknown_embedding, known_embedding) / (np.linalg.norm(unknown_embedding) * np.linalg.norm(known_embedding))
        
        if similarity > RECOGNITION_THRESHOLD and similarity > best_similarity:
            best_similarity = similarity
            best_match_name = name
            best_match_id = user_id

    # --- Log to DB if recognized ---
    if best_match_id is not None:
        try:
            conn = get_db_connection()
            conn.execute("INSERT INTO attendance_log (user_id) VALUES (?)", (best_match_id,))
            conn.commit()
            conn.close()
            logging.info(f"Attendance recorded for: {best_match_name}")
            return jsonify({'status':'success', 'name': best_match_name, 'similarity': f"{best_similarity:.2f}"})
        except Exception as e:
            logging.error(f"Attendance logging error: {e}")
            return jsonify({'status':'error','message': 'Face recognized, but failed to log attendance.'}), 500
    
    # --- No match found ---
    logging.info("Attendance attempt by unknown user.")
    return jsonify({'status':'unknown','name':'Unknown'})


# --- Main ---
if __name__ == "__main__":
    if not os.path.exists(DATABASE_FILE):
        init_db()
    
    # Load all known users into memory at start
    load_known_embeddings()
    
    logging.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc', debug=True)