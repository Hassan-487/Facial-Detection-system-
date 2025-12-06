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
from werkzeug.security import generate_password_hash, check_password_hash
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import threading

app = flask.Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
DATABASE_FILE = 'secure_attendance.db'
ENCRYPTION_KEY = b'gA5N_bF-a-Y_Y0h-b_iP-q_zT-a-K_lP-x-i_cE-o_E='
FERNET_SUITE = Fernet(ENCRYPTION_KEY)
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
RECOGNITION_THRESHOLD = 0.6 
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "hassaniftikhar487@gmail.com"      
SENDER_PASSWORD = "cgzs iqmk oonj ykku"     
ADMIN_EMAIL = "l226856@lhr.nu.edu.pk"     

OFFICE_START_HOUR = 9   
OFFICE_END_HOUR = 18    

SYSTEM_MODE = "active"
known_embeddings = {}

logging.basicConfig(level=logging.INFO)

# --- DATABASE HELPERS ---
def get_db_connection():
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encrypted_embedding BLOB NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    create_default_admin()
    logging.info("Database initialized.")

def create_default_admin():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admins WHERE username = ?", ('admin',))
    if not cursor.fetchone():
        p_hash = generate_password_hash('admin123')
        cursor.execute("INSERT INTO admins (username, password_hash) VALUES (?, ?)", ('admin', p_hash))
        conn.commit()
        logging.info("Default admin created (admin/admin123)")
    conn.close()

# --- UTILITIES ---
def encrypt_data(data):
    return FERNET_SUITE.encrypt(data)

def decrypt_data(encrypted_data):
    return FERNET_SUITE.decrypt(encrypted_data)

def load_known_embeddings():
    global known_embeddings
    known_embeddings.clear()
    conn = get_db_connection()
    rows = conn.execute("SELECT id, name, encrypted_embedding FROM users").fetchall()
    conn.close()

    for row in rows:
        try:
            decrypted_bytes = decrypt_data(row['encrypted_embedding'])
            embedding = np.frombuffer(decrypted_bytes, dtype=np.float32)
            known_embeddings[row['id']] = {"name": row['name'], "embedding": embedding}
        except Exception as e:
            logging.error(f"Error loading user {row['id']}: {e}")
    logging.info(f"Loaded {len(known_embeddings)} users into memory.")

def image_from_base64(base64_string):
    try:
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None: return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        return None

def get_embedding(img):
    try:
        embeddings = DeepFace.represent(
            img_path=img, model_name=MODEL_NAME, 
            enforce_detection=True, detector_backend=DETECTOR_BACKEND
        )
        if len(embeddings) == 0: return None, "No face detected"
        if len(embeddings) > 1: return None, "Multiple faces detected"
        return np.array(embeddings[0]["embedding"], dtype=np.float32), None
    except Exception as e:
        if "Face could not be detected" in str(e): return None, "No face detected"
        return None, str(e)

# --- EMAIL ALERT FUNCTION ---
def send_anomaly_email(user_name, user_id, timestamp):
    """Sends an email in a separate thread to avoid blocking the API"""
    def _send():
        subject = f"Security Alert: Anomaly Attendance Detected ({user_name})"
        body = f"""
        WARNING: Anomaly Detected.
        
        User: {user_name} (ID: {user_id})
        Time: {timestamp}
        
        This attendance was marked OUTSIDE of office hours ({OFFICE_START_HOUR}:00 - {OFFICE_END_HOUR}:00).
        Please investigate.
        """
        
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = ADMIN_EMAIL

        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, ADMIN_EMAIL, msg.as_string())
            logging.info(f"Anomaly email sent for {user_name}")
        except Exception as e:
            logging.error(f"Failed to send email: {e}")

    # Run in background thread
    threading.Thread(target=_send).start()

# --- SYSTEM STATE ROUTES ---
@app.route('/system/status', methods=['GET', 'POST'])
def system_status():
    global SYSTEM_MODE
    if request.method == 'POST':
        data = request.get_json()
        new_mode = data.get('mode')
        if new_mode in ['active', 'maintenance']:
            SYSTEM_MODE = new_mode
            logging.info(f"System Mode changed to: {SYSTEM_MODE}")
            return jsonify({'status': 'success', 'mode': SYSTEM_MODE})
        return jsonify({'status': 'error', 'message': 'Invalid mode'}), 400
    return jsonify({'mode': SYSTEM_MODE})

# --- ADMIN ROUTES ---
@app.route('/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()
    user = get_db_connection().execute('SELECT * FROM admins WHERE username = ?', (data.get('username'),)).fetchone()
    if user and check_password_hash(user['password_hash'], data.get('password')):
        return jsonify({'status': 'success', 'message': 'Login successful'})
    return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401

@app.route('/admin/users', methods=['GET'])
def get_users():
    rows = get_db_connection().execute('SELECT id, name FROM users').fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/admin/logs', methods=['GET'])
def get_logs():
    query = '''SELECT l.id, u.name, l.timestamp FROM attendance_log l 
               JOIN users u ON l.user_id = u.id ORDER BY l.timestamp DESC LIMIT 100'''
    rows = get_db_connection().execute(query).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/admin/delete_user', methods=['POST'])
def delete_user():
    uid = request.get_json().get('user_id')
    try:
        conn = get_db_connection()
        conn.execute('DELETE FROM users WHERE id = ?', (uid,))
        conn.commit()
        conn.close()
        if int(uid) in known_embeddings: del known_embeddings[int(uid)]
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- CORE ROUTES ---
@app.route('/enroll', methods=['POST'])
def enroll():
    # Only allow enrollment if system is in maintenance (Admin is doing it)
    if SYSTEM_MODE != 'maintenance':
         return jsonify({'status':'error', 'message': 'Enrollment unauthorized. Must be in maintenance mode.'}), 403

    data = request.get_json()
    img = image_from_base64(data.get('image'))
    if img is None: return jsonify({'status':'error','message':'Invalid Image'}), 400
    
    emb, err = get_embedding(img)
    if err: return jsonify({'status':'error','message': err}), 400

    try:
        enc_emb = encrypt_data(emb.tobytes())
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (name, encrypted_embedding) VALUES (?,?)", (data['name'], enc_emb))
        uid = cur.lastrowid
        conn.commit()
        conn.close()
        known_embeddings[uid] = {"name": data['name'], "embedding": emb}
        return jsonify({'status':'success', 'message': f"Enrolled {data['name']}"}), 201
    except Exception as e:
        return jsonify({'status':'error', 'message': str(e)}), 500

@app.route('/attend', methods=['POST'])
def attend():
    if SYSTEM_MODE == 'maintenance':
        return jsonify({'status': 'error', 'message': 'System in maintenance'}), 503

    img = image_from_base64(request.get_json().get('image'))
    if img is None: return jsonify({'status':'error'}), 400
    
    emb, err = get_embedding(img)
    if err: return jsonify({'status':'error', 'message': err}), 400

    best_name, best_id, best_score = None, None, 0.0
    for uid, data in known_embeddings.items():
        score = np.dot(emb, data["embedding"]) / (np.linalg.norm(emb) * np.linalg.norm(data["embedding"]))
        if score > RECOGNITION_THRESHOLD and score > best_score:
            best_score, best_name, best_id = score, data["name"], uid

    if best_id:
        now = datetime.now()
        conn = get_db_connection()
        conn.execute("INSERT INTO attendance_log (user_id) VALUES (?)", (best_id,))
        conn.commit()
        conn.close()
        
        # --- ANOMALY DETECTION LOGIC ---
        current_hour = now.hour
        # Check if OUTSIDE of 9am to 6pm (18:00)
        if not (OFFICE_START_HOUR <= current_hour < OFFICE_END_HOUR):
            print(f"[ALERT] Anomaly detected for {best_name} at {now}")
            send_anomaly_email(best_name, best_id, now.strftime("%Y-%m-%d %H:%M:%S"))

        return jsonify({'status':'success', 'name': best_name})
    
    return jsonify({'status':'unknown'})

if __name__ == "__main__":
    if not os.path.exists(DATABASE_FILE): init_db()
    load_known_embeddings()
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc', debug=True)
