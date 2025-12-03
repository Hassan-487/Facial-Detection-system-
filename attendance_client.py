
import cv2
import requests
import base64
import time
import sys

# ----------------------------
# Configuration
# ----------------------------
SERVER_URL = "https://127.0.0.1:5000/attend"
REQUESTS_VERIFY = False       # set True if you have a valid cert
TIME_LIMIT = 15               # seconds to detect blink
REQUIRED_BLINKS = 1
MIN_CLOSED_FRAMES = 2         # eyes must be "not detected" for this many consecutive frames

if not REQUESTS_VERIFY:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------------
# Load Haar Cascades (correct attribute name)
# ----------------------------
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'  # alternate: haarcascade_eye_tree_eyeglasses.xml

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# sanity checks: ensure cascades loaded
if face_cascade.empty():
    print("ERROR: failed to load face cascade from:", face_cascade_path)
    sys.exit(1)
if eye_cascade.empty():
    print("ERROR: failed to load eye cascade from:", eye_cascade_path)
    sys.exit(1)

print("[INFO] Loaded cascades:")
print("  face:", face_cascade_path)
print("  eye: ", eye_cascade_path)

# ----------------------------
# Helper: send cropped RGB face to server
# ----------------------------
def send_face_to_server(face_bgr):
    rgb_face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    success, img_encoded = cv2.imencode('.jpg', rgb_face)
    if not success:
        print("[ERROR] Failed to encode image to JPEG")
        return
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    payload = {"image": img_base64}
    try:
        response = requests.post(SERVER_URL, json=payload, verify=REQUESTS_VERIFY, timeout=10)
        print("[INFO] Server response:", response.status_code, response.text)
    except Exception as e:
        print("[ERROR] Could not reach server:", e)

# ----------------------------
# Main: start attendance
# ----------------------------
def start_attendance():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    start_time = time.time()
    blink_count = 0
    frames_closed = 0
    last_face_rect = None

    print(f"[INFO] Please blink once within {TIME_LIMIT} seconds to verify liveness.")

    while True:
        elapsed = time.time() - start_time
        remaining = max(0, int(TIME_LIMIT - elapsed))

        if elapsed >= TIME_LIMIT:
            print("[INFO] Time is up. No valid blink detected. Marking absent.")
            break

        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            cv2.putText(frame, f"No face detected | Time left: {remaining}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # reset eye counters because no face
            frames_closed = 0
        else:
            # if multiple faces - inform user and skip until single face
            if len(faces) > 1:
                cv2.putText(frame, f"Multiple faces detected | Time left: {remaining}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # do not attempt liveness if more than one face
                frames_closed = 0
            else:
                (x, y, w, h) = faces[0]
                last_face_rect = (x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_gray = gray[y:y + h, x:x + w]
                face_color = frame[y:y + h, x:x + w]   # BGR cropped face for sending later

                # detect eyes within face region
                eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3)

                # If eyes are detected -> consider eyes open
                if len(eyes) > 0:
                    # if we had closed frames earlier and they reached threshold,
                    # then this transition means a blink occurred
                    if frames_closed >= MIN_CLOSED_FRAMES:
                        blink_count += 1
                        print(f"[DEBUG] Blink counted! total={blink_count}")
                    frames_closed = 0
                else:
                    # possible closed eyes -> increment closed counter
                    frames_closed += 1
                    print(f"[DEBUG] eyes not found -> frames_closed={frames_closed}")

                cv2.putText(frame, f"Blinks: {blink_count}/{REQUIRED_BLINKS} | Time left: {remaining}s",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # If required number of blinks achieved -> send face for recognition
                if blink_count >= REQUIRED_BLINKS:
                    print("[INFO] Liveness check passed. Sending cropped face to server...")
                    send_face_to_server(face_color)
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        cv2.imshow("Attendance (press 'q' to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Attendance cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_attendance()
