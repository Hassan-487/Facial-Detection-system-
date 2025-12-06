import cv2
import requests
import base64
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIG ---
SERVER_URL = "https://127.0.0.1:5000"
VERIFY_SSL = False
# Liveness settings
LIVENESS_TIMEOUT = 7
REQUIRED_BLINKS = 1
MIN_CLOSED_FRAMES = 2

# Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def send_to_server(frame, rect):
    (x, y, w, h) = rect
    face = frame[y:y+h, x:x+w]
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    _, buf = cv2.imencode('.jpg', rgb)
    b64 = base64.b64encode(buf).decode('utf-8')
    try:
        resp = requests.post(f"{SERVER_URL}/attend", json={'image': b64}, verify=VERIFY_SSL, timeout=5)
        return resp.json()
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def run_kiosk():
    cap = None
    
    # State
    session_active = False
    start_time = 0
    blink_count = 0
    closed_frames = 0
    msg = "System Starting..."
    msg_color = (255, 255, 255)
    last_poll = 0
    system_mode = "active"

    print("[INFO] Kiosk Started.")

    while True:
        if time.time() - last_poll > 2.0:
            try:
                r = requests.get(f"{SERVER_URL}/system/status", verify=VERIFY_SSL, timeout=1)
                system_mode = r.json().get('mode', 'active')
            except: 
                pass 
            last_poll = time.time()

        if system_mode == 'maintenance':
            if cap is not None:
                print("[INFO] Switching to Maintenance. Releasing camera.")
                cap.release()
                cap = None
                cv2.destroyAllWindows()
        
            time.sleep(1)
            continue

        if system_mode == 'active' and cap is None:
            print("[INFO] Switching to Active. Starting camera.")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("[ERROR] Camera failed.")
                cap = None
                time.sleep(2)
                continue

        ret, frame = cap.read()
        if not ret: continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw UI Background
        cv2.rectangle(frame, (0,0), (640, 60), (0,0,0), -1)

        if len(faces) == 0:
            msg = "Ready - Stand in front of camera"
            msg_color = (255, 255, 255)
            session_active = False # Reset if face lost
        
        elif len(faces) > 1:
            msg = "One person at a time!"
            msg_color = (0, 0, 255)
            session_active = False

        else:
            # Single Face Detected
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if not session_active:
                session_active = True
                start_time = time.time()
                blink_count = 0
                closed_frames = 0
                msg = "Please BLINK to verify..."
                msg_color = (0, 255, 255)

            # Timer
            elapsed = time.time() - start_time
            remaining = int(LIVENESS_TIMEOUT - elapsed)

            if elapsed > LIVENESS_TIMEOUT:
                msg = "Timeout. Try Again."
                msg_color = (0,0,255)
                # Display timeout, then reset
                cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, msg_color, 2)
                cv2.imshow("Attendance Kiosk", frame)
                cv2.waitKey(2000)
                session_active = False
                continue

            # Blink Detection
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            
            if len(eyes) == 0: closed_frames += 1
            else:
                if closed_frames >= MIN_CLOSED_FRAMES:
                    blink_count += 1
                closed_frames = 0
            
            cv2.putText(frame, f"Blinks: {blink_count} | Time: {remaining}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Check Success
            if blink_count >= REQUIRED_BLINKS:
                msg = "Processing..."
                cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                cv2.imshow("Attendance Kiosk", frame)
                cv2.waitKey(1)

                res = send_to_server(frame, faces[0])
                status = res.get('status')
                
                if status == 'success':
                    msg = f"Welcome, {res.get('name')}!"
                    msg_color = (0, 255, 0)
                elif status == 'unknown':
                    msg = "Not Registered"
                    msg_color = (0, 0, 255)
                else:
                    msg = "Error"
                    msg_color = (0, 0, 255)
                
                # Show Result
                cv2.rectangle(frame, (0,0), (640, 60), (0,0,0), -1)
                cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, msg_color, 2)
                cv2.imshow("Attendance Kiosk", frame)
                cv2.waitKey(3000) # Show result for 3 seconds
                session_active = False # Reset for next person
                continue

        cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, msg_color, 2)
        cv2.imshow("Attendance Kiosk", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cap: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_kiosk()
