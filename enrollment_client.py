
import cv2
import requests
import base64

SERVER_URL = "https://127.0.0.1:5000/enroll"
REQUESTS_VERIFY = False  # self-signed SSL

if not REQUESTS_VERIFY:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def enroll_user():
    name = input("Enter the user's name: ")
    if not name:
        print("Enrollment cancelled.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("\n--- Webcam Activated ---")
    print("Look at the camera. A green box will appear.")
    print("Press 'c' to capture your image, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        capture_ready = False
        face_crop = None

        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            # Draw a green rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Get the cropped face
            face_crop = frame[y:y+h, x:x+w]
            capture_ready = True
        elif len(faces) > 1:
            cv2.putText(frame, "Multiple Faces Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Enrollment - Press 'c' to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if not capture_ready:
                print("Cannot capture. Please ensure one face is clearly visible.")
                continue

            # Encode the CROPPED face image (BGR) as JPEG
            _, img_encoded = cv2.imencode('.jpg', face_crop)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')

            # Send to server
            print("Sending image to server for enrollment...")
            payload = {'name': name, 'image': img_base64}
            try:
                response = requests.post(SERVER_URL, json=payload, verify=REQUESTS_VERIFY)
                print("Server Response:", response.json())
            except Exception as e:
                print(f"Error connecting to server: {e}")
            break

        elif key == ord('q'):
            print("Enrollment cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    enroll_user()
