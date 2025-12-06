# Facial-Detection-system-

Here is a comprehensive **README.md** file tailored specifically for your project. You can copy-paste this directly into your GitHub repository.

-----

# Secure Facial Recognition Attendance System

A robust, secure, and user-friendly attendance system built with Python. It utilizes **DeepFace (FaceNet512)** for high-accuracy recognition, **Flask** for the backend API, and **CustomTkinter** for a modern Admin Dashboard.

The system features **Anti-Spoofing (Liveness Detection)**, **Encrypted Storage**, and **Anomaly Detection** via Email Alerts.

## 🚀 Key Features

  * **High-Accuracy Recognition:** Uses the `FaceNet512` model via DeepFace.
  * **Liveness Detection:** Prevents photo spoofing by requiring user eye blinks before recognition.
  * **Secure Storage:**
      * Face embeddings are encrypted using `Fernet` (symmetric encryption).
      * Admin passwords are hashed using `Scrypt` (via Werkzeug).
  * **Modern Admin Dashboard:** A GUI built with `CustomTkinter` to enroll users, manage the database, and view logs.
  * **Smart Hardware Locking:** Automatically manages camera access between the Kiosk and Admin dashboard to prevent conflicts.
  * **Anomaly Detection:** Automatically sends email alerts to the admin if attendance is marked outside of office hours (e.g., late night).

## 🛠️ Tech Stack

  * **Language:** Python 3.8+
  * **Backend:** Flask (REST API), SQLite3
  * **Computer Vision:** OpenCV, DeepFace
  * **GUI:** CustomTkinter (Modern UI wrapper for Tkinter)
  * **Security:** Cryptography (Fernet), Werkzeug (Hashing)
  * **Utilities:** NumPy, Pillow, Requests

## 📂 Project Structure

```text
Facial-Detection_Attendance-System/
│
├── server.py                # The central API server (Logic, DB, Email)
├── admin_dashboard.py       # GUI for Admin (Enrollment, Logs, Deletion)
├── attendance_kiosk.py      # Client script for marking attendance (Camera)
│
├── secure_attendance.db     # Created automatically on first run
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
```

## ⚙️ Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Hassan-487/Facial-Detection_Attendance-System.git
    cd Facial-Detection_Attendance-System
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment.

    ```bash
    pip install flask flask-cors deepface opencv-python cryptography requests pillow werkzeug customtkinter numpy
    ```

## 🚦 Usage Guide

To run the system, you will need to open **three separate terminals**.

### Step 1: Start the Server

This initializes the database and listens for requests.

```bash
python server.py
```

  * **Note:** On the first run, it creates a default admin account.
  * **Default Credentials:** Username: `admin` | Password: `admin123`

### Step 2: Open the Admin Dashboard

Use this to enroll new users.

```bash
python admin_dashboard.py
```

1.  Login using the default credentials.
2.  Go to **"Enroll User"**.
3.  Enter a name and capture the face.
      * *Note:* The system will automatically pause the Kiosk camera to free up hardware resources.

### Step 3: Run the Attendance Kiosk

This is the client-facing screen.

```bash
python attendance_kiosk.py
```

1.  The camera will open.
2.  Follow the on-screen instructions (e.g., "Please Blink").
3.  If recognized, attendance is marked. If not, it shows "Not Registered".

## 📧 Configuration (Optional)

To enable **Email Anomaly Detection** (alerts for after-hours attendance):

1.  Open `server.py`.
2.  Find the `EMAIL CONFIGURATION` section at the top.
3.  Update the fields:
    ```python
    SENDER_EMAIL = "your_email@gmail.com"
    SENDER_PASSWORD = "your_app_password"  # Generate via Google Account > Security > App Passwords
    ADMIN_EMAIL = "admin_email@gmail.com"
    ```
4.  Set your office hours:
    ```python
    OFFICE_START_HOUR = 9   # 9 AM
    OFFICE_END_HOUR = 18    # 6 PM
    ```

## ⚠️ Troubleshooting

**1. Camera Error: `cv2.error` or "Can't grab frame"**

  * Ensure you are not running `admin_dashboard.py` and `attendance_kiosk.py` camera functions at the exact same time. The "Smart Locking" feature usually handles this, but if it fails, close both scripts and restart the server.
  * If on Windows, the code uses `cv2.CAP_DSHOW` for better compatibility.

**2. Email not sending**

  * Ensure you are using a **Google App Password**, not your regular login password.
  * Check your internet connection.

**3. "Face not detected" during enrollment**

  * Ensure good lighting.
  * Make sure only **one** face is visible in the frame.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome\! Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/Hassan-487/Facial-Detection_Attendance-System/issues).

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
