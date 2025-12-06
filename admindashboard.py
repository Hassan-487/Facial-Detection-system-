
import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import requests
import base64
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURATION ---
SERVER_URL = "https://127.0.0.1:5000"
VERIFY_SSL = False
ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("blue")  

class ModernAdminApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Secure Attendance - Command Center")
        self.geometry("1100x700")
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.cap = None
        self.video_loop_id = None
        self.current_frame = None

        # Start App
        self.show_login_screen()

    def clear_ui(self):
        """Removes all widgets from the window/content area"""
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_loop_id:
            self.after_cancel(self.video_loop_id)
            self.video_loop_id = None
            
        for widget in self.winfo_children():
            widget.destroy()

    def clear_content_area(self):
        """Clears only the right-side content area"""
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_loop_id:
            self.after_cancel(self.video_loop_id)
            self.video_loop_id = None

        for widget in self.right_frame.winfo_children():
            widget.destroy()

    def show_login_screen(self):
        self.clear_ui()
        
        # Background Frame (Centered Card)
        self.login_frame = ctk.CTkFrame(self, width=400, height=500, corner_radius=15)
        self.login_frame.place(relx=0.5, rely=0.5, anchor="center")

        # Title
        ctk.CTkLabel(self.login_frame, text="ADMIN LOGIN", font=("Roboto Medium", 24)).pack(pady=(40, 20))

        # Username
        self.user_entry = ctk.CTkEntry(self.login_frame, width=250, placeholder_text="Username")
        self.user_entry.pack(pady=10)

        # Password
        self.pass_entry = ctk.CTkEntry(self.login_frame, width=250, placeholder_text="Password", show="*")
        self.pass_entry.pack(pady=10)

        # Login Button
        ctk.CTkButton(self.login_frame, text="Login", width=250, command=self.perform_login).pack(pady=30)
        
        # Status Label
        self.status_label = ctk.CTkLabel(self.login_frame, text="", text_color="red")
        self.status_label.pack()

    def perform_login(self):
        u = self.user_entry.get()
        p = self.pass_entry.get()
        
        self.status_label.configure(text="Connecting...", text_color="yellow")
        self.update()
        
        try:
            resp = requests.post(f"{SERVER_URL}/admin/login", 
                                 json={'username': u, 'password': p}, 
                                 verify=VERIFY_SSL)
            if resp.status_code == 200:
                self.setup_dashboard_layout()
                self.show_welcome_screen()
            else:
                self.status_label.configure(text="Access Denied", text_color="red")
        except Exception as e:
            self.status_label.configure(text="Server Error", text_color="red")
            print(e)

    # ==========================
    # DASHBOARD LAYOUT
    # ==========================
    def setup_dashboard_layout(self):
        self.clear_ui()

        # --- Sidebar (Left) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1) # Spacer at bottom

        # Logo / Title
        ctk.CTkLabel(self.sidebar_frame, text="Admin Portal", font=("Roboto Medium", 20)).grid(row=0, column=0, padx=20, pady=20)

        # Buttons
        self.btn_enroll = ctk.CTkButton(self.sidebar_frame, text="Enroll User", command=self.init_enrollment, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.btn_enroll.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.btn_logs = ctk.CTkButton(self.sidebar_frame, text="View Logs", command=self.show_logs, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.btn_logs.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.btn_users = ctk.CTkButton(self.sidebar_frame, text="Manage Users", command=self.show_users, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"))
        self.btn_users.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        # Logout (Bottom)
        ctk.CTkButton(self.sidebar_frame, text="Log Out", command=self.show_login_screen, fg_color="#e74c3c", hover_color="#c0392b").grid(row=6, column=0, padx=20, pady=20, sticky="ew")

        # --- Content Area (Right) ---
        self.right_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

    def show_welcome_screen(self):
        self.clear_content_area()
        ctk.CTkLabel(self.right_frame, text="Welcome, Admin", font=("Roboto Medium", 32)).pack(pady=50)
        ctk.CTkLabel(self.right_frame, text="Select an option from the sidebar to begin.", font=("Roboto", 16)).pack()

    # ==========================
    # ENROLLMENT LOGIC
    # ==========================
    def init_enrollment(self):
        self.clear_content_area()
        ctk.CTkLabel(self.right_frame, text="Initializing Camera...", font=("Roboto", 18)).pack(pady=50)
        self.update()

        # Lock System
        try:
            requests.post(f"{SERVER_URL}/system/status", json={'mode': 'maintenance'}, verify=VERIFY_SSL)
        except:
            tk.messagebox.showerror("Error", "Could not connect to server")
            return

        # Wait for release (3 seconds for safety)
        self.after(3000, self.setup_enrollment_ui)

    def setup_enrollment_ui(self):
        self.clear_content_area()

        # Header
        header = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        header.pack(fill="x", pady=10)
        ctk.CTkButton(header, text="< Back", width=60, command=self.exit_enrollment, fg_color="gray").pack(side="left")
        ctk.CTkLabel(header, text="Enroll New User", font=("Roboto Medium", 22)).pack(side="left", padx=20)

        # Form
        self.name_entry = ctk.CTkEntry(self.right_frame, width=300, placeholder_text="Enter Full Name")
        self.name_entry.pack(pady=10)

        # Camera Box
        self.cam_frame = ctk.CTkFrame(self.right_frame, width=640, height=480)
        self.cam_frame.pack(pady=10)
        self.cam_label = ctk.CTkLabel(self.cam_frame, text="")
        self.cam_label.pack()

        # Capture Button
        ctk.CTkButton(self.right_frame, text="Capture & Save", command=self.capture_and_enroll, width=200, height=40).pack(pady=10)

        # Start Camera (DirectShow for Windows speed)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
        self.update_webcam()

    def update_webcam(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.last_frame = frame.copy()
                
                # Draw Box
                cv2.rectangle(frame, (200, 100), (440, 380), (0, 255, 0), 2)
                
                # Convert to Tkinter
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.cam_label.configure(image=imgtk)
                self.cam_label.image = imgtk
            
            self.video_loop_id = self.after(20, self.update_webcam)

    def capture_and_enroll(self):
        name = self.name_entry.get()
        if not name: return tk.messagebox.showwarning("Missing Info", "Please enter a name")

        # Crop Face
        h, w, _ = self.last_frame.shape
        crop = self.last_frame[100:380, 200:440]
        
        _, buf = cv2.imencode('.jpg', crop)
        b64 = base64.b64encode(buf).decode('utf-8')
        
        try:
            r = requests.post(f"{SERVER_URL}/enroll", json={'name': name, 'image': b64}, verify=VERIFY_SSL)
            if r.status_code == 201:
                tk.messagebox.showinfo("Success", f"User {name} Enrolled!")
                self.name_entry.delete(0, 'end')
            else:
                tk.messagebox.showerror("Error", r.json().get('message'))
        except Exception as e:
            tk.messagebox.showerror("Error", str(e))

    def exit_enrollment(self):
        # Stop Admin Camera
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.video_loop_id:
            self.after_cancel(self.video_loop_id)

        # Unlock System
        try:
            requests.post(f"{SERVER_URL}/system/status", json={'mode': 'active'}, verify=VERIFY_SSL)
        except: pass

        self.show_welcome_screen()

    # DATA TABLES (Logs & Users)
  
    def style_treeview(self):
        # Apply Dark Mode styling to standard Tkinter Treeview
        style = ttk.Style()
        style.theme_use("clam")
        
        style.configure("Treeview",
                        background="#2b2b2b",
                        foreground="white",
                        rowheight=30,
                        fieldbackground="#2b2b2b",
                        font=("Roboto", 12))
        
        style.map('Treeview', background=[('selected', '#1f538d')])
        
        style.configure("Treeview.Heading",
                        background="#1f538d",
                        foreground="white",
                        font=("Roboto Medium", 13))

    def show_logs(self):
        self.clear_content_area()
        self.style_treeview()
        
        ctk.CTkLabel(self.right_frame, text="Recent Attendance Logs", font=("Roboto Medium", 22)).pack(pady=20)

        # Table Container
        table_frame = ctk.CTkFrame(self.right_frame)
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Treeview
        cols = ('ID', 'Name', 'Timestamp')
        tree = ttk.Treeview(table_frame, columns=cols, show='headings')
        for col in cols: 
            tree.heading(col, text=col)
            tree.column(col, anchor="center")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)

        # Fetch Data
        try:
            logs = requests.get(f"{SERVER_URL}/admin/logs", verify=VERIFY_SSL).json()
            for l in logs:
                tree.insert("", "end", values=(l['id'], l['name'], l['timestamp']))
        except:
            tk.messagebox.showerror("Error", "Failed to fetch logs")

    def show_users(self):
        self.clear_content_area()
        self.style_treeview()
        
        ctk.CTkLabel(self.right_frame, text="User Management", font=("Roboto Medium", 22)).pack(pady=20)

        table_frame = ctk.CTkFrame(self.right_frame)
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)

        cols = ('ID', 'Name')
        tree = ttk.Treeview(table_frame, columns=cols, show='headings')
        tree.heading('ID', text='ID'); tree.heading('Name', text='Name')
        tree.column('ID', width=100, anchor="center")
        tree.column('Name', anchor="center")
        tree.pack(fill="both", expand=True)

        # Action Buttons
        btn_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        btn_frame.pack(pady=20)

        def fetch_data():
            for i in tree.get_children(): tree.delete(i)
            try:
                users = requests.get(f"{SERVER_URL}/admin/users", verify=VERIFY_SSL).json()
                for u in users: tree.insert("", "end", values=(u['id'], u['name']))
            except: pass

        def delete_selected():
            sel = tree.selection()
            if not sel: return tk.messagebox.showwarning("Select", "Please select a user")
            uid = tree.item(sel)['values'][0]
            
            if tk.messagebox.askyesno("Confirm", "Delete this user permanently?"):
                try:
                    requests.post(f"{SERVER_URL}/admin/delete_user", json={'user_id': uid}, verify=VERIFY_SSL)
                    fetch_data()
                except Exception as e:
                    tk.messagebox.showerror("Error", str(e))

        ctk.CTkButton(btn_frame, text="Refresh List", command=fetch_data, fg_color="gray").pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Delete Selected User", command=delete_selected, fg_color="#e74c3c", hover_color="#c0392b").pack(side="left", padx=10)

        fetch_data()

if __name__ == "__main__":
    app = ModernAdminApp()
    app.mainloop()