import os
import numpy as np
import cv2
import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox, Frame, Label, Button, Entry, Text, StringVar, OptionMenu
from PIL import Image, ImageTk
import threading
import mediapipe as mp
import joblib
import pyttsx3
import time
from datetime import datetime

# Initialize mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Global variables
current_user = None
recording = False
camera_active = False

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "sign_language.db")

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Color scheme
PRIMARY_COLOR = "#4361EE"
SECONDARY_COLOR = "#3A0CA3"
BG_COLOR = "#F5F7FA"
TEXT_COLOR = "#1A1A2E"
ACCENT_COLOR = "#7209B7"
SUCCESS_COLOR = "#4CC9F0"
WARNING_COLOR = "#F72585"

# Styles and Fonts
LARGE_FONT = ("Helvetica", 16, "bold")
MEDIUM_FONT = ("Helvetica", 12)
SMALL_FONT = ("Helvetica", 10)

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        action_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    conn.commit()
    conn.close()

# MediaPipe functions
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    # Draw face landmarks
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )
    
    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )
    
    # Draw left hand landmarks
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    
    # Draw right hand landmarks
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    # Extract pose landmarks
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Extract face landmarks
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    # Extract left hand landmarks
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Extract right hand landmarks
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

# Custom Tkinter Components
class RoundedButton(Button):
    def __init__(self, master=None, **kwargs):
        Button.__init__(self, master, **kwargs)
        self.config(
            relief="flat",
            bd=0,
            bg=kwargs.get("bg", PRIMARY_COLOR),
            fg="white",
            activebackground=SECONDARY_COLOR,
            activeforeground="white",
            highlightthickness=0,
            padx=20,
            pady=10,
            font=MEDIUM_FONT
        )
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        
    def on_enter(self, e):
        self.config(bg=SECONDARY_COLOR)
        
    def on_leave(self, e):
        self.config(bg=PRIMARY_COLOR)

class EntryWithPlaceholder(Entry):
    def __init__(self, master=None, placeholder="", **kwargs):
        super().__init__(master, **kwargs)
        self.placeholder = placeholder
        self.placeholder_color = 'grey'
        self.default_fg_color = self['fg']
        self.bind("<FocusIn>", self.focus_in)
        self.bind("<FocusOut>", self.focus_out)
        self.show_placeholder()
        
    def focus_in(self, *args):
        if self['fg'] == self.placeholder_color:
            self.delete(0, 'end')
            self['fg'] = self.default_fg_color
            
    def focus_out(self, *args):
        if not self.get():
            self.show_placeholder()
            
    def show_placeholder(self):
        self.delete(0, 'end')
        self.insert(0, self.placeholder)
        self['fg'] = self.placeholder_color

# GUI classes
class SignLanguageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sign Language Action Detection")
        self.geometry("1000x620")
        self.configure(bg=BG_COLOR)
        self.resizable(False, False)
        
        # Initialize database
        init_db()
        
        # Initialize TTS engine
        self.tts_engine = pyttsx3.init()
        
        # Center the window
        self.center_window()
        
        # Create container
        container = Frame(self, bg=BG_COLOR)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        
        # Initialize all frames/pages
        for F in (LoginPage, RegisterPage, DashboardPage, AddActionPage):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            
        # Show the login page first
        self.show_frame(LoginPage)
    
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        
    def center_window(self):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width/2) - (1000/2)
        y = (screen_height/2) - (620/2)
        self.geometry(f'1000x620+{int(x)}+{int(y)}')

class LoginPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=BG_COLOR)
        self.controller = controller
        
        # Create a frame for the form
        form_frame = Frame(self, bg=BG_COLOR, padx=40, pady=40)
        form_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # App title
        title_label = Label(form_frame, text="Sign Language Detector", font=("Helvetica", 24, "bold"), bg=BG_COLOR, fg=SECONDARY_COLOR)
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 30))
        
        # Username
        username_label = Label(form_frame, text="Username", font=MEDIUM_FONT, bg=BG_COLOR, fg=TEXT_COLOR)
        username_label.grid(row=1, column=0, sticky="w", pady=(0, 5))
        
        self.username_entry = EntryWithPlaceholder(form_frame, placeholder="Enter your username", width=30, font=MEDIUM_FONT)
        self.username_entry.grid(row=2, column=0, columnspan=2, pady=(0, 15), ipady=8)
        
        # Password
        password_label = Label(form_frame, text="Password", font=MEDIUM_FONT, bg=BG_COLOR, fg=TEXT_COLOR)
        password_label.grid(row=3, column=0, sticky="w", pady=(0, 5))
        
        self.password_entry = EntryWithPlaceholder(form_frame, placeholder="Enter your password", width=30, font=MEDIUM_FONT, show="*")
        self.password_entry.grid(row=4, column=0, columnspan=2, pady=(0, 25), ipady=8)
        
        # Login button
        login_btn = RoundedButton(form_frame, text="Login", bg=PRIMARY_COLOR, width=15, command=self.login)
        login_btn.grid(row=5, column=0, pady=(0, 15))
        
        # Register link
        register_link = Label(form_frame, text="Don't have an account? Register", font=SMALL_FONT, bg=BG_COLOR, fg=ACCENT_COLOR, cursor="hand2")
        register_link.grid(row=6, column=0, columnspan=2, pady=(0, 15))
        register_link.bind("<Button-1>", lambda e: controller.show_frame(RegisterPage))
        
    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        # Check if username or password is empty or placeholder
        if username == "" or username == "Enter your username" or password == "" or password == "Enter your password":
            messagebox.showerror("Error", "Please enter both username and password")
            return
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT id, username, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        if user and user[2] == password:
            # Set current user
            global current_user
            current_user = {"id": user[0], "username": user[1]}
            
            # Show dashboard
            self.controller.show_frame(DashboardPage)
        else:
            messagebox.showerror("Error", "Invalid username or password")
            
        conn.close()

class RegisterPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=BG_COLOR)
        self.controller = controller
        
        # Create a frame for the form
        form_frame = Frame(self, bg=BG_COLOR, padx=40, pady=40)
        form_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # App title
        title_label = Label(form_frame, text="Create an Account", font=("Helvetica", 24, "bold"), bg=BG_COLOR, fg=SECONDARY_COLOR)
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 30))
        
        # Username
        username_label = Label(form_frame, text="Username", font=MEDIUM_FONT, bg=BG_COLOR, fg=TEXT_COLOR)
        username_label.grid(row=1, column=0, sticky="w", pady=(0, 5))
        
        self.username_entry = EntryWithPlaceholder(form_frame, placeholder="Choose a username", width=30, font=MEDIUM_FONT)
        self.username_entry.grid(row=2, column=0, columnspan=2, pady=(0, 15), ipady=8)
        
        # Password
        password_label = Label(form_frame, text="Password", font=MEDIUM_FONT, bg=BG_COLOR, fg=TEXT_COLOR)
        password_label.grid(row=3, column=0, sticky="w", pady=(0, 5))
        
        self.password_entry = EntryWithPlaceholder(form_frame, placeholder="Create a password", width=30, font=MEDIUM_FONT, show="*")
        self.password_entry.grid(row=4, column=0, columnspan=2, pady=(0, 25), ipady=8)
        
        # Register button
        register_btn = RoundedButton(form_frame, text="Register", bg=PRIMARY_COLOR, width=15, command=self.register)
        register_btn.grid(row=5, column=0, pady=(0, 15))
        
        # Login link
        login_link = Label(form_frame, text="Already have an account? Login", font=SMALL_FONT, bg=BG_COLOR, fg=ACCENT_COLOR, cursor="hand2")
        login_link.grid(row=6, column=0, columnspan=2, pady=(0, 15))
        login_link.bind("<Button-1>", lambda e: controller.show_frame(LoginPage))
        
    def register(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        # Check if username or password is empty or placeholder
        if username == "" or username == "Choose a username" or password == "" or password == "Create a password":
            messagebox.showerror("Error", "Please enter both username and password")
            return
        
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            # Insert new user
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            
            messagebox.showinfo("Success", "Account created successfully! You can now login.")
            self.controller.show_frame(LoginPage)
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Username already exists")
        finally:
            conn.close()

class DashboardPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=BG_COLOR)
        self.controller = controller
        self.cap = None
        self.camera_thread = None
        self.running = False
        self.model = None
        self.actions = []
        self.sequence_length = 30
        
        import pyttsx3
        self.tts_engine = pyttsx3.init()

        # Create header frame         
        header_frame = Frame(self, bg=PRIMARY_COLOR, height=60)         
        header_frame.pack(fill="x")                  

        # App title in header         
        title_label = Label(header_frame, text="Sign Language Action Detection", font=LARGE_FONT, bg=PRIMARY_COLOR, fg="white")         
        title_label.pack(side="left", padx=20, pady=10)                  

        # Username display         
        self.username_var = StringVar()         
        self.username_var.set("")         
        username_label = Label(header_frame, textvariable=self.username_var, font=MEDIUM_FONT, bg=PRIMARY_COLOR, fg="white")         
        username_label.pack(side="right", padx=20, pady=10)                  

        # Logout button         
        logout_btn = ttk.Button(header_frame, text="Logout", command=self.logout)         
        logout_btn.pack(side="right", padx=10, pady=10)                  

        # Create main content frame         
        content_frame = Frame(self, bg=BG_COLOR)         
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)                  

        # Create left frame for camera         
        left_frame = Frame(content_frame, bg=BG_COLOR, width=480, height=480)         
        left_frame.pack(side="left", padx=(0, 10), fill="both", expand=True)         
        left_frame.pack_propagate(False)                  

        # Camera label         
        self.camera_label = Label(left_frame, bg="black", width=480, height=360)         
        self.camera_label.pack(pady=(0, 10))                  

        # Create right frame for text output and controls         
        right_frame = Frame(content_frame, bg=BG_COLOR, width=400)         
        right_frame.pack(side="right", padx=(10, 0), fill="both", expand=True)         
        right_frame.pack_propagate(False)                  

        # Text output label         
        output_label = Label(right_frame, text="Detected Actions", font=MEDIUM_FONT, bg=BG_COLOR, fg=TEXT_COLOR)         
        output_label.pack(anchor="w", pady=(0, 5))                  

        # Text output box         
        self.output_text = Text(right_frame, width=40, height=15, font=MEDIUM_FONT, state="disabled")         
        self.output_text.pack(fill="both", expand=True, pady=(0, 10))                  

        # Control buttons frame         
        controls_frame = Frame(right_frame, bg=BG_COLOR)         
        controls_frame.pack(fill="x")                  

        # Add action button - Reduced width with smaller text and padding
        add_action_btn = RoundedButton(controls_frame, text="Add", bg=ACCENT_COLOR, 
                                    command=lambda: self.prepare_add_action(),
                                    width=8, height=1)         
        add_action_btn.pack(side="left", padx=2)                  

        # Start/Stop Camera button - Reduced width with smaller text
        self.start_camera_btn = RoundedButton(controls_frame, text="Camera", bg=PRIMARY_COLOR, 
                                            command=self.toggle_camera,
                                            width=8, height=1)         
        self.start_camera_btn.pack(side="left", padx=2)                  

        # Read text button - Reduced width with smaller text
        read_btn = RoundedButton(controls_frame, text="Read", bg=SUCCESS_COLOR, 
                                command=self.read_text,
                                width=8, height=1)         
        read_btn.pack(side="left", padx=2)                  

        # Clear button - Reduced width with smaller text
        clear_btn = RoundedButton(controls_frame, text="Clear", bg=WARNING_COLOR, 
                                command=self.clear_text,
                                width=8, height=1)         
        clear_btn.pack(side="left", padx=2)                  

        # Status bar         
        self.status_var = StringVar()         
        self.status_var.set("Ready")         
        status_bar = Label(self, textvariable=self.status_var, bd=1, relief="sunken", anchor="w")         
        status_bar.pack(side="bottom", fill="x")
        
        # Bind to frame visibility event
        self.bind("<Visibility>", self.on_visibility)
        
    def on_visibility(self, event):
        # Update username when dashboard becomes visible
        if current_user:
            self.username_var.set(f"Welcome, {current_user['username']}")
            
        # Load available actions
        self.load_actions()
        
    def load_actions(self):
        if not current_user:
            return
            
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get actions for current user
        cursor.execute("SELECT action_name FROM actions WHERE user_id = ?", (current_user["id"],))
        actions = cursor.fetchall()
        
        # Update actions list
        self.actions = [action[0] for action in actions]
        
        # Load model if actions exist
        if self.actions:
            model_path = os.path.join(MODELS_DIR, f"model_{current_user['id']}.pkl")
            if os.path.exists(model_path):
                try:
                    self.model = joblib.load(model_path)
                    self.status_var.set(f"Loaded model with actions: {', '.join(self.actions)}")
                except:
                    self.status_var.set("Failed to load model")
        
        conn.close()
    
    def prepare_add_action(self):
        # Stop camera if running
        if self.running:
            self.toggle_camera()
        
        # Switch to add action page
        self.controller.show_frame(AddActionPage)
        
    def toggle_camera(self):
        if not self.running:
            self.running = True
            self.start_camera_btn.config(text="Stop Camera")
            self.camera_thread = threading.Thread(target=self.update_camera, daemon=True)
            self.camera_thread.start()
        else:
            self.running = False
            self.start_camera_btn.config(text="Start Camera")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                self.camera_label.config(bg="black")  # Reset camera preview to black when stopped
    
    def update_camera(self):
        self.cap = cv2.VideoCapture(0)
        
        # Initialize holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            # Variables for prediction
            sequence = []
            predictions = []
            threshold = 0.7
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                
                # Draw landmarks
                draw_landmarks(image, results)
                
                # Make prediction if model is loaded
                if self.model is not None and self.actions:
                    # Extract keypoints
                    keypoints = extract_keypoints(results)
                    sequence.append(keypoints)
                    
                    # Keep only the last n frames
                    sequence = sequence[-self.sequence_length:]
                    
                    if len(sequence) == self.sequence_length:
                        # Make prediction
                        # Reshape the sequence data into a 2D array for the model
                        reshaped_data = np.array(sequence).reshape(1, -1)
                        
                        # Get the predicted class index
                        predicted_class_idx = self.model.predict(reshaped_data)[0]
                        
                        # Use predict_proba to get probabilities for confidence check
                        if hasattr(self.model, 'predict_proba'):
                            proba = self.model.predict_proba(reshaped_data)[0]
                            confidence = proba[predicted_class_idx] if isinstance(predicted_class_idx, (int, np.integer)) else max(proba)
                        else:
                            # If model doesn't support probabilities, assume high confidence
                            confidence = 1.0
                        
                        # Store the prediction
                        predictions.append(predicted_class_idx)
                        
                        # Keep only the last 5 predictions
                        predictions = predictions[-5:]
                        
                        if len(predictions) == 5:
                            # Get most common prediction using Counter
                            from collections import Counter
                            counter = Counter(predictions)
                            detected_action_idx = counter.most_common(1)[0][0]
                            detected_action = self.actions[detected_action_idx]
                            
                            # Check confidence
                            if confidence > threshold:
                                # Display prediction on frame
                                cv2.putText(image, detected_action, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                                
                                # Update text output
                                self.append_to_output(detected_action)
                
                # Convert to RGB for tkinter
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb_image = cv2.resize(rgb_image, (480, 360))
                
                # Use after method to update UI in the main thread
                self.update_camera_display(rgb_image)
                
                # Add a small delay to reduce CPU usage
                time.sleep(0.03)
                
            # Release camera when done
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                
        # Update UI to show camera is off
        self.after(0, lambda: self.camera_label.config(bg="black"))
    
    def append_to_output(self, text):
        # Update in main thread
        self.after(0, lambda: self._update_output(text))
    
    def _update_output(self, text):
        self.output_text.config(state="normal")
        
        # Get current text
        current_text = self.output_text.get("1.0", "end-1c")
        
        # Add word only if it's not the same as the last word
        words = current_text.split()
        if not words or words[-1] != text:
            if current_text and not current_text.endswith(" "):
                self.output_text.insert("end", " ")
            self.output_text.insert("end", text)
        
        self.output_text.config(state="disabled")
    
    def read_text(self):
        # Get text from output
        text = self.output_text.get("1.0", "end-1c").strip()
        
        if text:
            try:
                # Read text using TTS
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read text: {str(e)}")
        else:
            messagebox.showinfo("Info", "No text to read")
    
    def clear_text(self):
        # Clear output text
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.config(state="disabled")
    
    def logout(self):
        # Stop camera if running
        if self.running:
            self.toggle_camera()
            
        # Reset current user
        global current_user
        current_user = None
        
        # Go back to login page
        self.controller.show_frame(LoginPage)
    
    def update_camera_display(self, image):
        # This function updates the camera display in the main thread
        self.after(0, lambda: self._update_image(image))
    
    def _update_image(self, image):
        # Convert the image to ImageTk format
        pil_img = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=pil_img)
        
        # Update the label
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)

class AddActionPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent, bg=BG_COLOR)
        self.controller = controller
        self.cap = None
        self.camera_thread = None
        self.running = False
        self.recording = False
        self.action_name = ""
        self.sequence_data = []
        self.sequence_length = 30
        self.countdown = 0
        
        # Create header frame
        header_frame = Frame(self, bg=PRIMARY_COLOR, height=60)
        header_frame.pack(fill="x")
        
        # Back button in header
        back_btn = ttk.Button(header_frame, text="← Back to Dashboard", command=self.go_back)
        back_btn.pack(side="left", padx=20, pady=10)
        
        # Page title in header
        title_label = Label(header_frame, text="Add New Action", font=LARGE_FONT, bg=PRIMARY_COLOR, fg="white")
        title_label.pack(side="left", padx=20, pady=10)
        
        # Create main content frame
        content_frame = Frame(self, bg=BG_COLOR)
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Create form frame
        form_frame = Frame(content_frame, bg=BG_COLOR)
        form_frame.pack(fill="x", pady=(0, 20))
        
        # Action name label
        action_label = Label(form_frame, text="Action Name:", font=MEDIUM_FONT, bg=BG_COLOR, fg=TEXT_COLOR)
        action_label.pack(side="left", padx=(0, 10))
        
        # Action name entry
        self.action_entry = Entry(form_frame, font=MEDIUM_FONT, width=30)
        self.action_entry.pack(side="left", padx=(0, 20))
        
        # Record button
        self.record_btn = RoundedButton(form_frame, text="Start Recording", bg=WARNING_COLOR, command=self.toggle_recording)
        self.record_btn.pack(side="left", padx=(0, 10))
        
        # Train button
        self.train_btn = RoundedButton(form_frame, text="Train Model", bg=SUCCESS_COLOR, command=self.train_model)
        self.train_btn.pack(side="left")
        self.train_btn.config(state="disabled")
        
        # Create camera frame
        camera_frame = Frame(content_frame, bg=BG_COLOR)
        camera_frame.pack(fill="both", expand=True)
        
        # Camera label
        self.camera_label = Label(camera_frame, bg="black", width=640, height=480)
        self.camera_label.pack(expand=True)
        
        # Status label
        self.status_var = StringVar()
        self.status_var.set("Enter action name and click Start Recording")
        status_label = Label(content_frame, textvariable=self.status_var, font=MEDIUM_FONT, bg=BG_COLOR, fg=ACCENT_COLOR)
        status_label.pack(pady=10)
        
        # Bind to frame visibility event
        self.bind("<Visibility>", self.on_visibility)
        
    def on_visibility(self, event):
        # Start camera when page becomes visible
        if not self.running:
            self.running = True
            self.camera_thread = threading.Thread(target=self.update_camera)
            self.camera_thread.daemon = True
            self.camera_thread.start()
    
    def update_camera(self):
        self.cap = cv2.VideoCapture(0)
        
        # Initialize holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                
                # Draw landmarks
                draw_landmarks(image, results)
                
                # Handle recording
                if self.recording:
                    # Extract keypoints
                    keypoints = extract_keypoints(results)
                    
                    # Add to sequence data
                    self.sequence_data.append(keypoints)
                    
                    # Display recording status
                    cv2.putText(image, f"Recording: {len(self.sequence_data)}/{self.sequence_length*self.countdown}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    # Check if recording is complete
                    if len(self.sequence_data) == self.sequence_length * self.countdown:
                        self.recording = False
                        self.record_btn.config(text="Start Recording")
                        self.status_var.set(f"Recorded {len(self.sequence_data)} frames for '{self.action_name}'. You can now train the model.")
                        self.train_btn.config(state="normal")
                # Convert to RGB for tkinter
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Convert to ImageTk format
                img = Image.fromarray(image)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update camera label
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
                
            # Release camera when done
            if self.cap is not None:
                self.cap.release()
                self.cap = None
    
    def toggle_recording(self):
        if not self.recording:
            # Get action name
            self.action_name = self.action_entry.get().strip()
            
            # Validate action name
            if not self.action_name:
                messagebox.showerror("Error", "Please enter an action name")
                return
            
            # Set number of sequences to record
            self.countdown = 5  # Record 5 sequences
            
            # Reset sequence data
            self.sequence_data = []
            
            # Start recording
            self.recording = True
            self.record_btn.config(text="Stop Recording")
            self.status_var.set(f"Recording action '{self.action_name}'... Perform the action clearly")
            self.train_btn.config(state="disabled")
        else:
            # Stop recording
            self.recording = False
            self.record_btn.config(text="Start Recording")
            self.status_var.set(f"Recording stopped with {len(self.sequence_data)} frames. You can start again or train the model.")
            
            # Enable train button if we have enough data
            if len(self.sequence_data) >= self.sequence_length:
                self.train_btn.config(state="normal")
    
    def train_model(self):
        if not current_user:
            messagebox.showerror("Error", "You need to be logged in")
            return
            
        if not self.sequence_data:
            messagebox.showerror("Error", "No data recorded")
            return
            
        # Connect to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            # Check if action already exists
            cursor.execute("SELECT id FROM actions WHERE user_id = ? AND action_name = ?", 
                          (current_user["id"], self.action_name))
            existing = cursor.fetchone()
            
            if not existing:
                # Insert new action
                cursor.execute("INSERT INTO actions (user_id, action_name) VALUES (?, ?)", 
                             (current_user["id"], self.action_name))
                conn.commit()
            
            # Create action directory if it doesn't exist
            user_data_dir = os.path.join(DATA_DIR, str(current_user["id"]))
            if not os.path.exists(user_data_dir):
                os.makedirs(user_data_dir)
                
            # Save sequence data
            action_data_path = os.path.join(user_data_dir, f"{self.action_name}.npy")
            np.save(action_data_path, self.sequence_data)
            
            # Get all actions for the user
            cursor.execute("SELECT action_name FROM actions WHERE user_id = ?", (current_user["id"],))
            actions = [action[0] for action in cursor.fetchall()]
            
            # Prepare data for training
            X = []
            y = []
            
            # Load all action data
            for idx, action in enumerate(actions):
                action_data_path = os.path.join(user_data_dir, f"{action}.npy")
                if os.path.exists(action_data_path):
                    action_data = np.load(action_data_path)
                    
                    # Split into sequences
                    for i in range(0, len(action_data) - self.sequence_length + 1, self.sequence_length):
                        sequence = action_data[i:i+self.sequence_length]
                        X.append(sequence)
                        y.append(idx)
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Train model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X.reshape(X.shape[0], -1), y)
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f"model_{current_user['id']}.pkl")
            joblib.dump(model, model_path)
            
            messagebox.showinfo("Success", f"Model trained with {len(actions)} actions")
            self.status_var.set(f"Model trained with actions: {', '.join(actions)}")
            
            # Reset recording state
            self.sequence_data = []
            self.action_entry.delete(0, 'end')
            self.train_btn.config(state="disabled")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
        finally:
            conn.close()
    
    def go_back(self):
        # Stop camera
        self.running = False
        
        # Go back to dashboard
        self.controller.frames[DashboardPage].load_actions()  # Refresh actions
        self.controller.show_frame(DashboardPage)

if __name__ == "__main__":
    app = SignLanguageApp()
    app.mainloop()