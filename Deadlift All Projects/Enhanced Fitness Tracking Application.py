# File: utils.py
import math
import cv2
import numpy as np
from datetime import datetime
import json
import winsound
import sqlite3

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    
    return angle

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('fitness_tracker.db')
    c = conn.cursor()
    
    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS workouts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  exercise TEXT,
                  reps INTEGER,
                  form_score REAL,
                  duration INTEGER)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS form_analysis
                 (workout_id INTEGER,
                  back_angle REAL,
                  knee_angle REAL,
                  hip_angle REAL)''')
    
    conn.commit()
    conn.close()

def calculate_calories(exercise_duration, intensity=1.0):
    """Calculate calories burned based on exercise duration and intensity"""
    # Basic MET value for deadlifts
    met = 6.0 * intensity
    # Assuming average weight of 70kg
    weight_kg = 70
    
    # Calories = MET * weight (kg) * duration (hours)
    hours = exercise_duration / 3600  # Convert seconds to hours
    calories = met * weight_kg * hours
    
    return round(calories, 2)

# File: form_analyzer.py
class FormAnalyzer:
    def __init__(self):
        self.back_angle_threshold = 150
        self.knee_angle_threshold = 90
        self.hip_angle_threshold = 95

    def analyze_form(self, landmarks):
        """Analyze exercise form using pose landmarks"""
        if not landmarks:
            return None

        # Extract key points
        hip = [landmarks[24].x, landmarks[24].y]
        knee = [landmarks[26].x, landmarks[26].y]
        ankle = [landmarks[28].x, landmarks[28].y]
        shoulder = [landmarks[12].x, landmarks[12].y]

        # Calculate angles
        back_angle = calculate_angle(hip, shoulder, [shoulder[0], shoulder[1]-0.2])
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)

        form_analysis = {
            'back_straight': back_angle > self.back_angle_threshold,
            'knees_aligned': knee_angle > self.knee_angle_threshold,
            'hip_hinge': hip_angle > self.hip_angle_threshold,
            'angles': {
                'back': back_angle,
                'knee': knee_angle,
                'hip': hip_angle
            }
        }

        return form_analysis

    def get_feedback(self, form_analysis):
        """Generate feedback based on form analysis"""
        if not form_analysis:
            return "Cannot analyze form - please check camera position"

        feedback = []
        if not form_analysis['back_straight']:
            feedback.append("Keep your back straight!")
        if not form_analysis['knees_aligned']:
            feedback.append("Watch your knee alignment!")
        if not form_analysis['hip_hinge']:
            feedback.append("Hinge at your hips!")

        return " ".join(feedback) if feedback else "Good form!"

# File: app.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tkinter as tk 
import customtkinter as ck 
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import pickle 
import mediapipe as mp
import cv2
from PIL import Image, ImageTk 
from datetime import datetime, timedelta
from landmarks import landmarks
from utils import calculate_calories, init_database
from form_analyzer import FormAnalyzer

class FitnessTracker:
    def __init__(self):
        self.setup_gui()
        self.setup_model()
        self.setup_camera()
        self.setup_tracking()
        self.form_analyzer = FormAnalyzer()
        init_database()

    def setup_gui(self):
        self.window = tk.Tk()
        self.window.geometry("1000x800")
        self.window.title("Advanced Fitness Tracker")
        ck.set_appearance_mode("dark")

        # Main info panel
        self.setup_info_panel()
        
        # Form feedback panel
        self.setup_feedback_panel()
        
        # Statistics panel
        self.setup_stats_panel()
        
        # Camera frame
        self.setup_camera_frame()

    def setup_info_panel(self):
        # Exercise selection
        self.exercise_var = tk.StringVar(value="Deadlift")
        exercise_label = ck.CTkLabel(self.window, text="Exercise:", font=("Arial", 16))
        exercise_label.place(x=10, y=10)
        exercise_selector = ck.CTkComboBox(
            self.window,
            values=["Deadlift", "Squat", "Bench Press"],
            variable=self.exercise_var,
            font=("Arial", 16)
        )
        exercise_selector.place(x=100, y=10)

        # Main counters
        self.labels = {}
        self.boxes = {}
        
        labels_info = {
            'stage': ('STAGE', 10, 50),
            'reps': ('REPS', 160, 50),
            'form': ('FORM', 310, 50),
            'calories': ('CALORIES', 460, 50)
        }

        for key, (text, x, y) in labels_info.items():
            self.labels[key] = ck.CTkLabel(
                self.window, 
                height=40, 
                width=120, 
                font=("Arial", 20),
                text_color="black",
                text=text
            )
            self.labels[key].place(x=x, y=y)

            self.boxes[key] = ck.CTkLabel(
                self.window,
                height=40,
                width=120,
                font=("Arial", 20),
                text_color="white",
                fg_color="blue"
            )
            self.boxes[key].place(x=x, y=y+40)
            self.boxes[key].configure(text='0')

    def setup_feedback_panel(self):
        self.feedback_label = ck.CTkLabel(
            self.window,
            height=60,
            width=600,
            font=("Arial", 16),
            text_color="white",
            fg_color="gray20"
        )
        self.feedback_label.place(x=10, y=150)
        self.feedback_label.configure(text="Form feedback will appear here")

    def setup_stats_panel(self):
        self.stats_frame = ck.CTkFrame(self.window, width=200, height=400)
        self.stats_frame.place(x=780, y=220)
        
        stats_title = ck.CTkLabel(
            self.stats_frame,
            text="Workout Stats",
            font=("Arial", 16, "bold")
        )
        stats_title.pack(pady=10)

        self.stats_labels = {}
        stats = ['Total Reps', 'Avg Form Score', 'Time', 'Best Set']
        
        for stat in stats:
            frame = ck.CTkFrame(self.stats_frame)
            frame.pack(pady=5, padx=10, fill="x")
            
            ck.CTkLabel(frame, text=stat, font=("Arial", 14)).pack(side="left")
            self.stats_labels[stat] = ck.CTkLabel(frame, text="0", font=("Arial", 14))
            self.stats_labels[stat].pack(side="right")

    def setup_camera_frame(self):
        self.frame = tk.Frame(height=480, width=640)
        self.frame.place(x=10, y=220)
        self.lmain = tk.Label(self.frame)
        self.lmain.place(x=0, y=0)

    def setup_model(self):
        self.model = RandomForestClassifier(n_estimators=50)
        X_train = np.random.rand(100, len(landmarks))
        y_train = ['up'] * 50 + ['down'] * 50
        self.model.fit(X_train, y_train)

    def setup_camera(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5
        )

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open camera")

    def setup_tracking(self):
        self.current_stage = ''
        self.counter = 0
        self.bodylang_prob = np.array([0, 0])
        self.bodylang_class = ''
        self.start_time = datetime.now()
        self.calories_burned = 0
        self.form_scores = []

    def reset_counter(self):
        self.counter = 0
        self.start_time = datetime.now()
        self.calories_burned = 0
        self.form_scores = []

    def update_stats(self):
        elapsed = datetime.now() - self.start_time
        avg_form = np.mean(self.form_scores) if self.form_scores else 0
        
        self.stats_labels['Total Reps'].configure(text=str(self.counter))
        self.stats_labels['Avg Form Score'].configure(text=f"{avg_form:.1f}")
        self.stats_labels['Time'].configure(text=str(elapsed).split('.')[0])
        self.stats_labels['Best Set'].configure(text=str(max(self.form_scores) if self.form_scores else 0))

    def detect(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                self.mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10)
            )

            try:
                # Pose analysis
                form_analysis = self.form_analyzer.analyze_form(results.pose_landmarks.landmark)
                feedback = self.form_analyzer.get_feedback(form_analysis)
                self.feedback_label.configure(text=feedback)

                # Movement tracking
                row = np.array([[res.x, res.y, res.z, res.visibility] 
                              for res in results.pose_landmarks.landmark]).flatten().tolist()
                X = pd.DataFrame([row], columns=landmarks)
                bodylang_prob = self.model.predict_proba(X)[0]
                bodylang_class = self.model.predict(X)[0]

                # Rep counting
                if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                    self.current_stage = "down"
                elif (self.current_stage == "down" and 
                      bodylang_class == "up" and 
                      bodylang_prob[bodylang_prob.argmax()] > 0.7):
                    self.current_stage = "up"
                    self.counter += 1
                    
                    # Calculate form score
                    form_score = (form_analysis['angles']['back'] / 180.0 * 100 +
                                form_analysis['angles']['knee'] / 180.0 * 100 +
                                form_analysis['angles']['hip'] / 180.0 * 100) / 3
                    self.form_scores.append(form_score)

                    # Update calories
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    self.calories_burned = calculate_calories(elapsed)

                    # Play sound for completed rep
                    if form_score > 80:
                        winsound.Beep(1000, 100)
                    else:
                        winsound.Beep(500, 200)

            except Exception as e:
                print(f"Error in pose processing: {e}")

        # Update display
        img = cv2.resize(image, (640, 480))
        imgarr = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(imgarr)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)

        # Update labels
        self.boxes['stage'].configure(text=self.current_stage)
        self.boxes['reps'].configure(text=str(self.counter))
        self.boxes['form'].configure(text=f"{np.mean(self.form_scores):.1f}" if self.form_scores else "0")
        self.boxes['calories'].configure(text=f"{self.calories_burned:.1f}")

        # Update statistics
        self.update_stats()

        # Schedule next detection
        self.lmain.after(10, self.detect)

    def run(self):
        self.detect()
        self.window.mainloop()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FitnessTracker()
    app.run()