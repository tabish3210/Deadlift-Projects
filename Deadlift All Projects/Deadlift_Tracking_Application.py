import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tkinter as tk 
import customtkinter as ck 
import pandas as pd 
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
import pickle 
import mediapipe as mp
import cv2
from PIL import Image, ImageTk 
from landmarks import landmarks

# Initialize model
def initialize_model():
    model = DecisionTreeClassifier()
    # Sample data for demonstration
    X_train = np.random.rand(100, len(landmarks))
    y_train = ['up'] * 50 + ['down'] * 50
    model.fit(X_train, y_train)
    return model

# Create new model and save it
model = initialize_model()
with open('deadlift.pkl', 'wb') as f:
    pickle.dump(model, f)

window = tk.Tk()
window.geometry("480x700")
window.title("Deadlift Tracker") 
ck.set_appearance_mode("dark")

classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=1)
classLabel.configure(text='STAGE') 

counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
counterLabel.place(x=160, y=1)
counterLabel.configure(text='REPS') 

probLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
probLabel.place(x=300, y=1)
probLabel.configure(text='PROB') 

classBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
classBox.place(x=10, y=41)
classBox.configure(text='0') 

counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
counterBox.place(x=160, y=41)
counterBox.configure(text='0') 

probBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
probBox.place(x=300, y=41)
probBox.configure(text='0') 

def reset_counter(): 
    global counter
    counter = 0 

button = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, 
                     font=("Arial", 20), text_color="white", fg_color="blue")
button.place(x=10, y=600)

frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90) 
lmain = tk.Label(frame) 
lmain.place(x=0, y=0) 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5) 

# Initialize camera (try different indices if needed)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    for i in range(1, 5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break
    if not cap.isOpened():
        raise ValueError("Could not open camera. Please check camera connection.")

current_stage = ''
counter = 0 
bodylang_prob = np.array([0,0]) 
bodylang_class = '' 

def detect(): 
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob 

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        return
    
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = pose.process(image)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(106,13,173), thickness=4, circle_radius=5), 
            mp_drawing.DrawingSpec(color=(255,102,0), thickness=5, circle_radius=10)
        ) 

        try: 
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks) 
            bodylang_prob = model.predict_proba(X)[0]
            bodylang_class = model.predict(X)[0] 

            if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7: 
                current_stage = "down" 
            elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                current_stage = "up" 
                counter += 1 

        except Exception as e: 
            print(f"Error in pose processing: {e}") 

    img = cv2.resize(image[:, :, :], (460, 480))
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    
    counterBox.configure(text=str(counter))
    probBox.configure(text=f"{bodylang_prob[bodylang_prob.argmax()]:.2f}")
    classBox.configure(text=current_stage)
    
    lmain.after(10, detect)

detect() 
window.mainloop()

cap.release()
cv2.destroyAllWindows()