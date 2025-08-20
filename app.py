# app_opencv.py
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from collections import deque
import math
import logging
import time

# -------------------------
# Helper functions
# -------------------------
def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def landmark_to_point(landmark, image_shape):
    h, w = image_shape[:2]
    return (int(landmark.x * w), int(landmark.y * h))

def eye_aspect_ratio(eye_pts):
    A = dist(eye_pts[1], eye_pts[5])
    B = dist(eye_pts[2], eye_pts[4])
    C = dist(eye_pts[0], eye_pts[3])
    if C == 0:
        return 0.0
    return (A+B)/(2.0*C)

def compute_movement(centers_deque):
    if len(centers_deque) < 2:
        return 0.0
    x0, y0 = centers_deque[0]
    x1, y1 = centers_deque[-1]
    return math.hypot(x1-x0, y1-y0)

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Emotion Detection (OpenCV)", layout="wide")
st.title("Emotion Detection + Anti-Spoofing (OpenCV Webcam)")

with st.sidebar:
    st.header("Settings")
    process_every = st.slider("Process every Nth frame", 1, 10, 3)
    enable_anti_spoof = st.checkbox("Enable anti-spoofing", value=True)
    debug_mode = st.checkbox("Debug visuals", value=False)

st.markdown("""
**Instructions:**\n
* Allow webcam access when prompted.\n
* Anti-spoofing = blink + small head movement heuristic.\n
* Increasing 'Process every Nth frame' reduces CPU usage.
""")

# -------------------------
# Initialize OpenCV and Mediapipe
# -------------------------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=False)
right_eye_idx = [33, 159, 158, 133, 153, 145]
left_eye_idx  = [362, 380, 374, 263, 386, 385]

# Anti-spoofing variables
ear_thresh = 0.20
blink_consec_frames = 3
blink_total = 0
blink_counter = 0
centers = deque(maxlen=10)
movement_pixel_threshold = 8
frame_count = 0
last_emotion = "N/A"
last_scores = {}

# Placeholder to show frames
frame_placeholder = st.empty()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame from webcam.")
            break

        frame_count +=1
        img = frame.copy()
        scale = 0.6
        small = cv2.resize(img, (0,0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda r: r[2]*r[3])
            x = int(fx/scale)
            y = int(fy/scale)
            w = int(fw/scale)
            h = int(fh/scale)
            x1, y1 = max(0,x), max(0,y)
            x2, y2 = min(img.shape[1], x+w), min(img.shape[0], y+h)
            face_roi = img[y1:y2, x1:x2]

            center = (int(x1+(x2-x1)/2), int(y1+(y2-y1)/2))
            centers.append(center)
            movement = compute_movement(centers)
            movement_liveness = movement > movement_pixel_threshold

            blink_liveness = False
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                try:
                    left_eye_pts = [landmark_to_point(lm[idx], img.shape) for idx in left_eye_idx]
                    right_eye_pts = [landmark_to_point(lm[idx], img.shape) for idx in right_eye_idx]
                    ear_left = eye_aspect_ratio(left_eye_pts)
                    ear_right = eye_aspect_ratio(right_eye_pts)
                    ear = (ear_left + ear_right)/2.0
                    if ear < ear_thresh:
                        blink_counter += 1
                    else:
                        if blink_counter >= blink_consec_frames:
                            blink_total +=1
                        blink_counter = 0
                    blink_liveness = blink_total>0
                    if debug_mode:
                        for p in left_eye_pts + right_eye_pts:
                            cv2.circle(img, p, 2, (0,255,255), -1)
                        cv2.putText(img, f"EAR:{ear:.2f}", (x1,y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0),2)
                except:
                    pass

            is_live = True
            if enable_anti_spoof:
                is_live = movement_liveness or blink_liveness

            # Emotion detection throttling
            if is_live and (frame_count % process_every == 0):
                try:
                    analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    last_emotion = analysis.get('dominant_emotion','N/A')
                    last_scores = analysis.get('emotion',{})
                except Exception as e:
                    logging.warning("DeepFace error: %s", e)

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img,f"Liveness: {'YES' if is_live else 'NO'}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,128,0) if is_live else (0,0,255),2)
            cv2.putText(img,f"Emotion: {last_emotion}",(x1,y2+25),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

            if last_scores:
                y_offset = y2 + 55
                for k,v in sorted(last_scores.items(), key=lambda item: -item[1])[:3]:
                    cv2.putText(img,f"{k}:{v:.1f}",(x1,y_offset),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),2)
                    y_offset += 20

        else:
            cv2.putText(img,"No face detected",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

        # Show frame in Streamlit
        frame_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

except Exception as e:
    st.error(f"Error: {str(e)}")

finally:
    cap.release()
    face_mesh.close()
