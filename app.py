# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from collections import deque
import math
import logging
import av

# -------------------------
# Helper utilities
# -------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def landmark_to_point(landmark, image_shape):
    h, w = image_shape[:2]
    return (int(landmark.x * w), int(landmark.y * h))

def eye_aspect_ratio(eye_pts):
    A = dist(eye_pts[1], eye_pts[5])
    B = dist(eye_pts[2], eye_pts[4])
    C = dist(eye_pts[0], eye_pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def compute_movement(centers_deque):
    if len(centers_deque) < 2:
        return 0.0
    x0, y0 = centers_deque[0]
    x1, y1 = centers_deque[-1]
    return math.hypot(x1 - x0, y1 - y0)

# -------------------------
# Video transformer
# -------------------------
class EmotionTransformer(VideoTransformerBase):
    def __init__(self, process_every=3, enable_anti_spoof=True, debug=False):
        self.debug = debug
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                               max_num_faces=1,
                                               refine_landmarks=False,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5)
        self.right_eye_idx = [33, 159, 158, 133, 153, 145]
        self.left_eye_idx  = [362, 380, 374, 263, 386, 385]
        self.ear_thresh = 0.20
        self.blink_consec_frames = 3
        self.blink_total = 0
        self.blink_counter = 0
        self.centers = deque(maxlen=10)
        self.movement_pixel_threshold = 8
        self.process_every = int(process_every)
        self.frame_count = 0
        self.last_emotion = "N/A"
        self.last_scores = {}
        self.enable_anti_spoof = bool(enable_anti_spoof)
        self.last_error = None

    def __del__(self):
        try:
            self.face_mesh.close()
        except Exception:
            pass

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        img = self.process_frame(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def process_frame(self, img: np.ndarray) -> np.ndarray:
        orig = img.copy()
        try:
            scale = 0.6
            small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray,
                                                       scaleFactor=1.1,
                                                       minNeighbors=5,
                                                       minSize=(60, 60))
            if len(faces) == 0:
                cv2.putText(img, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                return img

            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            x = int(fx / scale)
            y = int(fy / scale)
            w = int(fw / scale)
            h = int(fh / scale)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
            face_roi = img[y1:y2, x1:x2]

            center = (int(x1 + (x2 - x1)/2), int(y1 + (y2 - y1)/2))
            self.centers.append(center)
            movement = compute_movement(self.centers)
            movement_liveness = movement > self.movement_pixel_threshold

            blink_liveness = False
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                try:
                    left_eye_pts = [landmark_to_point(lm[idx], img.shape) for idx in self.left_eye_idx]
                    right_eye_pts = [landmark_to_point(lm[idx], img.shape) for idx in self.right_eye_idx]
                    ear_left = eye_aspect_ratio(left_eye_pts)
                    ear_right = eye_aspect_ratio(right_eye_pts)
                    ear = (ear_left + ear_right)/2.0

                    if ear < self.ear_thresh:
                        self.blink_counter += 1
                    else:
                        if self.blink_counter >= self.blink_consec_frames:
                            self.blink_total += 1
                        self.blink_counter = 0

                    blink_liveness = self.blink_total > 0

                    if self.debug:
                        for p in left_eye_pts + right_eye_pts:
                            cv2.circle(img, p, 2, (0, 255, 255), -1)
                        cv2.putText(img, f"EAR:{ear:.2f}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                except Exception as ex:
                    logging.debug("Landmark parsing error: %s", ex)

            is_live = True
            if self.enable_anti_spoof:
                is_live = movement_liveness or blink_liveness

            if is_live and (self.frame_count % self.process_every == 0):
                try:
                    analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    self.last_emotion = analysis.get('dominant_emotion', 'N/A')
                    self.last_scores = analysis.get('emotion', {})
                except Exception as e:
                    self.last_error = str(e)
                    logging.warning("DeepFace error: %s", e)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Liveness: {'YES' if is_live else 'NO'}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,128,0) if is_live else (0,0,255), 2)
            cv2.putText(img, f"Emotion: {self.last_emotion}", (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if self.last_scores:
                y_offset = y2 + 55
                for k, v in sorted(self.last_scores.items(), key=lambda item: -item[1])[:3]:
                    cv2.putText(img, f"{k}: {v:.1f}", (x1, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
                    y_offset += 20

            if self.debug:
                cv2.putText(img, f"Movement:{movement:.1f}", (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(img, f"Blinks:{self.blink_total}", (200, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        except Exception as e:
            logging.exception("Frame processing error: %s", e)
            cv2.putText(orig, f"Error: {str(e)[:80]}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            return orig

        return img

# -------------------------
# Streamlit app
# -------------------------
def main():
    st.set_page_config(page_title="Emotion Detection (DeepFace) + Anti-Spoofing", layout="wide")
    st.title("Emotion Detection with DeepFace + Anti-Spoofing (Webcam)")

    with st.sidebar:
        st.header("Settings")
        process_every = st.slider("Process every Nth frame (CPU saver)", 1, 10, 3)
        enable_anti_spoof = st.checkbox("Enable anti-spoofing (blink/movement)", value=True)
        debug_mode = st.checkbox("Debug visuals (landmarks, EAR)", value=False)
        st.markdown("**Notes:** Increasing `N` reduces CPU usage but makes detection less responsive.")

    st.markdown("""
    **How it works:**\n
    * Face detection → crop face → DeepFace.analyze(emotion)\n
    * Anti-spoofing = blink detection + small head movement heuristic\n
    * Streamlit + streamlit-webrtc used for webcam streaming and overlaying results
    """)

    webrtc_streamer(
        key="emotion",
        video_transformer_factory=lambda: EmotionTransformer(
            process_every=process_every,
            enable_anti_spoof=enable_anti_spoof,
            debug=debug_mode
        ),
        media_stream_constraints={"video": True, "audio": False}
    )

if __name__ == "__main__":
    main()
