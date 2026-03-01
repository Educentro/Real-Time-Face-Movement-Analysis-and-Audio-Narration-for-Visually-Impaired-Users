import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import time
import joblib

MODEL_PATH = "models/word_gesture_model.pkl"
model = joblib.load(MODEL_PATH)

print("✅ Gesture model loaded")
def landmarks_to_feature_vector(landmarks):
    return np.array([[coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]])

# ---------------- CONFIG ----------------
MODEL_PATH = "models/word_gesture_model.pkl"
CONF_THRESHOLD = 0.75
STABLE_FRAMES = 3
COOLDOWN_TIME = 1.0  # seconds
# ----------------------------------------

# Load model
model = joblib.load(MODEL_PATH)

# TTS
engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

# Stability state
last_gesture = None
same_count = 0
last_spoken_time = 0

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = None
    conf = 0.0

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark

        mp_draw.draw_landmarks(
            frame,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS
        )

        # Extract features
        features = []
        for p in lm:
            features.extend([p.x, p.y, p.z])

        features = np.array(features).reshape(1, -1)

        # ML prediction
        probs = model.predict_proba(features)[0]
        pred = model.classes_[np.argmax(probs)]
        conf = np.max(probs)

        if conf > CONF_THRESHOLD:
            gesture = pred
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        X_live = landmarks_to_feature_vector(hand_landmarks.landmark)
        pred = model.predict(X_live)[0]
        confidence = np.max(model.predict_proba(X_live))

        if confidence > 0.7:   # safe threshold
          gesture = pred
        else:
          gesture = None

    # ---------------- STABILITY ----------------
    if gesture is None:
        same_count = 0
        last_gesture = None
    else:
        if gesture == last_gesture:
            same_count += 1
        else:
            same_count = 1
            last_gesture = gesture

        if same_count == STABLE_FRAMES:
            now = time.time()
            if now - last_spoken_time > COOLDOWN_TIME:
                print(f"DETECTED: {gesture} ({conf:.2f})")
                speak(gesture)
                last_spoken_time = now

    # ---------------- UI ----------------
    if gesture:
        cv2.putText(
            frame,
            f"{gesture} ({conf:.2f})",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("FSM Gesture Live (ML Powered)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
