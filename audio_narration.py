import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
from tensorflow.keras.models import load_model
import os
import subprocess
import time
import threading

# ======================
# CONFIG
# ======================
CONF_THRESHOLD = 0.5
STABLE_FRAMES = 8
COOLDOWN_TIME = 1.0 # seconds
display_label = None

# ======================
# POWERSHELL TTS
# ======================

def speak_async(text):
    if not text:
        return

    def _run():
        command = f'''
        Add-Type -AssemblyName System.Speech;
        $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
        $speak.Speak("{text}");
        '''
        subprocess.run(
            ["powershell", "-Command", command],
            creationflags=0x08000000,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    threading.Thread(target=_run, daemon=True).start()


# ======================
# RESPONSES
# ======================
RESPONSES = {
    "HELLO": "The person is greeting you.",
    "YOU": "The person is referring to you.",
    "WHAT": "The person is asking what.",
    "NAME": "The person is introducing their name.",
    "THANK_YOU": "The person is thanking you.",
    "WELCOME": "The person is responding politely.",
    "GOODBYE": "The person is saying goodbye.",
    "GOOD_MORNING": "The person is wishing you good morning.",
    "GOOD_NIGHT": "The person is wishing you good night.",
    "PLEASE": "The person is requesting politely.",
    "SORRY": "The person is apologizing.",
    "YES": "The person is agreeing.",
    "NO": "The person is disagreeing.",
    "STOP": "The person is asking to stop.",
    "WAIT": "The person is asking to wait.",
    "HELP": "The person is asking for help.",
    "I'M_HUNGRY": "The person is saying they are hungry.",
    "I'M_THIRSTY": "The person is saying they are thirsty.",
    "I'M_TIRED": "The person is saying they are tired.",
    "I'M_BUSY": "The person is saying they are busy.",
    "I'M_BORED": "The person is saying they are bored.",
    "EAT": "The person is referring to eating.",
    "WHAT": "The person is asking what.",
    "WHEN": "The person is asking when.",
    "FATHER": "The person is referring to their father.",
    "MOTHER": "The person is referring to their mother.",
    "FRIEND": "The person is referring to a friend."
}
# ======================
# LOAD MODEL & LABELS
# ======================
model = load_model("frame_model.keras")
labels = sorted([
    d for d in os.listdir("dataset")
    if os.path.isdir(os.path.join("dataset", d))
])

# ======================
# MEDIAPIPE
# ======================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ======================
# STATE
# ======================
buffer = deque(maxlen=15)
last_spoken_label = None
last_spoken_time = 0
hand_present = False

# ======================
# CAMERA
# ======================
cap = cv2.VideoCapture(0)

print("\n🎤 PowerShell Audio Mode ACTIVE (Press Q to quit)\n")

# ======================
# MAIN LOOP
# ======================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    final_label = None
    now = time.time()

    # ---------- HAND DETECTED ----------
    if result.multi_hand_landmarks:
        hand_present = True
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:
            X = np.array(landmarks, dtype=np.float32).reshape(1, -1)
            pred = model.predict(X, verbose=0)[0]

            conf = float(np.max(pred))
            idx = int(np.argmax(pred))

            if conf >= CONF_THRESHOLD:
                buffer.append(labels[idx])

        if len(buffer) >= STABLE_FRAMES:
            label, count = Counter(buffer).most_common(1)[0]
            if count >= STABLE_FRAMES - 1:
                final_label = label
                buffer.clear() 
    # ---------- HAND REMOVED ----------
    else:
        if hand_present:
            buffer.clear()
            hand_present = False

    # ---------- AUDIO TRIGGER ----------
    if final_label:
        if final_label != last_spoken_label and (now - last_spoken_time) > COOLDOWN_TIME:
            clean = final_label.replace("_", " ").replace("'", "")
            sentence = clean

            if final_label in RESPONSES:
                sentence = RESPONSES[final_label]
            

            speak_async(sentence)
            print(f"🗣 {sentence}")
            display_label = final_label
            last_spoken_label = final_label
            last_spoken_time = now

    # ---------- UI (NO FACE BLOCK) ----------
    if display_label:
        cv2.putText(
        frame,
        display_label.replace("_", " "),
        (20, 60),          # top-left, no face block
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Sign Language Recognition (PowerShell Audio)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
