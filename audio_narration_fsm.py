"""
Live Gesture Interpretation + Audio Narration
ML-based (Word + Alphabet Models)
Priority Routing + FSM-controlled Audio
Status: Active Development
"""
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
import pyttsx3
import time
from ai.sign_definitions import SIGN_TO_TEXT

# FSM CONFIG
CONF_THRESHOLD = 0.75
STABLE_FRAMES = 4
COOLDOWN_FRAMES = 15

# FSM STATE
current_label = None
frame_counter = 0
cooldown_counter = 0
fsm_locked = False
WORD_BUFFER = deque(maxlen=3)

def fsm_update(predicted_label, confidence):
    global current_label, frame_counter, cooldown_counter, fsm_locked

    # Cooldown phase
    if fsm_locked:
        cooldown_counter += 1
        if cooldown_counter >= COOLDOWN_FRAMES:
            fsm_locked = False
            cooldown_counter = 0
        return None

    # Confidence gate
    required_conf = WORD_CONF_THRESHOLDS.get(
       predicted_label, CONF_THRESHOLD
)

    if confidence < required_conf:

        current_label = None
        frame_counter = 0
        return None

    # Gesture consistency
    if predicted_label == current_label:
        frame_counter += 1
    else:
        current_label = predicted_label
        frame_counter = 1

    # Confirm gesture
    if frame_counter >= STABLE_FRAMES:
        confirmed = current_label
        fsm_locked = True
        frame_counter = 0
        current_label = None
        return confirmed

    return None

WORD_CONF_THRESHOLDS = {
    "YES": 0.70,
    "NO": 0.70,
    "EAT": 0.70,
    "FATHER": 0.70,

    "HELLO": 0.60,
    "THANK_YOU": 0.60,
    "SORRY": 0.60,

    "WHAT": 0.55,
    "WHEN": 0.55,
    "YOU": 0.55,
    "ME": 0.55,
}



def speak(output_type, output_value):
    if output_type != "WORD":
        return

    text = SIGN_TO_TEXT.get(output_value)
    if not text:
        return
    
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def handle_name_question():
    words = list(WORD_BUFFER)

    if words[-3:] == ["WHAT", "NAME", "YOU"]:
        speak_sentence("What is your name?")
        WORD_BUFFER.clear()
        return True

    return False


# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
time.sleep(1)

if not cap.isOpened():
    print("❌ Camera unavailable. Exiting.")
    exit()

print("Camera opened:", cap.isOpened())


# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,   # 🔥 FORCE ONE HAND
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)



# ---------------- STATE ---------------- 
last_confirmed_word = None

def extract_live_landmarks(hand_landmarks):
    landmarks = []

    # use wrist as origin
    wrist = hand_landmarks.landmark[0]

    for lm in hand_landmarks.landmark:
        landmarks.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])

    landmarks = np.array(landmarks)

    # 🔥 L2 normalization (VERY IMPORTANT)
    norm = np.linalg.norm(landmarks)
    if norm > 0:
        landmarks = landmarks / norm

    return landmarks


from ai.gesture_router import GestureRouter
router = GestureRouter()


# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    current_time = time.time()

    display_text = "NONE : None"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )

        landmark_vector = extract_live_landmarks(hand_landmarks)

        output_type, output_value,confidence = router.route(landmark_vector)
        landmark_buffer.append(landmark_vector)
        smoothed = np.mean(landmark_buffer, axis=0)

        output_type, output_value, confidence = router.route(smoothed)

        if output_type == "WORD" and output_value:
            print("[DIRECT]", output_value, confidence)
            speak("WORD", output_value)

           
        else:
           # reset FSM when router says NOT a word
            fsm_update(None, 0.0)

        if last_confirmed_word:
            display_text = f"WORD : {last_confirmed_word}"

        print(f"[FSM] label={output_value}, conf={confidence:.2f}, counter={frame_counter}, locked={fsm_locked}")
        print(f"[ROUTER] {output_value} {confidence:.2f}")

    cv2.putText(
        frame,
        display_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Live Gesture System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
