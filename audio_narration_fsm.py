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

locked_word = None
LOCK_DURATION = 1.5  # seconds
lock_time = 0

last_spoken_word = None
last_spoken_time = 0
AUDIO_COOLDOWN = 2.5  # seconds


SIGN_TO_TEXT = {
    "HELLO": "The person is greeting you.",
    "HOW_ARE_YOU": "The person is asking how you are.",
    "NICE_TO_MEET_YOU": "The person is expressing pleasure in meeting you.",
    "WHAT_IS_YOUR_NAME": "The person is asking your name.",
    "MY_NAME_IS": "The person is introducing their name.",
    "THANK_YOU": "The person is thanking you.",
    "WELCOME": "The person is responding politely.",
    "GOODBYE": "The person is saying goodbye.",
    "SEE_YOU_LATER": "The person is saying they will see you later.",
    "MORNING": "The person is wishing you good morning.",
    "NIGHT": "The person is wishing you good night.",
    "AFTERNOON":"The person is wishing you good afternoon.",
    "PLEASE": "The person is requesting politely.",
    "SORRY": "The person is apologizing.",
    "EXCUSE_ME": "The person is asking for attention politely.",
    "YES": "The person is agreeing.",
    "NO": "The person is disagreeing.",
    "STOP": "The person is asking to stop.",
    "WAIT": "The person is asking to wait.",
    "HELP_ME": "The person is asking for help.",
    "IM_HUNGRY": "The person is saying they are hungry.",
    "IM_THIRSTY": "The person is saying they are thirsty.",
    "IM_TIRED": "The person is saying they are tired.",
    "IM_BUSY": "The person is saying they are busy.",
    "IM_BORED": "The person is saying they are bored.",
    "EAT": "The person is referring to eating.",
    "WHAT": "The person is asking what.",
    "WHEN": "The person is asking when.",
    "FATHER": "The person is referring to their father.",
    "MOTHER": "The person is referring to their mother.",
    "FRIEND": "The person is referring to a friend."
}

def speak(output_type, output_value):
    global last_spoken_word, last_spoken_time

    # Prevent repeating the same word
    if output_type == "WORD" and output_value == last_spoken_word:
        return

    if output_type == "WORD":
        text = SIGN_TO_TEXT.get(output_value)
    else:
        return

    if not text:
        return

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

    last_spoken_word = output_value
    last_spoken_time = time.time()


# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(1)

if not cap.isOpened():
    print("❌ Camera unavailable. Exiting.")
    exit()

print("Camera opened:", cap.isOpened())

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------- STATE ----------------
# FSM + audio will be activated after router stability checks 
STABLE_FRAMES_REQUIRED = 4

def extract_live_landmarks(hand_landmarks):
    landmarks = []
    wrist = hand_landmarks.landmark[0]

    for lm in hand_landmarks.landmark:
        landmarks.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])

    landmarks = np.array(landmarks)
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks = landmarks / max_val

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

        output_type, output_value = router.route(landmark_vector)

        # 🔒 LOCK WORD FOR STABILITY
        if output_type == "WORD" and output_value:
            locked_word = output_value
            lock_time = current_time

            # 🔊 AUDIO TRIGGER (ONCE)
            if (
                locked_word != last_spoken_word and
                (current_time - last_spoken_time) > AUDIO_COOLDOWN
            ):
                speak("WORD", locked_word)
                last_spoken_gesture = locked_word
                last_spoken_time = current_time

        if locked_word and (current_time - lock_time) < LOCK_DURATION:
            display_text = f"WORD : {locked_word}"

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
