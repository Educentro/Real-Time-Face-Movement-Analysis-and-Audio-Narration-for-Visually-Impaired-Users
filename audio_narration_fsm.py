"""
Day 5 – FINAL FROZEN VERSION
Gesture Detection + Meaning + Audio Narration
Rule-based FSM (No ML, No Dataset)
Status: Stable and Demo-Ready
"""
import cv2
import mediapipe as mp
from collections import deque
import pyttsx3
import time
from ai.sign_definitions import SIGN_TO_TEXT

last_spoken_gesture = None
last_spoken_time = 0
AUDIO_COOLDOWN = 2.5  # seconds

ALLOWED_GESTURES = set(SIGN_TO_TEXT.keys())

SIGN_TO_TEXT = {
    "HELLO": "The person is greeting you.",
    "HOW_ARE_YOU": "The person is asking how you are.",
    "NICE_TO_MEET_YOU": "The person is expressing pleasure in meeting you.",
    "WHAT_IS_YOUR_NAME": "The person is asking your name.",
    "MY_NAME_IS": "The person is introducing their name.",
    "THANK_YOU": "The person is thanking you.",
    "YOU_ARE_WELCOME": "The person is responding politely.",
    "GOODBYE": "The person is saying goodbye.",
    "SEE_YOU_LATER": "The person is saying they will see you later.",
    "GOOD_MORNING": "The person is wishing you good morning.",
    "GOOD_NIGHT": "The person is wishing you good night.",
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

def speak(gesture_label):
    global last_spoken_gesture

    if gesture_label == last_spoken_gesture:
        return

    text = SIGN_TO_TEXT.get(gesture_label)
    if not text:
        return

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

    last_spoken_gesture = gesture_label



# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
gesture_buffer = deque(maxlen=6)
previous_gesture = None

# ---------------- LOGIC ----------------
def fingers_up(hand):
    tips = [4, 8, 12, 16, 20]
    return [
        hand.landmark[tip].y < hand.landmark[tip - 2].y
        for tip in tips
    ]

def detect_gesture(hand_list):

    # ---------- ONE HAND ----------
    if len(hand_list) == 1:
        h = fingers_up(hand_list[0])

        # HELLO ✋
        if all(h):
            return "Hello"

        # BYE ✊ (ignore thumb)
        if not any(h[1:]):
            return "Bye"

    # ---------- TWO HANDS ----------
    if len(hand_list) == 2:
        h1 = fingers_up(hand_list[0])
        h2 = fingers_up(hand_list[1])

        # THANK YOU 👍👍 (thumbs up, others down)
        if (
            h1[0] and h2[0] and
            not any(h1[1:]) and
            not any(h2[1:])
        ):
            return "Thank you"

        # HOW ARE YOU ☝️☝️ (index only)
        if (
            h1[1] and not any(h1[2:]) and
            h2[1] and not any(h2[2:])
        ):
            return "How are you"

        # I AM TIRED ✊✊ (fists, thumb ignored)
        if (
            not any(h1[1:]) and
            not any(h2[1:])
        ):
            return "I am tired"

    return None

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = None

    if results.multi_hand_landmarks:
        gesture = detect_gesture(results.multi_hand_landmarks)

        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

    current_time = time.time()

    if gesture:
        gesture_buffer.append(gesture)

        if (
            gesture_buffer.count(gesture) >= 4 and
            gesture != last_spoken_gesture and
            current_time - last_spoken_time > AUDIO_COOLDOWN
        ):
            speak(gesture)
            last_spoken_time = current_time

        cv2.putText(
            frame,
            gesture,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:
        gesture_buffer.clear()

    cv2.imshow("Day 5 - Gesture to Audio", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
