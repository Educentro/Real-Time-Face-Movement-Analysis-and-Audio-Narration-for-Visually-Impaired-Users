import cv2
import mediapipe as mp
from collections import deque
import pyttsx3
import time

def speak(text):
    global last_audio_time

    current_time = time.time()
    if current_time - last_audio_time < AUDIO_COOLDOWN:
        return

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

    last_audio_time = current_time


last_audio_time = 0
AUDIO_COOLDOWN = 1.2  # seconds

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

    if gesture:
        gesture_buffer.append(gesture)

        if gesture_buffer.count(gesture) >= 4 and gesture != previous_gesture:
            speak(gesture)
            previous_gesture = gesture

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
        previous_gesture = None

    cv2.imshow("Day 4 - Gesture to Audio", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
