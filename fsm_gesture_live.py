import cv2
import mediapipe as mp
import math
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

last_gesture = None
gesture_hold_time = 0
GESTURE_STABLE_TIME = 0.8  # seconds

def distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def detect_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    wrist = landmarks[0]

    fingers_open = (
        index_tip.y < wrist.y and
        middle_tip.y < wrist.y and
        ring_tip.y < wrist.y and
        pinky_tip.y < wrist.y
    )

    fist = (
        index_tip.y > wrist.y and
        middle_tip.y > wrist.y and
        ring_tip.y > wrist.y and
        pinky_tip.y > wrist.y
    )

    if fingers_open:
        return "OPEN_PALM"
    elif fist:
        return "FIST"
    else:
        return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            gesture = detect_gesture(hand_landmarks.landmark)

            if gesture == last_gesture:
                if current_time - gesture_hold_time >= GESTURE_STABLE_TIME:
                    cv2.putText(frame, f"Gesture: {gesture}", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                last_gesture = gesture
                gesture_hold_time = current_time

    cv2.imshow("Day 7 - FSM Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
