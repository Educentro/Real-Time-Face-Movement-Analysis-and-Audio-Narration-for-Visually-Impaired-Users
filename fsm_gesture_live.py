import cv2
import mediapipe as mp
import math
import time
from collections import deque

gesture_buffer = deque(maxlen=10)
stable_gesture = None

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
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    wrist = landmarks[0]

    # FIST: all finger tips clearly BELOW wrist
    fist = (
        index_tip.y > wrist.y + 0.03 and
        middle_tip.y > wrist.y + 0.03 and
        ring_tip.y > wrist.y + 0.03 and
        pinky_tip.y > wrist.y + 0.03
    )

    if fist:
        return "FIST"

    # OPEN PALM: all finger tips clearly ABOVE MCP
    open_palm = (
        index_tip.y < index_mcp.y - 0.02 and
        middle_tip.y < middle_mcp.y - 0.02 and
        ring_tip.y < ring_mcp.y - 0.02 and
        pinky_tip.y < pinky_mcp.y - 0.02
    )

    if open_palm:
        return "OPEN_PALM"

    return None

    
def get_stable_gesture(buffer):
    if len(buffer) < buffer.maxlen:
        return None

    counts = {}
    for g in buffer:
        if g is None:
            return None
        counts[g] = counts.get(g, 0) + 1

    for gesture, count in counts.items():
        if count >= 7:
            return gesture

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
            gesture_buffer.append(gesture)

            #if gesture == last_gesture:
                #if current_time - gesture_hold_time >= GESTURE_STABLE_TIME:
                    #cv2.putText(frame, f"Gesture: {gesture}", (30, 60),
                                #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #else:
                #last_gesture = gesture
                #gesture_hold_time = current_time

            new_stable = get_stable_gesture(gesture_buffer)

            if new_stable and new_stable != stable_gesture:
              stable_gesture = new_stable

            if stable_gesture:
               cv2.putText(frame, f"Gesture: {stable_gesture}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Day 7 - FSM Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

