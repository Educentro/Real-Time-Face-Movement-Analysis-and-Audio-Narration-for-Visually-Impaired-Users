import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
from tensorflow.keras.models import load_model
import os

# -------- LOAD MODEL --------
print("Loading model...")
model = load_model("frame_model.keras")

labels = sorted([
    d for d in os.listdir("dataset")
    if os.path.isdir(os.path.join("dataset", d))
])

print(f"Loaded {len(labels)} labels:")
for i, label in enumerate(labels):
    print(f"  {i}: {label}")

# -------- MEDIAPIPE SETUP --------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,           # ✅ FALSE for live video
    max_num_hands=1,
    min_detection_confidence=0.5,      # Lower for better detection
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# -------- SMOOTHING --------
pred_buffer = deque(maxlen=15)
CONF_THRESHOLD = 0.50
MIN_BUFFER_SIZE = 8
MIN_CONSENSUS = 5  # Need at least 5 same predictions

# -------- CAMERA --------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n🎥 Starting camera...")
print("📌 HOLD gesture steady for 1-2 seconds")
print("❌ Press 'q' to quit\n")

frame_skip = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 2nd frame for performance
    frame_skip += 1
    if frame_skip % 2 != 0:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    
    final_label = None
    confidence_display = 0.0
    predicted_label = "None"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        
        # Draw landmarks
        mp_draw.draw_landmarks(
            frame, 
            hand, 
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
        )

        # Extract landmarks
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:
            # Convert to numpy
            landmarks_array = np.array(landmarks, dtype=np.float32).reshape(1, -1)

            # Predict
            pred = model.predict(landmarks_array, verbose=0)[0]
            confidence = float(np.max(pred))
            pred_id = int(np.argmax(pred))
            predicted_label = labels[pred_id]
            confidence_display = confidence

            # Add to buffer if confidence is good
            if confidence > CONF_THRESHOLD:
                pred_buffer.append(predicted_label)

            # Get consensus from buffer
            if len(pred_buffer) >= MIN_BUFFER_SIZE:
                counter = Counter(pred_buffer)
                most_common = counter.most_common(1)[0]
                
                # Need strong consensus
                if most_common[1] >= MIN_CONSENSUS:
                    final_label = most_common[0]
        else:
            pred_buffer.clear()
    else:
        # No hand detected
        pred_buffer.clear()

    # -------- DISPLAY --------
    # Background overlay for better readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Raw prediction
    color = (0, 255, 255) if confidence_display > CONF_THRESHOLD else (100, 100, 100)
    cv2.putText(
        frame,
        f"Raw: {predicted_label} ({confidence_display:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    # Buffer status
    buffer_percent = (len(pred_buffer) / 15) * 100
    cv2.putText(
        frame,
        f"Buffer: {len(pred_buffer)}/15 ({buffer_percent:.0f}%)",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Final detected gesture
    if final_label:
        # Draw big label in center
        text_size = cv2.getTextSize(final_label, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = 400
        
        # Background for text
        cv2.rectangle(
            frame,
            (text_x - 20, text_y - text_size[1] - 20),
            (text_x + text_size[0] + 20, text_y + 20),
            (0, 255, 0),
            -1
        )
        
        # Text
        cv2.putText(
            frame,
            final_label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 0, 0),
            4
        )
    else:
        cv2.putText(
            frame,
            "HOLD GESTURE STEADY",
            (80, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 165, 255),
            3
        )

    # Instructions
    cv2.putText(
        frame,
        "Press 'Q' to quit",
        (10, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Closed successfully")

