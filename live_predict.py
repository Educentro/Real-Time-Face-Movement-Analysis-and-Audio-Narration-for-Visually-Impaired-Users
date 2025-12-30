import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

model = load_model("model/asl_mlp.h5")
labels = np.load("model/labels.npy")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

def normalize(landmarks):
    landmarks = landmarks[:, :2]
    landmarks -= landmarks[0]
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks /= max_dist
    return landmarks.flatten()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        data = normalize(landmarks).reshape(1, -1)

        pred = model.predict(data, verbose=0)
        label = labels[np.argmax(pred)]

        cv2.putText(frame, label, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    cv2.imshow("ASL Live", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
