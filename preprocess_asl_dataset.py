import os
import cv2
import numpy as np
import mediapipe as mp

DATASET_PATH = r"C:\Users\mawli\Downloads\dataset_images\Data"
OUTPUT_PATH = "dataset_processed"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

X, y = [], []

def normalize(landmarks):
    landmarks = landmarks[:, :2]
    landmarks = landmarks - landmarks[0]
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    if max_dist > 0:
        landmarks /= max_dist
    return landmarks.flatten()

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(label_path):
        continue

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if not result.multi_hand_landmarks:
            continue

        hand = result.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        X.append(normalize(landmarks))
        y.append(label)

os.makedirs(OUTPUT_PATH, exist_ok=True)
np.save(f"{OUTPUT_PATH}/X.npy", np.array(X))
np.save(f"{OUTPUT_PATH}/y.npy", np.array(y))

print("✅ Landmark extraction completed")
print("Samples:", len(X))
