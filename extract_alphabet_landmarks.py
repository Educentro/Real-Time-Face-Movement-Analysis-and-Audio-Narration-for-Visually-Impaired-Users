import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tqdm import tqdm

# Paths
DATASET_DIR = "dataset"
OUTPUT_FILE = "features.pkl"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

X = []
y = []

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    # Normalize
    landmarks = np.array(landmarks)
    landmarks = landmarks - np.mean(landmarks)
    landmarks = landmarks / (np.std(landmarks) + 1e-6)

    return landmarks


print("🚀 Extracting landmarks...")

for label in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(label_path):
        continue

    for root, _, files in os.walk(label_path):

     for img_name in files:
        if not img_name.lower().endswith(('.jpg', '.png')):
            continue

        img_path = os.path.join(root, img_name)

        features = extract_landmarks(img_path)
        if features is None:
            continue

        X.append(features)
        y.append(label)


X = np.array(X)
y = np.array(y)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump((X, y), f)

print("✅ DONE")
print(f"Total samples: {len(X)}")
print(f"Feature shape: {X.shape}")
