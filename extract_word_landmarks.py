import os
import cv2
import mediapipe as mp
import numpy as np
import json
from tqdm import tqdm

# Paths
DATASET_DIR = "Words_dataset"
OUTPUT_DIR = "dataset_processed/words"

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

X = []
y = []
label_map = {}

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

    landmarks = np.array(landmarks)
    landmarks = landmarks - np.mean(landmarks)
    landmarks = landmarks / (np.std(landmarks) + 1e-6)

    return landmarks


print("🚀 Extracting WORD gesture landmarks...")

label_id = 0

for label_name in os.listdir(DATASET_DIR):
    label_path = os.path.join(DATASET_DIR, label_name)

    if not os.path.isdir(label_path):
        continue
    print("Checking label folder:", label_name)
    label_map[label_id] = label_name

    for root, _, files in os.walk(label_path):
        for img_name in files:
            if not img_name.lower().endswith(('.jpg', '.png')):
                continue

            print("   🖼 Found image:", img_name)
            found_any_image = True

            img_path = os.path.join(root, img_name)
            features = extract_landmarks(img_path)

            if features is None:
                continue

            X.append(features)
            y.append(label_id)

    if not found_any_image:
        print("   ❌ No images found in this label")

    label_id += 1
    

X = np.array(X)
y = np.array(y)

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

print("✅ DONE")
print("Total samples:", len(X))
print("Feature shape:", X.shape)
print("Training labels order:")
print(sorted(os.listdir(DATASET_DIR)))

