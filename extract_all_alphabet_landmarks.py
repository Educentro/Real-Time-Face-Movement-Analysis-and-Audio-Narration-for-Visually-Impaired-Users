import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tqdm import tqdm

if len(sys.argv) < 2:
    print("Usage: python extract_all_alphabet_landmarks.py <dataset_path>")
    sys.exit(1)

DATASET_DIR = sys.argv[1]
OUTPUT_FILE = "dataset_processed/X_alphabet.pkl"

if not os.path.exists("dataset_processed"):
    os.makedirs("dataset_processed")

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

print(f"Extracting landmarks from {DATASET_DIR}...")

# grassknoted/asl-alphabet dataset structure: asl_alphabet_train/asl_alphabet_train/A/
train_dir = os.path.join(DATASET_DIR, "asl_alphabet_train", "asl_alphabet_train")
if not os.path.exists(train_dir):
    # Try just asl_alphabet_train
    train_dir = os.path.join(DATASET_DIR, "asl_alphabet_train")
    if not os.path.exists(train_dir):
        # Maybe it's just the root
        train_dir = DATASET_DIR

# Get all labels
labels = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
# Filter only A-Z
labels = [l for l in labels if len(l) == 1 and l.isalpha()]

print(f"Found {len(labels)} alphabet labels: {labels}")

for label in tqdm(labels):
    label_path = os.path.join(train_dir, label)
    
    files = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.png'))]
    # To save time and avoid overfitting, take a subset (e.g., 500 images per class)
    # The original dataset has 3000 per class, which might take a long time to extract
    files = files[:50] 
    
    for img_name in files:
        img_path = os.path.join(label_path, img_name)
        features = extract_landmarks(img_path)
        if features is None:
            continue

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("\nExtraction Complete")
print(f"Total samples extracted: {len(X)}")
print(f"Feature shape: {X.shape}")

np.save("dataset_processed/X_alphabet.npy", X)
np.save("dataset_processed/y_alphabet.npy", y)

print("Saved to dataset_processed/X_alphabet.npy and y_alphabet.npy")
