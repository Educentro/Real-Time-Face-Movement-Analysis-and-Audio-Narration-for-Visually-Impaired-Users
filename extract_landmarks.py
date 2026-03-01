import cv2
import mediapipe as mp
import numpy as np
import os

DATASET_DIR = "dataset"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6
)

X = []
y = []

labels = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

print("Labels:", labels)
print(f"Total classes: {len(labels)}")

for label_id, label in enumerate(labels):
    folder = os.path.join(DATASET_DIR, label)
    samples_count = 0

    for file in os.listdir(folder):
        if not file.lower().endswith((".mp4", ".avi", ".mov")):
            continue

        video_path = os.path.join(folder, file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Sample every 5th frame
            if frame_count % 5 != 0:
                continue

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            if not result.multi_hand_landmarks:
                continue

            hand = result.multi_hand_landmarks[0]

            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                X.append(landmarks)
                y.append(label_id)
                samples_count += 1

        cap.release()
    
    print(f"  {label}: {samples_count} samples")

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

np.save("X.npy", X)
np.save("y.npy", y)

print(f"\n✅ Saved X.npy and y.npy")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Total samples: {len(X)}")
print(f"Samples per class (avg): {len(X) / len(labels):.1f}")
