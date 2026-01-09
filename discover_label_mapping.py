import numpy as np
import pickle
from ai.sign_labels import WORD_LABELS

with open("models/word_gesture_model.pkl", "rb") as f:
    model = pickle.load(f)
X = np.load("dataset_processed/words/X.npy")
y = np.load("dataset_processed/words/y.npy")

print("=== DISCOVERING LABEL MAPPING ===")

for label_id in range(29):
    
    idx = np.where(y == label_id)[0][0]
    sample = X[idx].reshape(1, -1)

    probs = model.predict_proba(sample)[0]
    predicted = probs.argmax()
    conf = probs[predicted]

    print(
        f"Numeric label {label_id} "
        f"-> predicted index {predicted} "
        f"({WORD_LABELS[predicted]}) "
        f"conf={conf:.2f}"
    )
