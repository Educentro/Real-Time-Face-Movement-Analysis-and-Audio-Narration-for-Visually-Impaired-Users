import pickle
import numpy as np
from ai.sign_labels import WORD_LABELS, ALPHABET_LABELS

WORD_MODEL_PATH = "models/word_gesture_model.pkl"
ALPHA_MODEL_PATH = "models/gesture_model.pkl"

WORD_THRESHOLD = 0.20



class GestureRouter:
    def __init__(self):
        with open(WORD_MODEL_PATH, "rb") as f:
            self.word_model = pickle.load(f)

        with open(ALPHA_MODEL_PATH, "rb") as f:
            self.alpha_model = pickle.load(f)


    def route(self, landmarks):
       landmarks = landmarks.reshape(1, -1)

       word_probs = self.word_model.predict_proba(landmarks)[0]
       word_id = int(np.argmax(word_probs))
       word_conf = float(word_probs[word_id])

       top5 = np.argsort(word_probs)[-5:][::-1]
       print("TOP 5 PREDICTIONS:")
       for idx in top5:
               print(f"  id={idx} label={WORD_LABELS[idx]} conf={word_probs[idx]:.2f}")

       if word_conf >= WORD_THRESHOLD:
          label = WORD_LABELS[word_id]
          return "WORD", label, word_conf

       return "NONE", None, word_conf
    


if __name__ == "__main__":
    dummy = np.random.rand(63)
    router = GestureRouter()
    print(router.route(dummy))
