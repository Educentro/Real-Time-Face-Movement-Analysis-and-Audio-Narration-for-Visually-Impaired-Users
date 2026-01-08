import pickle
import numpy as np
from ai.sign_labels import WORD_LABELS, ALPHABET_LABELS

WORD_MODEL_PATH = "models/word_gesture_model.pkl"
ALPHA_MODEL_PATH = "models/gesture_model.pkl"

WORD_THRESHOLD = 0.65
STABLE_FRAMES = 4


class GestureRouter:
    def __init__(self):
        with open(WORD_MODEL_PATH, "rb") as f:
            self.word_model = pickle.load(f)

        with open(ALPHA_MODEL_PATH, "rb") as f:
            self.alpha_model = pickle.load(f)

        self.word_counter = 0
        self.last_word_id = None

    def route(self, landmarks):        
        landmarks = landmarks.reshape(1, -1)

        # ----- WORD MODEL -----
        word_probs = self.word_model.predict_proba(landmarks)[0]
        word_id = int(np.argmax(word_probs))
        word_conf = word_probs[word_id]

        print(f"[DEBUG] word_id={word_id}, conf={word_conf:.2f}, counter={self.word_counter}")

        if word_conf >= WORD_THRESHOLD:
          if self.last_word_id == word_id:
            self.word_counter += 1
          else:
             self.word_counter = 1
             self.last_word_id = word_id
        else:
            self.word_counter = 0
            self.last_word_id = None

        if self.word_counter >= STABLE_FRAMES:
           label = WORD_LABELS[word_id]
           self.word_counter = 0
           self.last_word_id = None
           return "WORD", label

        return "NONE", None

if __name__ == "__main__":
    dummy = np.random.rand(63)
    router = GestureRouter()
    print(router.route(dummy))
