import pickle
import numpy as np

WORD_MODEL_PATH = "models/word_gesture_model.pkl"
ALPHA_MODEL_PATH = "models/gesture_model.pkl"

WORD_THRESHOLD = 0.85
STABLE_FRAMES = 6


class GestureRouter:
    def __init__(self):
        with open(WORD_MODEL_PATH, "rb") as f:
            self.word_model = pickle.load(f)

        with open(ALPHA_MODEL_PATH, "rb") as f:
            self.alpha_model = pickle.load(f)

        self.word_counter = 0

    def route(self, landmarks):
        """
        landmarks: np.array of shape (63,)
        returns: ("WORD", class_id) or ("ALPHABET", class_id)
        """

        landmarks = landmarks.reshape(1, -1)

        # Word prediction
        word_probs = self.word_model.predict_proba(landmarks)[0]
        word_id = np.argmax(word_probs)
        word_conf = word_probs[word_id]

        if word_conf >= WORD_THRESHOLD:
            self.word_counter += 1
        else:
            self.word_counter = 0

        if self.word_counter >= STABLE_FRAMES:
            self.word_counter = 0
            return "WORD", word_id

        # Fallback to alphabet
        alpha_id = self.alpha_model.predict(landmarks)[0]
        return "ALPHABET", alpha_id

if __name__ == "__main__":
    dummy = np.random.rand(63)
    router = GestureRouter()
    print(router.route(dummy))
