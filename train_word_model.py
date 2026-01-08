import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load WORD landmark data
X = np.load("dataset_processed/words/X.npy")
y = np.load("dataset_processed/words/y.npy")

print("✅ Word features loaded")
print("Total samples:", len(X))
print("Feature length:", X.shape[1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))

# MLP model (same architecture – proven to work)
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

print("🧠 Training word gesture MLP...")
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("🎯 Word model accuracy:", accuracy)

# Save WORD model (IMPORTANT: new name)
with open("models/word_gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ word_gesture_model.pkl saved")
