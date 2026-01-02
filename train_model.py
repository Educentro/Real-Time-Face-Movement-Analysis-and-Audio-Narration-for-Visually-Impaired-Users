import pickle

with open("features.pkl", "rb") as f:
    X, y = pickle.load(f)

print("Features loaded:", len(X))
print("Feature length of one sample:", len(X[0]))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train samples:", len(X_train))
print("Test samples:", len(X_test))
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

print("MLP model created")

model.fit(X_train, y_train)
print("Model training completed")

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

import pickle

with open("gesture_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("gesture_model.pkl saved")
