import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load landmark data
X = np.load("landmarks.npy")
y = np.load("labels.npy")

# One-hot encode labels
y = to_categorical(y, num_classes=29)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model architecture
model = Sequential([
    Dense(128, activation="relu", input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(29, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model
model.save("sign_language_model.h5")

print("✅ Model trained and saved successfully!")

