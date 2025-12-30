import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

X = np.load("dataset_processed/X.npy")
y = np.load("dataset_processed/y.npy")

le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

model = Sequential([
    Dense(128, activation="relu", input_shape=(42,)),
    Dense(64, activation="relu"),
    Dense(y_cat.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y_cat, epochs=25, batch_size=32, validation_split=0.2)

model.save("model/asl_mlp.h5")
np.save("model/labels.npy", le.classes_)

print("✅ Model trained & saved")
