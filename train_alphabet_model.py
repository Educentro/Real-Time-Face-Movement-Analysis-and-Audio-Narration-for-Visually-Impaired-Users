import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

X = np.load("dataset_processed/X_alphabet.npy")
y = np.load("dataset_processed/y_alphabet.npy")

print(f"Loaded data:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
classes = encoder.classes_
print(f"  Classes ({len(classes)}): {classes}")

# Save labels for backend to use
if not os.path.exists("model"):
    os.makedirs("model")
np.save("model/labels.npy", classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

num_classes = len(classes)

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

model = Sequential([
    Dense(256, activation="relu", input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation="relu"),
    Dropout(0.2),
    
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

print(f"\nTraining alphabet model with {len(X_train)} samples...")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {test_acc*100:.2f}%")

model.save("alphabet_model.keras")
print("Saved alphabet_model.keras")
