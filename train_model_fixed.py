import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

print(f"Loaded data:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  Number of classes: {len(set(y))}")
print(f"  Samples per class: {len(X) / len(set(y)):.1f}")

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
print(f"\nClass distribution:")
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} samples")
    if count < 60:
        print(f"    ⚠️ WARNING: Class {cls} has few samples, may perform poorly")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_classes = len(set(y))

# Compute class weights to handle imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Build model with more capacity
model = Sequential([
    Dense(256, activation="relu", input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.4),
    
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

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

# Train
print(f"\n🚀 Training model with {len(X_train)} training samples...")
print(f"   Validation set: {len(X_test)} samples")
print(f"   Using class weights to handle imbalance\n")

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)

# Evaluate
print("\n" + "="*60)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Final Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Final Test Loss: {test_loss:.4f}")
print("="*60)

if test_acc < 0.80:
    print("\n⚠️ WARNING: Accuracy is below 80%")
    print("   Consider collecting more training data")
elif test_acc < 0.90:
    print("\n⚠️ Accuracy is acceptable but could be improved")
    print("   Model should work but may have some errors")
else:
    print("\n🎉 Great accuracy! Model should work well")

# Save
model.save("frame_model.keras")
print("\n✅ Model saved as frame_model.keras")
print("   Run live_inference_fixed.py to test!")