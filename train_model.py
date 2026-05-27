import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

print("TensorFlow version:", tf.__version__)

# ── 1. Load YOUR CSV dataset ─────────────────────────────────────────────────
# Put your CSV file in the same folder as this script
# Change the filename below to match your actual file name

CSV_FILE = "train.csv"   # <-- change this to your actual filename if different

if not os.path.exists(CSV_FILE):
    print(f"ERROR: '{CSV_FILE}' not found!")
    print("Make sure your CSV file is in the same folder as this script.")
    print("Also check the filename matches exactly (capital letters matter).")
    exit()

print(f"Loading dataset from '{CSV_FILE}' ...")
df = pd.read_csv(CSV_FILE)
print(f"Dataset shape: {df.shape}")
print(f"Columns: label + {df.shape[1]-1} pixel columns")
print(f"Sample labels: {df['label'].value_counts().sort_index().to_dict()}")

# ── 2. Separate label and pixels ─────────────────────────────────────────────
y = df['label'].values                        # labels: 0-9
X = df.drop('label', axis=1).values          # pixel values: 784 columns

print(f"\nTotal samples : {len(X)}")
print(f"Label range   : {y.min()} to {y.max()}")
print(f"Pixel range   : {X.min()} to {X.max()}")

# ── 3. Preprocess ─────────────────────────────────────────────────────────────
X = X.astype("float32") / 255.0              # normalize 0-1
X = X.reshape(-1, 28, 28, 1)                 # reshape to (N, 28, 28, 1)
y = to_categorical(y, 10)                    # one-hot encode

# ── 4. Train / Validation / Test split ───────────────────────────────────────
# 80% train, 10% validation, 10% test  (from your single CSV file)
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"\nTrain   : {X_train.shape[0]} samples")
print(f"Val     : {X_val.shape[0]} samples")
print(f"Test    : {X_test.shape[0]} samples")

# ── 5. Build 4-Layer CNN (Deng 2024 — 99.76% accuracy) ──────────────────────
model = models.Sequential([

    # Block 1 — 32 filters, 5x5
    layers.Conv2D(32, (5,5), padding='same', activation='relu',
                  input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (5,5), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    # Block 2 — 64 filters, 3x3
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    # Classifier
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.50),
    layers.Dense(10, activation='softmax')
])

model.summary()

# ── 6. Compile ────────────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── 7. Train ──────────────────────────────────────────────────────────────────
print("\nStarting training ... (10-20 mins on CPU, be patient)")

history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=15,
    validation_data=(X_val, y_val),
    verbose=1
)

# ── 8. Evaluate on test set ───────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*40}")
print(f"  Test Accuracy : {acc*100:.2f}%")
print(f"  Test Loss     : {loss:.4f}")
print(f"{'='*40}")

# ── 9. Save model ─────────────────────────────────────────────────────────────
model.save("digit_cnn_model.h5")
print("\nModel saved as digit_cnn_model.h5")

# ── 10. Plot accuracy and loss ────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(history.history['accuracy'],     label='Train Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Val Accuracy',   linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'],     label='Train Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Val Loss',   linewidth=2)
ax2.set_title('Model Loss', fontsize=14)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()
print("Training graph saved as training_results.png")

# ── 11. Show a few sample predictions ─────────────────────────────────────────
print("\nSample predictions from test set:")
sample_preds = model.predict(X_test[:10], verbose=0)
true_labels  = np.argmax(y_test[:10], axis=1)
pred_labels  = np.argmax(sample_preds, axis=1)

fig2, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    color = 'green' if pred_labels[i] == true_labels[i] else 'red'
    ax.set_title(f"True:{true_labels[i]}  Pred:{pred_labels[i]}",
                 color=color, fontsize=11)
    ax.axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=13)
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.show()
print("Sample predictions saved as sample_predictions.png")