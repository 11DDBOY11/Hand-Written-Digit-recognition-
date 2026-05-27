# confusion_matrix.py — Run once to generate confusion matrix
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ── Load model ────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model("digit_cnn_model.h5")
print("✓ Model loaded")

# ── Load test data (same split as training) ───────────────────────────────────
df = pd.read_csv("train.csv")
X = df.drop("label", axis=1).values.astype("float32") / 255.0
y = df["label"].values

# Use last 10% as test set (same as your train_model.py)
split = int(0.9 * len(X))
X_test = X[split:].reshape(-1, 28, 28, 1)
y_test = y[split:]

print(f"✓ Test samples: {len(X_test)}")

# ── Predict ───────────────────────────────────────────────────────────────────
y_pred_probs = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.5, linecolor='gray',
            annot_kws={"size": 12})

plt.title("Confusion Matrix — 4-Layer CNN on MNIST\n(Test Set: 4,200 samples)",
          fontsize=15, fontweight='bold', pad=15)
plt.xlabel("Predicted Label", fontsize=13)
plt.ylabel("True Label", fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11, rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("✓ Saved: confusion_matrix.png")

# ── Per-class metrics ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred,
      target_names=[f"Digit {i}" for i in range(10)]))

# ── Per-digit accuracy ────────────────────────────────────────────────────────
print("\nPer-Digit Accuracy:")
print("-"*30)
for i in range(10):
    correct = cm[i][i]
    total = cm[i].sum()
    acc = correct / total * 100
    print(f"  Digit {i}: {correct}/{total} = {acc:.2f}%")