import cv2
import numpy as np
import tensorflow as tf
import os

# Check model exists
if not os.path.exists("digit_cnn_model.h5"):
    print("ERROR: digit_cnn_model.h5 not found!")
    print("Please run train_model.py first.")
    exit()

model = tf.keras.models.load_model("digit_cnn_model.h5")
print("✓ Model loaded!")
print("Opening camera...")

def preprocess(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Invert: MNIST is white digit on black, camera gives black ink on white paper
    gray = cv2.bitwise_not(gray)
    _, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    gray = gray.astype("float32") / 255.0
    gray = gray.reshape(1, 28, 28, 1)
    return gray

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    print("Try changing VideoCapture(0) to VideoCapture(1) in the code.")
    exit()

print("\n" + "="*45)
print("  Handwritten Digit Recognition — LIVE")
print("="*45)
print("  Place paper with digit inside GREEN BOX")
print("  Press  Q  to quit")
print("="*45 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera frame grab failed. Exiting.")
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Green ROI box — centered
    box  = 260
    cx, cy = w // 2, h // 2
    x1, y1 = cx - box//2, cy - box//2
    x2, y2 = cx + box//2, cy + box//2

    # Make sure box is within frame
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        continue

    # Predict
    processed   = preprocess(roi)
    preds       = model.predict(processed, verbose=0)[0]
    digit       = int(np.argmax(preds))
    confidence  = float(preds[digit]) * 100

    # ── Top banner ────────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0,0), (w, 75), (0,0,0), -1)

    cv2.putText(frame,
                f"Digit: {digit}",
                (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,255), 3)

    conf_color = (0,255,0) if confidence > 80 else (0,165,255) if confidence > 50 else (0,0,255)
    cv2.putText(frame,
                f"Confidence: {confidence:.1f}%",
                (200, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, conf_color, 2)

    # ── Green ROI box ─────────────────────────────────────────────────────────
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
    cv2.putText(frame, "Place digit here",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # ── Probability bars (right side) ─────────────────────────────────────────
    bx    = w - 170
    by0   = 90
    bh    = 30
    gap   = 4

    cv2.putText(frame, "All Probabilities:",
                (bx, by0 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    for i, p in enumerate(preds):
        by   = by0 + i * (bh + gap)
        fill = int(p * 140)
        # Background
        cv2.rectangle(frame, (bx, by), (bx+140, by+bh), (40,40,40), -1)
        # Fill
        clr  = (0,255,0) if i == digit else (80,80,180)
        cv2.rectangle(frame, (bx, by), (bx+fill, by+bh), clr, -1)
        # Border highlight for predicted digit
        if i == digit:
            cv2.rectangle(frame, (bx, by), (bx+140, by+bh), (0,255,255), 2)
        # Labels
        cv2.putText(frame, str(i),
                    (bx-22, by+bh-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"{p*100:.0f}%",
                    (bx+145, by+bh-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # ── Debug window: what the model sees (bottom-left) ───────────────────────
    dbg = preprocess(roi)[0,:,:,0]
    dbg = (dbg * 255).astype(np.uint8)
    dbg = cv2.resize(dbg, (120, 120), interpolation=cv2.INTER_NEAREST)
    dbg = cv2.cvtColor(dbg, cv2.COLOR_GRAY2BGR)

    # Only draw if it fits
    if h >= 140 and w >= 140:
        frame[h-130: h-10, 10: 130] = dbg
        cv2.rectangle(frame, (10, h-130), (130, h-10), (0,255,0), 1)
        cv2.putText(frame, "Model sees (28x28)",
                    (10, h-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    # ── Bottom note ───────────────────────────────────────────────────────────
    cv2.putText(frame, "Press Q to quit",
                (cx - 70, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

    cv2.imshow("Digit Recognition | CNN - MNIST", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed. Bye!")