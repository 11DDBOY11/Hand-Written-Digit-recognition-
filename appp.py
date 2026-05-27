import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Digit Recognition", page_icon="✏️", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_cnn_model.h5")

model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    - **Model:** 4-Layer CNN
    - **Dataset:** MNIST (42,000 images)
    - **Accuracy:** ~99.76%
    - **Guide:** Prof. Mahesh Kini M
    - **College:** AIET, Moodbidri
    """)
    st.markdown("---")
    st.markdown("**Supports:**")
    st.markdown("- ✏️ Single digit drawing")
    st.markdown("- 📁 Multi-digit image upload")

st.title("✏️ Handwritten Digit Recognition")
st.markdown("**ML Mini Project | BCS602 | Darshan Dashyal & Darshan G M | AIET**")
st.markdown("---")

# ── Helper: preprocess single digit crop ─────────────────────────────────────
def preprocess_single(crop_gray):
    img = cv2.resize(crop_gray, (28, 28))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = img.astype("float32") / 255.0
    return img.reshape(1, 28, 28, 1)

# ── Multi-digit detector ──────────────────────────────────────────────────────
def detect_and_predict_all(pil_img):
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Invert + threshold
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dilate slightly to connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small noise contours
    min_area = 300
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > min_area:
            boxes.append((x, y, w, h))

    # Sort left to right
    boxes = sorted(boxes, key=lambda b: b[0])

    results = []
    annotated = img.copy()

    for (x, y, w, h) in boxes:
        # Add padding around each digit
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)

        crop = thresh[y1:y2, x1:x2]
        processed = preprocess_single(crop)

        preds = model.predict(processed, verbose=0)[0]
        digit = int(np.argmax(preds))
        confidence = float(preds[digit]) * 100

        # Draw box on annotated image
        color = (0, 200, 80)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{digit} ({confidence:.0f}%)",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        results.append({
            "digit": digit,
            "confidence": confidence,
            "preds": preds,
            "crop": crop
        })

    full_number = "".join([str(r["digit"]) for r in results])
    return results, annotated, full_number

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["✏️ Draw Single Digit", "📁 Upload (Multi-Digit)"])

# ════════════════════════════════════════
# TAB 1 — DRAW (single digit)
# ════════════════════════════════════════
with tab1:
    st.markdown("### Draw a digit (0–9) on the canvas")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280, width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        c1, c2 = st.columns(2)
        predict_btn = c1.button("🔍 Predict", use_container_width=True, type="primary")
        c2.button("🗑️ Clear", use_container_width=True)

    with col2:
        if predict_btn and canvas_result.image_data is not None:
            img_data = canvas_result.image_data
            if img_data.sum() > 1000:
                gray = cv2.cvtColor(img_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
                processed = preprocess_single(gray)
                preds = model.predict(processed, verbose=0)[0]
                digit = int(np.argmax(preds))
                confidence = float(preds[digit]) * 100

                st.markdown(f"""
                <div style='text-align:center;padding:20px;background:#1a1a2e;
                            border-radius:15px;margin-top:10px;'>
                    <h1 style='color:#00ff88;font-size:80px;margin:0;'>{digit}</h1>
                    <p style='color:white;font-size:18px;'>Predicted Digit</p>
                    <p style='color:#ffd700;font-size:22px;font-weight:bold;'>{confidence:.1f}% Confidence</p>
                </div>""", unsafe_allow_html=True)

                st.markdown("#### Class Probabilities")
                colors = ["#00C853" if i == digit else "#5C6BC0" for i in range(10)]
                fig = go.Figure(go.Bar(
                    x=list(range(10)), y=preds * 100,
                    marker_color=colors,
                    text=[f"{p*100:.1f}%" for p in preds],
                    textposition="outside"
                ))
                fig.update_layout(
                    xaxis_title="Digit", yaxis_title="Confidence (%)",
                    xaxis=dict(tickmode="linear", tick0=0, dtick=1),
                    height=280, margin=dict(t=10, b=10),
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font_color="white", yaxis=dict(range=[0, 115])
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Please draw a digit first!")

# ════════════════════════════════════════
# TAB 2 — UPLOAD (multi-digit)
# ════════════════════════════════════════
with tab2:
    st.markdown("### Upload an image with one or more handwritten digits")
    st.info("💡 Write digits clearly with black marker on white paper, with some space between each digit")

    uploaded_file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        pil_img = Image.open(uploaded_file)

        results, annotated_img, full_number = detect_and_predict_all(pil_img)

        if not results:
            st.error("❌ No digits detected! Make sure digits are written dark and clear on white paper.")
        else:
            # ── Full number result banner ─────────────────────────────────────
            st.markdown(f"""
            <div style='text-align:center;padding:25px;background:#1a1a2e;
                        border-radius:15px;margin:15px 0;'>
                <p style='color:white;font-size:18px;margin:0;'>Recognized Number</p>
                <h1 style='color:#00ff88;font-size:72px;margin:10px 0;
                           letter-spacing:10px;'>{full_number}</h1>
                <p style='color:#ffd700;font-size:16px;'>{len(results)} digit(s) detected</p>
            </div>""", unsafe_allow_html=True)

            # ── Annotated image ───────────────────────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original with detection boxes:**")
                st.image(annotated_img, use_column_width=True)
            with col2:
                st.markdown("**Original uploaded:**")
                st.image(pil_img, use_column_width=True)

            st.markdown("---")

            # ── Per digit breakdown ───────────────────────────────────────────
            st.markdown("### Per-Digit Breakdown")
            cols = st.columns(min(len(results), 4))

            for i, res in enumerate(results):
                with cols[i % 4]:
                    conf_color = "#00C853" if res["confidence"] > 80 else "#FF9800" if res["confidence"] > 50 else "#F44336"
                    st.markdown(f"""
                    <div style='text-align:center;padding:15px;background:#1a1a2e;
                                border-radius:12px;margin:5px;'>
                        <h2 style='color:#00ff88;font-size:50px;margin:0;'>{res["digit"]}</h2>
                        <p style='color:{conf_color};font-size:14px;font-weight:bold;margin:5px 0;'>
                            {res["confidence"]:.1f}%</p>
                    </div>""", unsafe_allow_html=True)

                    # Show crop
                    crop_pil = Image.fromarray(res["crop"])
                    st.image(crop_pil, caption=f"28×28 input", width=80)

            # ── Combined probability chart ────────────────────────────────────
            if len(results) > 1:
                st.markdown("---")
                st.markdown("### Confidence Chart — All Digits")
                digit_labels = [f"Pos {i+1}: '{r['digit']}'" for i, r in enumerate(results)]
                confidences = [r["confidence"] for r in results]
                bar_colors = ["#00C853" if c > 80 else "#FF9800" if c > 50 else "#F44336"
                              for c in confidences]

                fig2 = go.Figure(go.Bar(
                    x=digit_labels, y=confidences,
                    marker_color=bar_colors,
                    text=[f"{c:.1f}%" for c in confidences],
                    textposition="outside"
                ))
                fig2.update_layout(
                    xaxis_title="Digit Position",
                    yaxis_title="Confidence (%)",
                    height=300, margin=dict(t=20, b=10),
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font_color="white", yaxis=dict(range=[0, 115])
                )
                st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>ML Mini Project | BCS602 | AIET 2025–26</p>",
            unsafe_allow_html=True)