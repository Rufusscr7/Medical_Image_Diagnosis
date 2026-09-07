"""Skin-Scan AI — user-friendly Streamlit UI for skin-lesion (cancer) screening."""

import streamlit as st
from PIL import Image

import model

st.set_page_config(
    page_title="Skin-Scan AI",
    page_icon="🔬",
    layout="centered",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.5rem; }
      .result-box {
        border-radius: 12px; padding: 1.2rem 1.5rem; margin-top: 1rem;
        font-size: 1.05rem; border-left: 6px solid;
      }
      .result-benign {
        background: #e8f8ee; border-color: #28a745; color: #155724;
      }
      .result-malignant {
        background: #fdecea; border-color: #dc3545; color: #7f1d1d;
      }
      .result-pending {
        background: #eef2f7; border-color: #6c757d; color: #495057;
      }
      .sidebar-note { font-size: 0.85rem; color: #6c757d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🔬 Skin-Scan AI")
    st.markdown("AI-assisted screening for skin lesions (moles & spots).")
    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "1. Upload a clear photo of the lesion.\n"
        "2. Click **Analyse**.\n"
        "3. See the model's prediction and confidence."
    )
    st.divider()
    st.caption("Checklist for a good photo:")
    st.markdown(
        "- Good lighting, no flash glare\n"
        "- Lesion fills most of the frame\n"
        "- In focus and not blurry"
    )
    st.divider()
    model_status = "🟡 Placeholder (demo)" if not model.is_model_available() \
        else "🟢 Trained model loaded"
    st.markdown(f"**Model status:** {model_status}")
    st.markdown(
        '<p class="sidebar-note">This tool is for screening only and is not '
        "a substitute for a professional medical opinion.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- main body
st.title("Skin-Scan AI")
st.markdown(
    "Upload a picture of a **mole or skin lesion** and the system will "
    "calculate how likely it is to be **cancerous**."
)

uploaded = st.file_uploader(
    "Upload a lesion image (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="For best results, make sure the lesion is well lit and in focus.",
)

col_left, col_right = st.columns(2)

st.session_state.setdefault("history", [])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    with col_left:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with col_right:
        st.markdown("### 📊 Analysis")
        if st.button("Analyse lesion", type="primary", use_container_width=True):
            with st.spinner("Running prediction calculations..."):
                result = model.predict(image)

            st.session_state.history.append({
                "name": uploaded.name,
                "label": result["label"],
                "prob": result["malignant_prob"],
            })

            is_malignant = result["label"] == "Malignant"
            box_class = "result-malignant" if is_malignant else "result-benign"
            emoji = "⚠️" if is_malignant else "✅"
            st.markdown(
                f'<div class="result-box {box_class}">'
                f"<strong>{emoji} Result: {result['label']}</strong><br/>"
                f"{result['message']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown("**Malignancy probability:**")
            st.progress(float(result["malignant_prob"]),
                        text=f"{result['malignant_prob'] * 100:.1f}%")
            st.markdown(f"**Model confidence:** {result['confidence'] * 100:.0f}%")

            st.caption("⚠️ This is a screening result — always confirm with a "
                       "dermatologist.")
        else:
            st.info("Upload an image, then press **Analyse lesion**.")
else:
    col_left.markdown("### 🖼️ Upload preview")
    col_right.markdown("### 📊 Analysis")
    col_right.info("Please upload an image to get started.")

# ------------------------------------------------------------------ history
st.divider()
st.markdown("### 📋 Session history")
if st.session_state.history:
    rows = ""
    for entry in reversed(st.session_state.history[-5:]):
        colour = "#dc3545" if entry["label"] == "Malignant" else "#28a745"
        rows += (
            f"<tr><td>{entry['name']}</td>"
            f'<td style="color:{colour};font-weight:600">{entry["label"]}</td>'
            f"<td>{entry['prob'] * 100:.1f}%</td></tr>"
        )
    st.markdown(
        f'<table style="width:100%;font-size:0.95rem">'
        f"<tr><th>Image</th><th>Result</th><th>Probability</th></tr>{rows}</table>",
        unsafe_allow_html=True,
    )
else:
    st.caption("No analyses yet in this session.")

st.markdown(
    '<p class="sidebar-note" style="margin-top:2rem">Built with Streamlit · '
    "Prediction placeholder ready to be swapped for a trained CNN "
    "(HANM10000 / ISIC).</p>",
    unsafe_allow_html=True,
)