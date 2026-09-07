"""SkinCare AI — Modern Medical Image Diagnosis dashboard (Streamlit).

Recreates the dashboard/index.html design as a responsive Streamlit app.
The prediction flow is wired to `model.py` so it can be swapped for a real
trained model later.
"""

import base64
import io

import streamlit as st
from PIL import Image

import model

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SkinCare AI — Medical Image Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A square skin-lesion preview (matches dashboard/skin-lesion.svg).
PREVIEW_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320" width="320" height="320">
  <defs>
    <radialGradient id="psk" cx="50%" cy="40%" r="80%">
      <stop offset="0%" stop-color="#f4cea4"/><stop offset="55%" stop-color="#e9bd93"/>
      <stop offset="100%" stop-color="#d8a57c"/>
    </radialGradient>
    <radialGradient id="phalo" cx="45%" cy="45%" r="60%">
      <stop offset="0%" stop-color="#8a5f3c" stop-opacity="0.45"/>
      <stop offset="70%" stop-color="#8a5f3c" stop-opacity="0.15"/>
      <stop offset="100%" stop-color="#8a5f3c" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="pmole" cx="38%" cy="34%" r="75%">
      <stop offset="0%" stop-color="#1f1308"/><stop offset="55%" stop-color="#4a2f1b"/>
      <stop offset="100%" stop-color="#6d4c30"/>
    </radialGradient>
  </defs>
  <rect width="320" height="320" fill="url(#psk)"/>
  <g fill="#c4956b" opacity="0.18">
    <circle cx="52" cy="84" r="3"/><circle cx="96" cy="42" r="2.4"/><circle cx="138" cy="70" r="2.8"/>
    <circle cx="242" cy="64" r="3.2"/><circle cx="286" cy="120" r="2.6"/><circle cx="54" cy="196" r="2.8"/>
    <circle cx="112" cy="262" r="3"/><circle cx="232" cy="248" r="2.4"/><circle cx="288" cy="230" r="2.8"/>
    <circle cx="176" cy="284" r="2.4"/><circle cx="30" cy="140" r="2.2"/><circle cx="290" cy="170" r="2.2"/>
    <circle cx="128" cy="156" r="2"/>
  </g>
  <g opacity="0.28">
    <path d="M84 108 C 92 92 108 78 128 78 C 150 78 166 92 172 112 C 178 132 168 152 148 158 C 128 164 108 156 98 140 C 90 128 88 118 84 108 Z"
      transform="rotate(-14 130 118)" fill="url(#phalo)"/>
  </g>
  <g opacity="0.22">
    <path d="M186 178 C 196 166 214 160 230 170 C 244 178 244 196 234 208 C 224 220 204 222 192 212 C 182 204 182 188 186 178 Z"
      transform="rotate(10 212 190)" fill="url(#phalo)"/>
  </g>
  <circle cx="118" cy="132" r="46" fill="url(#phalo)"/>
  <ellipse cx="122" cy="136" rx="34" ry="40" fill="url(#pmole)" transform="rotate(-16 122 136)"/>
  <ellipse cx="120" cy="132" rx="22" ry="27" fill="#422a17" opacity="0.85" transform="rotate(-16 120 132)"/>
  <circle cx="112" cy="124" r="5" fill="#6d4c30" opacity="0.5"/>
  <circle cx="133" cy="148" r="4" fill="#6d4c30" opacity="0.5"/>
  <circle cx="130" cy="120" r="3" fill="#6d4c30" opacity="0.4"/>
</svg>
"""


def svg_to_data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")


PREVIEW_URI = svg_to_data_uri(PREVIEW_SVG)


def img_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
  --primary: #2563eb;
  --primary-strong: #1d4ed8;
  --primary-soft: #e8f1ff;
  --sky: #0ea5e9;
  --green: #16a34a;
  --green-strong: #15803d;
  --green-soft: #ecfdf3;
  --green-border: #bbf7d0;
  --ink: #0f172a;
  --slate: #475569;
  --muted: #64748b;
  --faint: #94a3b8;
  --border: #e7edf7;
  --card: #ffffff;
}

/* ---- page background ---- */
.stApp {
  background:
    radial-gradient(560px 320px at 88% 6%, rgba(37,99,235,.08), transparent 60%),
    radial-gradient(480px 300px at 30% 100%, rgba(14,165,233,.06), transparent 60%),
    linear-gradient(180deg, #f7faff 0%, #f5f8fd 40%, #edf3fb 100%);
}
#MainMenu, footer { visibility: hidden; }

.block-container { padding: 1.6rem 2.6rem 2.5rem; max-width: 1320px; }

/* ---- sidebar (SkinCare AI brand + nav) ---- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }

.brand {
  display: flex; align-items: center; gap: 13px;
  padding: 0 8px 18px; border-bottom: 1px solid var(--border); margin-bottom: 16px;
}
.brand-mark {
  width: 46px; height: 46px; flex-shrink: 0; border-radius: 13px;
  display: grid; place-items: center;
  background: linear-gradient(135deg, var(--primary) 0%, var(--sky) 100%);
  box-shadow: 0 8px 18px rgba(37,99,235,.35);
}
.brand-cross { color: #fff; font-size: 28px; line-height: 1; font-weight: 300; }
.brand-text { display: flex; flex-direction: column; }
.brand-title { font-size: 1.06rem; font-weight: 800; letter-spacing: -.02em; color: var(--ink); }
.brand-subtitle { font-size: .72rem; font-weight: 500; color: var(--muted); margin-top: 2px; }

.side-promo {
  margin-top: 22px; padding-top: 16px; border-top: 1px solid var(--border);
  background: linear-gradient(135deg, #f0f6ff 0%, #e6f1ff 100%);
  border: 1px solid #d9e8ff; border-radius: 14px; padding: 16px 14px;
  display: flex; flex-direction: column; align-items: center; text-align: center; gap: 4px;
}
.side-promo-icon {
  width: 40px; height: 40px; border-radius: 50%; display: grid; place-items: center;
  background: #fff; color: var(--primary); box-shadow: 0 2px 8px rgba(37,99,235,.06); margin-bottom: 6px; font-size: 18px;
}
.side-promo-title { font-weight: 700; font-size: .88rem; color: var(--ink); }
.side-promo-text { font-size: .74rem; color: var(--muted); }

/* ---- top bar / status card ---- */
.topbar { display: flex; align-items: center; justify-content: flex-end; margin-bottom: 6px; }
.status-card {
  display: flex; align-items: center; gap: 12px;
  background: #fff; border: 1px solid var(--border);
  border-radius: 14px; padding: 9px 16px; box-shadow: 0 2px 8px rgba(37,99,235,.06);
}
.status-icon {
  width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center;
  background: linear-gradient(135deg, var(--primary-soft), #d9e9ff); color: var(--primary); font-size: 18px;
}
.status-body { display: flex; flex-direction: column; }
.status-body strong { font-size: .83rem; font-weight: 700; color: var(--ink); }
.status-body small { font-size: .72rem; color: var(--muted); }
.live-dot {
  width: 9px; height: 9px; border-radius: 50%; background: var(--green); flex-shrink: 0;
  box-shadow: 0 0 0 0 rgba(22,163,74,.5); animation: sc-pulse 2s infinite;
}
@keyframes sc-pulse {
  0% { box-shadow: 0 0 0 0 rgba(22,163,74,.45); }
  70% { box-shadow: 0 0 0 8px rgba(22,163,74,0); }
  100% { box-shadow: 0 0 0 0 rgba(22,163,74,0); }
}

/* ---- hero ---- */
.hero h1 { font-size: 1.9rem; font-weight: 800; letter-spacing: -.03em; color: var(--ink); margin: 0; }
.hero-sub { margin-top: 8px; font-size: .98rem; color: var(--muted); }

.info-alert {
  display: flex; align-items: flex-start; gap: 11px; margin-top: 20px;
  padding: 13px 16px; background: linear-gradient(135deg, #eaf2ff, #e3eeff);
  border: 1px solid #cfe0ff; border-radius: 12px; color: #1e429f;
  font-size: .85rem; line-height: 1.5;
}

/* ---- cards ---- */
.sc-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 16px; padding: 22px; box-shadow: 0 2px 8px rgba(37,99,235,.06);
  height: 100%;
}
.card-header { display: flex; align-items: center; gap: 11px; margin-bottom: 18px; }
.card-header h2 { font-size: 1rem; font-weight: 700; color: var(--ink); margin: 0; }
.card-icon {
  width: 36px; height: 36px; border-radius: 10px; display: grid; place-items: center; font-size: 17px;
}
.card-icon.blue { background: var(--primary-soft); color: var(--primary); }
.card-icon.indigo { background: #eef0ff; color: #6366f1; }

/* upload button styling */
div.stButton > button {
  border-radius: 12px; font-weight: 600; font-size: .9rem;
  border: 1px solid #c9dafb; color: var(--primary); background: #fff;
  box-shadow: 0 2px 8px rgba(37,99,235,.06);
}
div.stButton > button:hover { background: var(--primary-soft); border-color: var(--primary); color: var(--primary); }

/* green predict button */
.green-btn > div.stButton > button {
  width: 100%; margin-top: 6px; padding: 13px 20px; font-size: .98rem;
  background: linear-gradient(135deg, #22c55e 0%, var(--green) 55%, var(--green-strong) 130%);
  color: #fff !important; border: none; box-shadow: 0 10px 22px rgba(22,163,74,.32);
}
.green-btn > div.stButton > button:hover {
  transform: translateY(-1px); background: linear-gradient(135deg, #22c55e 0%, var(--green) 55%, var(--green-strong) 130%);
  border: none; box-shadow: 0 14px 26px rgba(22,163,74,.4);
}

.upload-meta { display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 16px 2px 0; font-size: .78rem; color: var(--muted); }
.upload-meta strong { color: var(--slate); font-weight: 600; }

/* preview frame */
.preview-frame {
  aspect-ratio: 1 / 1; border-radius: 13px; overflow: hidden;
  border: 1px solid var(--border); background: #f4f7fc;
  box-shadow: inset 0 1px 3px rgba(15,23,42,.04);
}
.preview-frame img { width: 100%; height: 100%; object-fit: contain; display: block; background: #fff; }

/* result */
.success-panel {
  display: flex; align-items: center; gap: 16px; padding: 20px 18px;
  border-radius: 14px; background: linear-gradient(135deg, #f0fdf4 0%, #e6faf0 100%);
  border: 1px solid var(--green-border);
}
.success-check {
  width: 54px; height: 54px; flex-shrink: 0; border-radius: 50%; display: grid; place-items: center;
  background: linear-gradient(135deg, #34d399, var(--green)); color: #fff;
  box-shadow: 0 8px 18px rgba(22,163,74,.35); font-size: 26px;
}
.success-body { display: flex; flex-direction: column; }
.result-label { font-size: .8rem; color: var(--muted); font-weight: 500; }
.success-body strong { font-size: 1.5rem; font-weight: 800; letter-spacing: -.02em; color: var(--green-strong); }
.result-conf { font-size: .86rem; color: var(--slate); margin-top: 3px; }
.result-conf b { color: var(--green-strong); }
.mini-alert {
  display: flex; align-items: flex-start; gap: 10px; margin-top: 16px; padding: 12px 14px;
  background: var(--primary-soft); border: 1px solid #cfe0ff; border-radius: 11px;
  color: #1e429f; font-size: .8rem; line-height: 1.5;
}

/* empty preview / result placeholders */
.placeholder-box {
  aspect-ratio: 1/1; border-radius: 13px; border: 1px solid var(--border);
  background: #f4f7fc; display: flex; flex-direction: column; align-items: center;
  justify-content: center; color: var(--faint); font-size: .85rem; gap: 6px; text-align: center;
}
.placeholder-box .big { font-size: 34px; }

/* features */
.features { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-top: 26px; }
.feature-card {
  background: #fff; border: 1px solid var(--border); border-radius: 16px;
  padding: 22px 20px; box-shadow: 0 2px 8px rgba(37,99,235,.06);
  transition: transform .18s ease, box-shadow .18s ease;
}
.feature-card:hover { transform: translateY(-3px); box-shadow: 0 10px 28px rgba(30,64,175,.10); }
.feature-icon {
  font-size: 1.6rem; display: inline-flex; align-items: center; justify-content: center;
  width: 50px; height: 50px; border-radius: 13px; margin-bottom: 14px;
}
.feature-icon.green { background: var(--green-soft); }
.feature-icon.blue  { background: #edf4ff; }
.feature-icon.amber { background: #fdf3e4; }
.feature-icon.purple{ background: #f4edff; }
.feature-card h3 { font-size: .95rem; font-weight: 700; color: var(--ink); margin-bottom: 5px; }
.feature-card p { font-size: .82rem; color: var(--muted); line-height: 1.5; margin: 0; }

.page-footer { margin-top: 30px; padding-top: 18px; border-top: 1px solid var(--border);
  text-align: center; font-size: .76rem; color: var(--faint); }

/* hide empty sidebar chrome */
[data-testid="stSidebarNav"] { display: none; }
section[data-testid="stSidebar"] hr { display: none; }

@media (max-width: 1100px) {
  .features { grid-template-columns: repeat(2, 1fr); }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("home", "Home"),
    ("about", "About Project"),
    ("info", "Disease Information"),
    ("team", "Team"),
]
NAV_ICONS = {"home": "⌂", "about": "📘", "info": "🔬", "team": "👥"}

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark"><span class="brand-cross">+</span></div>
          <div class="brand-text">
            <span class="brand-title">SkinCare AI</span>
            <span class="brand-subtitle">Medical Image Diagnosis</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Menu**")
    page = None
    for key, label in NAV_ITEMS:
        col_l, col_r = st.columns([1, 10])
        with col_l:
            st.markdown(f"<div style='text-align:center;color:#64748b'>{NAV_ICONS[key]}</div>",
                        unsafe_allow_html=True)
        with col_r:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                page = key

    # Highlight the home (active) nav item in blue
    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] div:has(button[kind="secondary"]) {
        }
        section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
          background: transparent; border: none; text-align: left;
        }
        section[data-testid="stSidebar"] button[kind="secondary"] {
          width: 100%; background: transparent; border: none;
          text-align: left; padding: 10px 12px; font-size: .92rem; color: var(--slate);
        }
        section[data-testid="stSidebar"] button[kind="secondary"]:hover {
          background: #f1f6ff; color: var(--primary);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-promo">
          <span class="side-promo-icon">🩺</span>
          <span class="side-promo-title">Early Detection</span>
          <span class="side-promo-text">For a Healthier Tomorrow</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Default to home page
if page is None:
    page = "home"

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
if page != "home":
    st.markdown('<div class="hero"><h1>Medical Image Diagnosis</h1>'
                '<p class="hero-sub">Upload a skin lesion image and let our AI model predict the possible disease.</p></div>',
                unsafe_allow_html=True)
    pages = {
        "about": ("About Project", "SkinCare AI is a student-built medical imaging demo that uses a deep "
                  "learning model to classify skin diseases from photographs of lesions. It is intended "
                  "for educational and research purposes to demonstrate how convolutional neural networks "
                  "can assist in the early screening of skin conditions."),
        "info": ("Disease Information", "Skin Cancer / Melanoma: an abnormal growth of skin cells that "
                 "often develops on sun-exposed skin. Early detection greatly improves treatment outcomes. "
                 "Other common classes include benign nevi, basal cell carcinoma and squamous cell carcinoma. "
                 "Always consult a dermatologist for an expert evaluation."),
        "team": ("Team", "Built by a team of students and developers passionate about applying artificial "
                 "intelligence to healthcare. Model, dataset and design contributions are documented in the "
                 "project repository."),
    }
    st.info(pages[page][1])
    st.stop()

# Header status card (top-right)
st.markdown(
    """
    <div class="topbar">
      <div class="status-card">
        <span class="status-icon">📈</span>
        <div class="status-body">
          <strong>AI Powered Detection</strong>
          <small>Faster • Smarter • Healthier</small>
        </div>
        <span class="live-dot"></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Hero
st.markdown('<div class="hero"><h1>Medical Image Diagnosis</h1>'
            '<p class="hero-sub">Upload a skin lesion image and let our AI model predict the possible disease.</p></div>',
            unsafe_allow_html=True)

# Info alert
st.markdown(
    """
    <div class="info-alert">
      <span>ℹ️</span>
      <p style="margin:0">This tool uses a deep learning model to classify skin diseases from images. It is for educational purposes only and should not replace professional medical advice.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Two-column dashboard
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1.05, 1.0], gap="large")

with col_left:
    st.markdown(
        """<div class="sc-card">
            <div class="card-header">
              <span class="card-icon blue">⬆️</span><h2>Upload Image</h2>
            </div></div>""",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drag and drop an image here, or choose a file",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="JPG, JPEG or PNG — 224 × 224 or larger recommended.",
    )

    st.markdown(
        '<div class="upload-meta">'
        "<span><strong>Supported formats:</strong> JPG, JPEG, PNG</span>"
        "<span><strong>Recommended size:</strong> 224 × 224 or larger</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

    predict_pressed = st.button("🔍 Predict Disease", type="primary", use_container_width=True,
                                key="predict_btn")

with col_right:
    st.markdown(
        """<div class="sc-card">
            <div class="card-header">
              <span class="card-icon blue">🖼️</span><h2>Uploaded Image</h2>
            </div></div>""",
        unsafe_allow_html=True,
    )
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        preview_uri = img_to_data_uri(img)
    else:
        preview_uri = PREVIEW_URI
    st.markdown(
        f'<div class="preview-frame"><img src="{preview_uri}" alt="Skin lesion preview"/></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div style="margin-top:22px" class="sc-card">
            <div class="card-header">
              <span class="card-icon indigo">📄</span><h2>Prediction Result</h2>
            </div></div>""",
        unsafe_allow_html=True,
    )

    prediction_state = st.session_state.get("prediction", None)

    if predict_pressed:
        if uploaded is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Running prediction..."):
                res = model.predict(img)
            st.session_state["prediction"] = res
            prediction_state = res

    if prediction_state is not None:
        label = prediction_state["label"]
        conf = prediction_state["confidence"] * 100
        st.markdown(
            f"""
            <div class="success-panel">
              <span class="success-check">✓</span>
              <div class="success-body">
                <span class="result-label">Prediction:</span>
                <strong>{label}</strong>
                <span class="result-conf">Confidence: <b>{conf:.1f}%</b></span>
              </div>
            </div>
            <div class="mini-alert">
              <span>ℹ️</span>
              <p style="margin:0">This is an AI prediction. Please consult a dermatologist for a professional diagnosis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="placeholder-box">
              <span class="big">📄</span>
              <span>No prediction yet</span>
              <span>Upload and press Predict Disease</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Feature cards
# ---------------------------------------------------------------------------
FEATURES = [
    ("🛡️", "green", "Early Detection", "Helps in early identification of skin diseases"),
    ("⚙️", "blue", "AI Powered", "Uses deep learning technology"),
    ("⚡", "amber", "Fast Results", "Get prediction in seconds"),
    ("💜", "purple", "Better Health", "Support for a healthier tomorrow"),
]
cards_html = "".join(
    f"""<div class="feature-card">
        <span class="feature-icon {cls}">{icon}</span>
        <h3>{title}</h3>
        <p>{desc}</p>
      </div>"""
    for icon, cls, title, desc in FEATURES
)
st.markdown(f'<div class="features">{cards_html}</div>', unsafe_allow_html=True)

st.markdown('<div class="page-footer">SkinCare AI · Medical Image Diagnosis</div>', unsafe_allow_html=True)
