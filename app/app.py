"""
Skin-Scan AI
Streamlit application for HAM10000 skin lesion classification.
"""

import streamlit as st
from PIL import Image

import inference as model


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Skin-Scan AI",
    page_icon="🔬",
    layout="centered"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2.5rem;
    }

    .result-box {
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        font-size: 1.05rem;
        border-left: 6px solid;
        background-color: #f4f6f9;
    }

    .result-box h3 {
        margin-top: 0;
    }

    .sidebar-note {
        font-size: 0.85rem;
        color: #6c757d;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# DISEASE INFORMATION
# ==================================================

DISEASE_INFO = {

    "akiec":
    "Actinic keratoses and intraepithelial carcinoma",

    "bcc":
    "Basal cell carcinoma",

    "bkl":
    "Benign keratosis-like lesions",

    "df":
    "Dermatofibroma",

    "mel":
    "Melanoma",

    "nv":
    "Melanocytic nevi",

    "vasc":
    "Vascular lesions"
}


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 🔬 Skin-Scan AI")

    st.markdown(
        "AI-assisted skin lesion classification."
    )

    st.divider()


    # ----------------------------------------------
    # NAVIGATION
    # ----------------------------------------------

    page = st.radio(

        "Navigation",

        [
            "🔬 Analyse Image",
            "📚 Disease Information",
            "ℹ️ About"
        ]
    )


    st.divider()


    # ----------------------------------------------
    # MODEL STATUS
    # ----------------------------------------------

    if model.is_model_available():

        model_status = "🟢 Trained model loaded"

    else:

        model_status = "🔴 Model not found"


    st.markdown(

        f"**Model Status:** {model_status}"

    )


    st.divider()


    st.markdown("### 📷 Image Guidelines")

    st.markdown(

        """
        - Good lighting
        - Image should be clear
        - Avoid blurry images
        - Lesion should be visible
        - Avoid excessive shadows
        """

    )


    st.divider()


    st.caption(

        "⚠️ This application is for educational "
        "and research purposes only."

    )


# ==================================================
# ANALYSE IMAGE PAGE
# ==================================================

if page == "🔬 Analyse Image":

    st.title("🔬 Skin-Scan AI")

    st.markdown(

        """
        Upload a **skin lesion image** and the trained
        Artificial Intelligence model will classify it
        into one of the HAM10000 skin lesion categories.
        """

    )


    st.info(

        "⚠️ This system is not a medical diagnosis tool. "
        "Always consult a qualified dermatologist."

    )


    # ----------------------------------------------
    # IMAGE UPLOAD
    # ----------------------------------------------

    uploaded = st.file_uploader(

        "Upload a skin lesion image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]

    )


    # ----------------------------------------------
    # SESSION HISTORY
    # ----------------------------------------------

    if "history" not in st.session_state:

        st.session_state.history = []


    # ----------------------------------------------
    # IMAGE AVAILABLE
    # ----------------------------------------------

    if uploaded is not None:


        image = Image.open(

            uploaded

        ).convert(

            "RGB"

        )


        # ------------------------------------------
        # COLUMNS
        # ------------------------------------------

        col_left, col_right = st.columns(2)


        # ------------------------------------------
        # IMAGE PREVIEW
        # ------------------------------------------

        with col_left:


            st.subheader(

                "🖼️ Uploaded Image"

            )


            st.image(

                image,

                caption=uploaded.name,

                use_container_width=True

            )


        # ------------------------------------------
        # ANALYSIS
        # ------------------------------------------

        with col_right:


            st.subheader(

                "📊 AI Analysis"

            )


            if st.button(

                "🔍 Analyse Image",

                type="primary",

                use_container_width=True

            ):


                # ----------------------------------
                # CHECK MODEL
                # ----------------------------------

                if not model.is_model_available():


                    st.error(

                        "❌ Trained model not found!"

                    )


                    st.warning(

                        "Please place best_model.pth "
                        "inside the models folder."

                    )


                else:


                    # ------------------------------
                    # PREDICTION
                    # ------------------------------

                    with st.spinner(

                        "🧠 AI is analysing the image..."

                    ):


                        result = model.predict(

                            image

                        )


                    # ------------------------------
                    # ERROR CHECK
                    # ------------------------------

                    if "error" in result:


                        st.error(

                            result["error"]

                        )


                    else:


                        # --------------------------
                        # RESULT
                        # --------------------------

                        predicted_label = (

                            result["label"]

                        )


                        disease_name = (

                            result["disease_name"]

                        )


                        confidence = (

                            result["confidence"]

                        )


                        probabilities = (

                            result["probabilities"]

                        )


                        # --------------------------
                        # DISPLAY RESULT
                        # --------------------------

                        st.success(

                            "Analysis Completed!"

                        )


                        st.markdown(

                            f"""
                            <div class="result-box">

                            <h3>
                            🧠 Prediction
                            </h3>

                            <b>Class:</b>
                            {predicted_label.upper()}

                            <br>

                            <b>Disease:</b>
                            {disease_name}

                            <br>

                            <b>Confidence:</b>
                            {confidence:.2f}%

                            </div>
                            """,

                            unsafe_allow_html=True

                        )


                        # --------------------------
                        # CONFIDENCE BAR
                        # --------------------------

                        st.markdown(

                            "### 📊 Prediction Confidence"

                        )


                        st.progress(

                            min(
                                float(confidence) / 100,
                                1.0
                            ),

                            text=(
                                f"{confidence:.2f}%"
                            )

                        )


                        # --------------------------
                        # SAVE HISTORY
                        # --------------------------

                        st.session_state.history.append(

                            {

                                "name":
                                uploaded.name,

                                "label":
                                predicted_label,

                                "disease":
                                disease_name,

                                "confidence":
                                confidence

                            }

                        )


                        # --------------------------
                        # TOP PREDICTIONS
                        # --------------------------

                        st.markdown(

                            "### 🏆 Top Predictions"

                        )


                        sorted_predictions = sorted(

                            probabilities.items(),

                            key=lambda x: x[1],

                            reverse=True

                        )


                        for class_name, probability in (

                            sorted_predictions[:3]

                        ):


                            full_name = (

                                DISEASE_INFO.get(

                                    class_name,

                                    class_name

                                )

                            )


                            st.write(

                                f"**{class_name.upper()} "
                                f"({full_name})**"

                            )


                            st.progress(

                                min(

                                    float(probability) / 100,

                                    1.0

                                ),

                                text=(
                                    f"{probability:.2f}%"
                                )

                            )


                        # --------------------------
                        # MEDICAL DISCLAIMER
                        # --------------------------

                        st.warning(

                            "⚠️ The prediction is generated "
                            "by an AI model trained on the "
                            "HAM10000 dataset. It should not "
                            "be considered a medical diagnosis. "
                            "Consult a qualified dermatologist."
                        )


            else:


                st.info(

                    "Upload an image and click "
                    "'Analyse Image'."

                )


    # ----------------------------------------------
    # NO IMAGE
    # ----------------------------------------------

    else:


        st.info(

            "📤 Please upload an image "
            "to begin analysis."

        )


# ==================================================
# DISEASE INFORMATION PAGE
# ==================================================

elif page == "📚 Disease Information":

    st.title(

        "📚 Disease Information"

    )


    st.markdown(

        """
        The HAM10000 dataset contains seven categories
        of skin lesions.
        """

    )


    for class_name, disease_name in (

        DISEASE_INFO.items()

    ):


        with st.expander(

            f"{class_name.upper()} — {disease_name}"

        ):


            st.write(

                f"""
                **Classification Code:** {class_name.upper()}

                **Disease Name:** {disease_name}
                """

            )


# ==================================================
# ABOUT PAGE
# ==================================================

elif page == "ℹ️ About":

    st.title(

        "ℹ️ About Skin-Scan AI"

    )


    st.markdown(

        """
        ### Project Description

        Skin-Scan AI is a machine learning project
        designed to classify skin lesion images.

        ### Artificial Intelligence Model

        - Model: ResNet-18
        - Framework: PyTorch
        - Learning Type: Supervised Learning
        - Dataset: HAM10000
        - Number of Classes: 7

        ### Classification Categories

        - AKIEC
        - BCC
        - BKL
        - DF
        - MEL
        - NV
        - VASC

        ### Disclaimer

        This project is designed for educational
        and research purposes.

        It should not replace professional medical
        diagnosis.
        """

    )


# ==================================================
# SESSION HISTORY
# ==================================================

if page == "🔬 Analyse Image":


    st.divider()


    st.subheader(

        "📋 Session History"

    )


    if st.session_state.history:


        for entry in reversed(

            st.session_state.history[-5:]

        ):


            st.markdown(

                f"""
                **Image:** {entry["name"]}

                **Prediction:** {entry["label"].upper()}

                **Disease:** {entry["disease"]}

                **Confidence:** {entry["confidence"]:.2f}%

                ---
                """

            )


    else:


        st.caption(

            "No analyses performed yet."

        )


# ==================================================
# FOOTER
# ==================================================

st.divider()


st.markdown(

    """
    <div style="text-align:center; color:gray;">

    🔬 Skin-Scan AI

    <br>

    Powered by PyTorch + ResNet-18 + Streamlit

    <br>

    Educational and Research Purpose Only

    </div>
    """,

    unsafe_allow_html=True

)