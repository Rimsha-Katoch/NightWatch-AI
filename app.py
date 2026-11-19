
import io
import time

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from models.adaptive_refinement import AdaptiveRefinement


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "checkpoints/best_resunet_exp5.keras"

MODEL_HEIGHT = 256
MODEL_WIDTH = 384
SATURATION_SCALE = 1.75

st.set_page_config(
    page_title="NightWatch AI",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PROFESSIONAL DARK THEME
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --background: #0B1220;
    --sidebar: #101B2D;
    --card: #142238;
    --border: #263B55;
    --cyan: #55D6E8;
    --text: #E8EEF8;
    --muted: #AFC2D9;
}

.stApp {
    background-color: var(--background);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background-color: var(--background) !important;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: var(--sidebar) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

[data-testid="stSidebar"] h1 {
    color: var(--cyan) !important;
}

/* Sidebar toggle visibility */

[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    opacity: 1 !important;
    visibility: visible !important;
}

[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {
    background: #142B40 !important;
    color: #55D6E8 !important;
    border: 1px solid #34758B !important;
    border-radius: 8px !important;
    opacity: 1 !important;
    visibility: visible !important;
}

[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] svg {
    color: #55D6E8 !important;
    stroke: #55D6E8 !important;
    opacity: 1 !important;
}

/* Hero */

.hero-box {
    background: linear-gradient(120deg, #14243B, #10364A);
    border: 1px solid #34758B;
    border-radius: 20px;
    padding: 38px;
    margin-bottom: 30px;
}

.hero-label {
    color: #55D6E8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2.5px;
    margin-bottom: 18px;
}

.hero-title {
    color: #FFFFFF;
    font-size: clamp(28px, 4vw, 42px);
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 18px;
}

.hero-description {
    color: #D0DEEE;
    font-size: 15px;
    line-height: 1.8;
}

/* Headings and text */

h1, h2, h3 {
    color: #F1F5FB !important;
}

p, li {
    color: #C3D1E2;
}

/* Cards */

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #142238 !important;
    border: 1px solid #263B55 !important;
    border-radius: 15px !important;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(90deg, #087E9B, #13A7B8);
    color: #FFFFFF !important;
    border: 1px solid #13A7B8;
    border-radius: 10px;
    min-height: 47px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #16AFC0;
    border-color: #55D6E8;
}

.stDownloadButton > button {
    background: #142B40;
    color: #55D6E8 !important;
    border: 1px solid #34758B;
    border-radius: 10px;
    min-height: 47px;
    font-weight: 700;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: #142238 !important;
    border: 1px dashed #34758B !important;
    border-radius: 14px !important;
    padding: 12px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #142238 !important;
    border: 1px dashed #34758B !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #E8EEF8 !important;
}

/* Images */

[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #263B55;
}

/* Metrics */

[data-testid="stMetric"] {
    background-color: #142238;
    border: 1px solid #263B55;
    padding: 16px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #AFC2D9 !important;
}

[data-testid="stMetricValue"] {
    color: #55D6E8 !important;
}


/* Success and warning messages */

[data-testid="stAlert"] {
    background: #12382F !important;
    border: 1px solid #24866A !important;
    border-left: 4px solid #36D6A0 !important;
    border-radius: 12px !important;
}

[data-testid="stAlert"] * {
    color: #E6FFF5 !important;
}

[data-testid="stAlert"] p {
    color: #E6FFF5 !important;
    font-weight: 600 !important;
}

[data-testid="stAlert"] svg {
    color: #36D6A0 !important;
    fill: #36D6A0 !important;
}


[data-testid="stAlert"] p {
    color: #E8EEF8 !important;
}

/* Footer */

.footer-text {
    color: #91A7C1;
    text-align: center;
    font-size: 12px;
    line-height: 1.9;
}

hr {
    border-color: #263B55;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "enhanced_image" not in st.session_state:
    st.session_state.enhanced_image = None

if "original_image" not in st.session_state:
    st.session_state.original_image = None

if "processing_time" not in st.session_state:
    st.session_state.processing_time = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None


# ============================================================
# 4. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ============================================================
# 5. SATURATION ADJUSTMENT
# ============================================================

def adjust_saturation(image, scale=SATURATION_SCALE):

    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image, 0, 1)

    image_uint8 = np.round(image * 255).astype(np.uint8)

    hsv = cv2.cvtColor(
        image_uint8,
        cv2.COLOR_RGB2HSV
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1].astype(np.float32) * scale,
        0,
        255
    ).astype(np.uint8)

    result = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB
    )

    return result.astype(np.float32) / 255.0


# ============================================================
# 6. IMAGE ENHANCEMENT PIPELINE
# ============================================================

def enhance_image(image, model):

    original_height, original_width = image.shape[:2]

    resized = cv2.resize(
        image,
        (MODEL_WIDTH, MODEL_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    normalized = resized.astype(np.float32) / 255.0

    model_input = np.expand_dims(
        normalized,
        axis=0
    )

    # Stage 1: Residual Attention U-Net
    prediction = model.predict(
        model_input,
        verbose=0
    )[0]

    prediction = np.clip(prediction, 0, 1)

    # Stage 2: Adaptive gamma correction and CLAHE
    refiner = AdaptiveRefinement()
    refined = refiner.enhance(prediction)

    # Stage 3: Saturation adjustment
    final = adjust_saturation(
        refined,
        SATURATION_SCALE
    )

    # Restore original image dimensions
    final = cv2.resize(
        final,
        (original_width, original_height),
        interpolation=cv2.INTER_CUBIC
    )

    return np.clip(final, 0, 1)


# ============================================================
# 7. HERO SECTION
# ============================================================

def show_hero():

    st.markdown("""
<div class="hero-box">
<div class="hero-label">INTELLIGENT COMPUTER VISION</div>
<div class="hero-title">See beyond the darkness.</div>
<div class="hero-description">
NightWatch AI transforms low-light images into
clearer, brighter visuals through deep learning
and adaptive image refinement.
</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 8. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.title("🌙 NightWatch")
    st.caption("AI IMAGE ENHANCEMENT SYSTEM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "✨ Image Enhancement",
            "🧠 About Model"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    with st.container(border=True):
        st.markdown("🟢 Application running")

    with st.container(border=True):
        st.markdown("🟢 Local inference mode")

    st.divider()

    st.caption("NightWatch AI | M.Tech Project")


# ============================================================
# 9. HOME PAGE
# ============================================================

if page == "🏠 Home":

    show_hero()

    st.subheader("Platform Overview")

    st.write(
        "An intelligent workflow for low-light image enhancement."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.caption("DEEP LEARNING")
            st.markdown("### ResUNet")
            st.write("Residual Attention U-Net")

    with col2:
        with st.container(border=True):
            st.caption("IMAGE REFINEMENT")
            st.markdown("### Adaptive")
            st.write("Gamma correction + CLAHE")

    with col3:
        with st.container(border=True):
            st.caption("INFERENCE")
            st.markdown("### Local AI")
            st.write("Runs on your computer")

    st.divider()

    st.subheader("How NightWatch Works")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):

            st.markdown("### 01 · Image Input")
            st.write(
                "Upload a low-light photograph for processing."
            )

            st.markdown("### 02 · AI Enhancement")
            st.write(
                "The trained Residual Attention U-Net "
                "predicts an enhanced image."
            )

    with col2:
        with st.container(border=True):

            st.markdown("### 03 · Adaptive Refinement")
            st.write(
                "Gamma correction and CLAHE refine "
                "image luminance and local contrast."
            )

            st.markdown("### 04 · Final Output")
            st.write(
                "Controlled saturation adjustment "
                "produces the final enhanced image."
            )

    st.divider()

    # Native Streamlit banner: no raw HTML
    with st.container(border=True):

        st.markdown(
            "<h3 style='color:#55D6E8;'>"
            "✨ Ready to get started?"
            "</h3>",
            unsafe_allow_html=True
        )

        st.write(
            "Select Image Enhancement from the sidebar to upload "
            "and process a low-light photograph using NightWatch AI."
        )


# ============================================================
# 10. IMAGE ENHANCEMENT PAGE
# ============================================================

elif page == "✨ Image Enhancement":

    show_hero()

    st.subheader("Image Enhancement Workspace")

    st.write(
        "Upload a low-light photograph and let NightWatch AI "
        "improve its visibility."
    )

    uploaded_file = st.file_uploader(
        "Choose a low-light photograph",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )

    if uploaded_file is not None:

        try:

            original_pil = Image.open(
                uploaded_file
            ).convert("RGB")

            original = np.array(original_pil)

            height, width = original.shape[:2]

            # Clear previous results when a new file is uploaded
            if st.session_state.uploaded_name != uploaded_file.name:

                st.session_state.enhanced_image = None
                st.session_state.original_image = None
                st.session_state.processing_time = None

                st.session_state.uploaded_name = uploaded_file.name

            st.divider()

            st.subheader("Image Information")

            col1, col2, col3 = st.columns(3)

            col1.metric("Image Width", f"{width} px")
            col2.metric("Image Height", f"{height} px")

            col3.metric(
                "File Size",
                f"{uploaded_file.size / 1024:.1f} KB"
            )

            st.subheader("Original Image")

            st.image(
                original,
                use_container_width=True
            )

            st.divider()

            if st.button(
                "✨ Enhance Image with NightWatch AI",
                use_container_width=True,
                type="primary"
            ):

                try:

                    start_time = time.perf_counter()

                    with st.spinner(
                        "NightWatch AI is processing your image..."
                    ):

                        model = load_model()

                        enhanced = enhance_image(
                            original,
                            model
                        )

                    elapsed = time.perf_counter() - start_time

                    st.session_state.original_image = original.copy()

                    st.session_state.enhanced_image = np.round(
                        enhanced * 255
                    ).astype(np.uint8)

                    st.session_state.processing_time = elapsed

                except Exception as error:

                    st.error(
                        f"Enhancement failed: {error}"
                    )

            # Display results
            if st.session_state.enhanced_image is not None:

                with st.container(border=True):
                    st.markdown(
                        """
                        <div style="
                            display: flex;
                            align-items: center;
                            gap: 12px;
                            padding: 8px 4px;
                        ">
                            <span style="
                                background: #123F38;
                                color: #43E6B0;
                                border: 1px solid #24866A;
                                border-radius: 50%;
                                width: 36px;
                                height: 36px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                font-size: 19px;
                                font-weight: bold;
                                flex-shrink: 0;
                            ">✓</span>
                            <div>
                                <div style="
                                    color: #55D6E8;
                                    font-size: 16px;
                                    font-weight: 700;
                                ">Enhancement Complete</div>
                                <div style="
                                    color: #D0DEEE;
                                    font-size: 13px;
                                    margin-top: 4px;
                                ">Your image has been successfully processed by NightWatch AI.</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.divider()

                st.subheader("Enhancement Results")

                st.metric(
                    "Processing Time",
                    f"{st.session_state.processing_time:.2f} seconds"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown("**ORIGINAL IMAGE**")

                    st.image(
                        st.session_state.original_image,
                        use_container_width=True
                    )

                with col2:

                    st.markdown("**NIGHTWATCH ENHANCED IMAGE**")

                    st.image(
                        st.session_state.enhanced_image,
                        use_container_width=True
                    )

                st.divider()

                st.subheader("Download Enhanced Image")

                output_buffer = io.BytesIO()

                Image.fromarray(
                    st.session_state.enhanced_image
                ).save(
                    output_buffer,
                    format="PNG"
                )

                st.download_button(
                    label="⬇️ Download Enhanced Image",
                    data=output_buffer.getvalue(),
                    file_name="NightWatch_Enhanced.png",
                    mime="image/png",
                    use_container_width=True
                )

        except Exception as error:

            st.error(
                f"Unable to read the uploaded image: {error}"
            )

    else:

        with st.container(border=True):

            st.markdown("### 📷 Ready to enhance your images?")

            st.write(
                "Upload a photograph to start the NightWatch "
                "AI enhancement pipeline."
            )

            st.caption(
                "Supported formats: JPG, JPEG and PNG"
            )


# ============================================================
# 11. ABOUT MODEL PAGE
# ============================================================

elif page == "🧠 About Model":

    show_hero()

    st.subheader("Model Architecture")

    with st.container(border=True):

        st.markdown("### Residual Attention U-Net")

        st.write(
            "NightWatch uses a custom Residual Attention U-Net "
            "trained for paired low-light image enhancement."
        )

        st.markdown("""
- Residual convolutional blocks
- Attention gates for skip connections
- Encoder-decoder architecture
- Residual image correction output
- 2,082,423 model parameters
""")

    st.subheader("Adaptive Refinement")

    with st.container(border=True):

        st.markdown("### Gamma Correction")

        st.write(
            "Adjusts image luminance using an image-dependent gamma value."
        )

        st.markdown("### CLAHE")

        st.write(
            "Improves local contrast through adaptive histogram equalization."
        )

        st.markdown("### Saturation Adjustment")

        st.write(
            "Applies a controlled HSV saturation multiplier of 1.75."
        )

    st.subheader("Inference Configuration")

    with st.container(border=True):

        st.code("""
Input image: RGB
Model input: 256 x 384 x 3
Architecture: Residual Attention U-Net
Checkpoint: best_resunet_exp5.keras
Refinement: Adaptive Gamma + CLAHE
Saturation multiplier: 1.75
""", language="text")


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.markdown("""
<div class="footer-text">
🌙 NightWatch AI · Intelligent Low-Light Image Enhancement
<br><br>
Powered by Deep Learning and Adaptive Image Processing
</div>
""", unsafe_allow_html=True)
