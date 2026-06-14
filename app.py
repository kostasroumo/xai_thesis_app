import streamlit as st

st.set_page_config(page_title="XAI Thesis App", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Manrope:wght@400;500;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: "Manrope", sans-serif;
    }

    h1, h2, h3 {
        font-family: "Fraunces", serif;
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 1.6rem;
    }

    .xai-home-hero {
        background:
            radial-gradient(circle at top left, rgba(229, 168, 94, 0.18), transparent 34%),
            linear-gradient(135deg, #f6f1e8 0%, #fffaf3 48%, #f3ede4 100%);
        border: 1px solid rgba(116, 84, 49, 0.14);
        border-radius: 24px;
        padding: 1.2rem 1.25rem;
        box-shadow: 0 14px 30px rgba(87, 64, 37, 0.08);
        color: #241d17;
        margin-bottom: 1.15rem;
    }

    .xai-home-hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.4fr) minmax(240px, 0.9fr);
        gap: 1rem;
        align-items: start;
    }

    .xai-home-kicker {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.76rem;
        color: #9a5a2a;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }

    .xai-home-title {
        font-family: "Fraunces", serif;
        font-size: 1.95rem;
        line-height: 1.08;
        margin: 0 0 0.5rem 0;
        max-width: 16ch;
    }

    .xai-home-copy {
        font-size: 1rem;
        line-height: 1.62;
        max-width: 66ch;
        color: #3b3027;
        margin: 0;
    }

    .xai-home-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.9rem;
        margin-top: 1rem;
    }

    .xai-home-side {
        display: grid;
        gap: 0.75rem;
    }

    .xai-home-pill {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(116, 84, 49, 0.12);
        border-radius: 16px;
        padding: 0.85rem 0.9rem;
    }

    .xai-home-pill strong {
        display: block;
        font-size: 0.92rem;
        color: #2a211a;
        margin-bottom: 0.18rem;
    }

    .xai-home-pill span {
        display: block;
        color: #5d4d40;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .xai-home-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(116, 84, 49, 0.12);
        border-radius: 18px;
        padding: 1rem;
        color: #241d17;
    }

    .xai-home-card h4 {
        margin: 0 0 0.45rem 0;
        font-size: 1rem;
    }

    .xai-home-card p {
        margin: 0;
        line-height: 1.5;
        color: #57493d;
        font-size: 0.95rem;
    }
    @media (max-width: 900px) {
        .xai-home-hero-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="xai-home-hero">
        <div class="xai-home-hero-grid">
            <div>
                <div class="xai-home-kicker">Interactive Demo</div>
                <div class="xai-home-title">Explore how image explainers justify a prediction.</div>
                <p class="xai-home-copy">
                    This app uses the official pretrained ResNet50 on ImageNet and lets you inspect one image at a time
                    with Grad-CAM, Integrated Gradients, Occlusion, and LIME. It is designed as a guided demo: upload an
                    image, choose an explainer, then inspect the visual explanation and per-image metrics.
                </p>
            </div>
            <div class="xai-home-side">
                <div class="xai-home-pill">
                    <strong>Visual explanations</strong>
                    <span>Overlay and heatmap views for one uploaded image at a time.</span>
                </div>
                <div class="xai-home-pill">
                    <strong>Per-image metrics</strong>
                    <span>Faithfulness, sensitivity, sparsity, robustness, and runtime.</span>
                </div>
                <div class="xai-home-pill">
                    <strong>Method comparison</strong>
                    <span>Compare multiple explainers on the same prediction inside one run.</span>
                </div>
            </div>
        </div>
        <div class="xai-home-grid">
            <div class="xai-home-card">
                <h4>1. Upload an image</h4>
                <p>Use the <strong>Single Image Analysis</strong> page to load one image and start a run.</p>
            </div>
            <div class="xai-home-card">
                <h4>2. Inspect the explanation</h4>
                <p>Switch between overlay and heatmap views, and compare multiple explainers on the same input.</p>
            </div>
            <div class="xai-home-card">
                <h4>3. Read the metrics</h4>
                <p>Review faithfulness, sensitivity, sparsity, robustness, runtime, and the superpixel-based summary.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info("Open the `Single Image Analysis` page from the sidebar to start the demo.")
st.caption(
    "Model context: this dashboard analyzes the behavior of an ImageNet-pretrained ResNet50. "
    "If you upload Oxford-IIIT Pet images, the explanations still describe the ImageNet model's "
    "prediction behavior on those inputs, not a dedicated Oxford pet-breed classifier."
)
