import streamlit as st

st.set_page_config(page_title="XAI Thesis App", layout="wide")

st.title("Explainable AI App for Image Analysis")
st.write(
    "This application uses the official pretrained ResNet50 model on ImageNet and explains its "
    "predictions with Grad-CAM, Integrated Gradients, Occlusion, and LIME."
)
st.info("Open the `Single Image Analysis` page from the sidebar to start.")
st.caption(
    "Research context: this dashboard analyzes the behavior of an ImageNet-pretrained model. "
    "If you upload Oxford-IIIT Pet images, the explanations still describe the ImageNet model's "
    "decision process, not a dedicated Oxford pet-breed classifier."
)
