import streamlit as st

st.set_page_config(page_title="XAI Thesis App", layout="wide")

st.title("Explainable AI App for Image Analysis")
st.write(
    "This application uses a pretrained ResNet50 model on ImageNet to classify images "
    "and explain predictions with Grad-CAM, Integrated Gradients, Occlusion, and LIME visualizations."
)
st.info("Open the `Single Image Analysis` page from the sidebar to start.")
