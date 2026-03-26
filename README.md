# xai_thesis_app

A Streamlit-based Explainable AI application for image analysis using a pretrained torchvision ResNet50 model on ImageNet classes.

## What The App Does

- Loads official pretrained `ResNet50` weights from `torchvision`
- Applies official inference transforms from the selected weights
- Predicts the top ImageNet class for an uploaded image
- Shows confidence and top-5 classes
- Supports multiple explainability methods:
  - `Grad-CAM`
  - `Integrated Gradients`
  - `Occlusion`
  - `LIME` (SLIC superpixels)
- Generates heatmap and overlay visualization for the selected method
- Computes per-image XAI metrics dashboard (inspired by your Colab protocol):
  - `Faithfulness`: Deletion AUC, Insertion AUC, AOPC
  - `Sensitivity`: Drop Top, Drop Random Mean, Sensitivity score
  - `Sparsity`: Hoyer sparsity on superpixel scores
  - `Robustness` (optional): Spearman + IoU@Top-K on noisy re-explanations

## Run The Project

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The first run downloads official ResNet50 pretrained weights.
- `LIME` can be slower than other methods; reduce `LIME samples` in the UI for faster runs.
- `Robustness` metrics are optional and slower because they require one extra explanation run.

## Deploy For Public Access (Streamlit Community Cloud)

1. Push this folder to your GitHub repository.
2. Go to https://share.streamlit.io/
3. Click **New app** and select:
   - Repository: your repo
   - Branch: your branch
   - Main file path: `app.py`
4. Click **Deploy**.

After deployment, anyone with the app URL can open and use it.
